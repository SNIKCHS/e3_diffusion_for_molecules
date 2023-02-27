from torch import nn
import torch
import math
from torch.nn import init
import geoopt


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return norm, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result
class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout,edge_dim=1):
        super(DenseAtt, self).__init__()

        self.att_mlp = nn.Sequential(
            nn.Linear(2 * in_features + edge_dim, in_features, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features, 3),
            nn.Sigmoid()
        )


    def forward (self, x_left, x_right, distances, edge_mask):

        distances = distances * edge_mask
        x_cat = torch.concat((x_left, x_right,distances), dim=1)  # (b*n*n,2*dim+1)
        att = self.att_mlp(x_cat)  # (b*n_node*n_node,3)

        return att * edge_mask
class GCLayer(nn.Module):
    def __init__(self, in_features, out_features, manifold_in, manifold_out, dropout, act,edge_dim=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.Tensor(1, out_features))
        self.edge_mlp = nn.Sequential(
            nn.Linear(out_features + edge_dim, out_features),
            act,
            nn.Linear(out_features, out_features),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(out_features*2, out_features),
            act,
            nn.Linear(out_features, out_features))
        self.dropout = nn.Dropout(dropout)
        self.normalization_factor = 1
        self.aggregation_method = 'sum'
        self.att = DenseAtt(out_features,dropout=dropout, edge_dim=edge_dim)
        self.act = act
        self.ln = nn.LayerNorm(out_features - 1)
        self.reset_parameters()

    def proj_tan0(self,u):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

    def reset_parameters(self):
        # init.xavier_uniform_(self.linear.weight, gain=0.01)
        init.constant_(self.bias, 0)

    def forward(self, input):
        x, edge_attr, edges, node_mask, edge_mask = input
        x0 = x[:,0]
        x1 = x[:,1]
        x2 = x[:,2]
        x0 = self.manifold_in[0].logmap0(x0)
        x1 = self.manifold_in[1].logmap0(x1)
        x = torch.cat([x0,x1,x2], dim=-1)

        x = self.linear(x)
        x = self.proj_tan0(x)
        # print('linear:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        x = x.view(x.size(0), 3, -1)
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]

        x0 = self.manifold_in[0].expmap0(x0)
        x1 = self.manifold_in[1].expmap0(x1)
        bias = self.proj_tan0(self.bias.view(1, -1))
        bias = bias.view(1, 3, -1)
        bias0 = self.manifold_in[0].transp0(x0, bias[:, 0])
        bias1 = self.manifold_in[1].transp0(x1, bias[:, 1])

        x0 = self.manifold_in[0].expmap(x0, bias0)
        x1 = self.manifold_in[1].expmap(x1, bias1)
        x2 = x2 + bias[:, 2]

        x_tan0 = self.manifold_in[0].logmap0(x0)
        x_tan1 = self.manifold_in[1].logmap0(x1)
        x_tan = torch.cat([x_tan0,x_tan1,x2], dim=-1)
        row, col = edges  # 0,0,0...0,1 0,1,2..,0

        geodesic0 = self.manifold_in[0].dist(x0[row], x0[col], keepdim=True)
        geodesic1 = self.manifold_in[1].dist(x1[row], x1[col], keepdim=True)

        edge_attr = torch.cat([edge_attr, geodesic0,geodesic1], dim=-1)

        att = self.att(x_tan[row], x_tan[col], edge_attr, edge_mask)  # (b*n_node*n_node,3)
        att = att.view(att.size(0),3,1)
        msg0 = self.manifold_in[0].logmap(x0[row], x0[col])  # (b*n_node*n_node,dim)  x_col落在x_row的切空间
        msg1 = self.manifold_in[1].logmap(x1[row], x1[col])  # (b*n_node*n_node,dim)  x_col落在x_row的切空间
        msg2 = x2[col] - x2[row]

        msg0 = self.manifold_in[0].transp0back(x0[row],msg0)
        msg1 = self.manifold_in[1].transp0back(x1[row], msg1)

        msg = torch.cat([msg0,msg1,msg2], dim=-1)
        msg = self.edge_mlp(torch.cat([msg,edge_attr], dim=-1)).view(msg.size(0),3,-1) * att
        msg = msg.view(msg.size(0),-1)
        msg = unsorted_segment_sum(msg, row, num_segments=x.size(0),  # num_segments=b*n_nodes
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)  # sum掉第二个n_nodes (b*n_nodes*n_nodes,dim)->(b*n_nodes,dim)
        msg = torch.cat([x_tan, msg], dim=-1)
        msg = self.node_mlp(msg)
        msg = self.proj_tan0(msg)
        msg = msg.view(x.size(0), 3,-1)
        msg0 = self.manifold_in[0].transp0(x0,msg[:,0])
        msg1 = self.manifold_in[1].transp0(x1,msg[:,1])
        msg2 = msg[:, 2]
        x0 = self.manifold_in[0].expmap(x0, msg0)
        x1 = self.manifold_in[1].expmap(x1, msg1)
        x2 = x2+msg2

        x0 = self.manifold_in[0].logmap0(x0)
        x1 = self.manifold_in[1].logmap0(x1)
        x = torch.cat([x0,x1,x2], dim=-1)
        # print('agg:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        x[..., 1:] = self.ln(x[..., 1:].clone())
        x = self.act(x)
        x = self.proj_tan0(x)
        x = x.view(x.size(0), 3, -1)
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0 = self.manifold_out[0].expmap0(x0)
        x1 = self.manifold_out[1].expmap0(x1)
        x = torch.cat([x0, x1, x2], dim=-1)
        x = x.view(x.size(0), 3, -1)
        output = (x, edge_attr, edges, node_mask, edge_mask)
        return output

class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=2, act_fn=nn.SiLU(), tanh=False, coords_range=10.0, skip_conn=False):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        self.skip_conn = skip_conn
        input_edge = hidden_nf * 2 + edges_in_d
        if skip_conn:
            input_edge = hidden_nf * 4 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)

        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, x, edge_index, coord_diff, edge_attr, edge_mask,h_skip=None):
        row, col = edge_index
        h = h.view(h.size(0),-1)
        if self.skip_conn:
            input_tensor = torch.cat([h[row], h[col], edge_attr,h_skip[row], h_skip[col]], dim=1)
        else:
            input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)

        trans = coord_diff * self.coord_mlp(input_tensor)

        if edge_mask is not None:
            trans = trans * edge_mask

        agg = unsorted_segment_sum(trans, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)

        x = x + agg
        return x

    def forward(self, h, x, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None,h_skip=None):
        x = self.coord_model(h, x, edge_index, coord_diff, edge_attr, edge_mask,h_skip)
        if node_mask is not None:
            x = x * node_mask
        return x


class EquivariantBlock(nn.Module):
    def __init__(self, input_nf, hidden_nf, output_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2,
                 attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum', hyp=False,
                 manifold0_in=None,manifold0_out=None,
                 manifold1_in=None,manifold1_out=None,
                 skip_conn=False):
        super(EquivariantBlock, self).__init__()
        self.manifold0_in = manifold0_in
        self.manifold0_out = manifold0_out
        self.manifold1_in = manifold1_in
        self.manifold1_out = manifold1_out
        self.input_nf = input_nf
        self.skip_conn = skip_conn
        if skip_conn:
            self.input_nf = self.input_nf*2
            if hyp:
                self.input_nf=self.input_nf-1
        self.hidden_nf = hidden_nf
        self.output_nf = output_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.hyp = hyp

        self.gcl = GCLayer(self.input_nf, self.output_nf,[manifold0_in,manifold1_in], [manifold0_out,manifold1_out], 0, act_fn, edge_dim=edge_feat_nf)
        self.gcl_equiv = EquivariantUpdate(output_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method, skip_conn=skip_conn)

        # self.apply(weight_init)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None,h_skip=None):
        # edge_index:list[rol,col] (b*n_nodes*n_nodes,)
        # print(self.manifold0_in.k.item(), ' : ', self.manifold1_in.k.item())
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        edge_attr = torch.cat([distances, edge_attr], dim=1)  # (b*n_nodes*n_nodes,2) edge_attr是最初的distance
        input = h, edge_attr, edge_index, node_mask, edge_mask
        output = self.gcl(input)
        h,edge_attr,_,_,_ = output
        h_in= h.clone()
        h_in[:, 0] = self.manifold0_out.logmap0(h[:, 0])
        h_in[:, 1] = self.manifold1_out.logmap0(h[:, 1])
        x = self.gcl_equiv(h_in, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask,h_skip)
        # print('h:', torch.max(h.view(-1)), torch.min(h.view(-1)))
        return h, x
    def proj_tan0(self,u):

        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight,gain=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
class MCNet(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 normalization_factor=100, aggregation_method='sum', hyp=False):
        super(MCNet, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf

        self.input_nf = [126, 126, 126, 126, 126, 126,126, 126, 126]
        self.hidden_nf = [126, 126, 126, 126, 126, 126,126, 126, 126]
        self.out_nf = [126, 126, 126, 126, 126, 126,126, 126, 126] #1655103.0
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range / n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.hyp = hyp

        edge_feat_nf = 4
        self.embedding = nn.Linear(in_node_nf, 126)

        self.manifolds0 = [geoopt.Lorentz(1, learnable=True) for _ in range(n_layers + 1)]
        self.manifolds1 = [geoopt.Stereographic(1, learnable=True) for _ in range(n_layers + 1)]
        self.embedding_out = nn.Linear(self.out_nf[-1], out_node_nf)
        for i in range(0, n_layers):
            skip_conn = False
            if i > 4:
                skip_conn = False
            self.add_module("e_block_%d" % i, EquivariantBlock(self.input_nf[i],self.hidden_nf[i],self.out_nf[i],
                                                               edge_feat_nf=edge_feat_nf, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method, hyp=hyp,
                                                               manifold0_in=self.manifolds0[i],manifold0_out=self.manifolds0[i+1],
                                                               manifold1_in=self.manifolds1[i],manifold1_out=self.manifolds1[i + 1]
                                                               ,skip_conn=skip_conn))

        self.apply(weight_init)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, t=None):

        distances, _ = coord2diff(x, edge_index)  # (b*n_node*n_node,1)

        emb = self.embedding(h)
        emb = self.proj_tan0(emb).view(-1,3,42)
        emb0 = self.manifolds0[0].expmap0(emb[:, 0])
        emb1 = self.manifolds1[0].expmap0(emb[:, 1])
        emb = torch.cat([emb0,emb1,emb[:, 2]],dim=-1).view(-1,3,42)
        # print('emb:', torch.max(emb.view(-1)), torch.min(emb.view(-1)))
        h0, x = self._modules["e_block_0"](emb, x, edge_index,node_mask,edge_mask,distances)
        h1, x = self._modules["e_block_1"](h0, x, edge_index, node_mask, edge_mask, distances)
        h2, x = self._modules["e_block_2"](h1, x, edge_index, node_mask, edge_mask, distances)
        h3, x = self._modules["e_block_3"](h2, x, edge_index, node_mask, edge_mask, distances)
        h4, x = self._modules["e_block_4"](h3, x, edge_index, node_mask, edge_mask, distances)
        h5, x = self._modules["e_block_5"](h4, x, edge_index, node_mask, edge_mask, distances)
        h6, x = self._modules["e_block_6"](h5, x, edge_index, node_mask, edge_mask, distances)
        h7, x = self._modules["e_block_7"](h6, x, edge_index, node_mask, edge_mask, distances)
        h8, x = self._modules["e_block_8"](h7, x, edge_index, node_mask, edge_mask, distances)

        h = torch.empty(h8.size(),device=h8.device)
        h[:, 0] = self.manifolds0[-1].logmap0(h8[:, 0])
        h[:, 1] = self.manifolds1[-1].logmap0(h8[:, 1])
        h = h.view(h.size(0),-1)
        h = self.embedding_out(h)
        h = self.proj_tan0(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x

    def proj_tan0(self, u):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals
