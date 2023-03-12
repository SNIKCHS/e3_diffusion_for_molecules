from torch import nn
import torch
import math
from torch.nn import init
import geoopt
from AutoEncoder.CentroidDistance import CentroidDistance
from layers.att_layers import DenseAtt
from layers.hyp_layers import HypLinear
from layers.layers import GCLayer


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


class HGCLayer(nn.Module):
    def __init__(self, in_features, out_features, manifold_in, manifold_out, dropout, act, edge_dim=2):
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
            nn.Linear(out_features, out_features),
            act,
            nn.Linear(out_features, out_features))
        self.dropout = nn.Dropout(dropout)
        self.normalization_factor = 1
        self.aggregation_method = 'sum'
        self.att = DenseAtt(out_features, dropout=dropout, edge_dim=edge_dim)
        self.act = act
        if self.manifold_in.name == 'Lorentz':
            self.ln = nn.LayerNorm(out_features - 1)
        else:
            self.ln = nn.LayerNorm(out_features)
        self.reset_parameters()

    def proj_tan0(self, u):
        if self.manifold_in.name == 'Lorentz':
            narrowed = u.narrow(-1, 0, 1)
            vals = torch.zeros_like(u)
            vals[:, 0:1] = narrowed
            return u - vals
        else:
            return u

    def reset_parameters(self):
        # init.xavier_uniform_(self.linear.weight, gain=0.01)
        init.constant_(self.bias, 0)

    def forward(self, input):
        x, edge_attr, edges, node_mask, edge_mask = input
        x = self.manifold_in.logmap0(x)
        x = self.linear(x)
        x = self.proj_tan0(x)
        x = self.manifold_in.expmap0(x)
        bias = self.proj_tan0(self.bias.view(1, -1))
        bias = self.manifold_in.transp0(x, bias)
        x = self.manifold_in.expmap(x, bias)

        x_tan = self.manifold_in.logmap0(x)
        row, col = edges  # 0,0,0...0,1 0,1,2..,0

        geodesic = self.manifold_in.dist(x[row], x[col], keepdim=True)  # (b*n_node*n_node,dim)
        edge_attr = torch.cat([edge_attr, geodesic], dim=-1)
        att = self.att(x_tan[row], x_tan[col], edge_attr, edge_mask)  # (b*n_node*n_node,dim)
        x_local_tangent = self.manifold_in.logmap(x[row], x[col])  # (b*n_node*n_node,dim)  x_col落在x_row的切空间
        x_local_tangent = self.manifold_in.transp0back(x[row], x_local_tangent)
        agg = self.edge_mlp(torch.cat([x_local_tangent, edge_attr], -1)) * att
        agg = unsorted_segment_sum(agg, row, num_segments=x.size(0),  # num_segments=b*n_nodes
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)  # sum掉第二个n_nodes (b*n_nodes*n_nodes,dim)->(b*n_nodes,dim)
        # agg = torch.cat([x_tan, agg], dim=1)
        out = self.node_mlp(agg)
        support_t = self.proj_tan0(out)
        support_t = self.manifold_in.transp0(x, support_t)
        x = self.manifold_in.expmap(x, support_t)  # 类似于残差连接 x+support_t
        x = self.manifold_in.logmap0(x)
        if self.manifold_in.name == 'Lorentz':
            x[..., 1:] = self.ln(x[..., 1:].clone())
        else:
            x = self.ln(x)
        x = self.act(x)
        x = self.proj_tan0(x)
        x = self.manifold_out.expmap0(x)
        output = (x, edge_attr, edges, node_mask, edge_mask)
        return output


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=2, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range

        input_edge = hidden_nf * 2 + edges_in_d
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

    def coord_model(self, h, x, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index

        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)

        trans = coord_diff * self.coord_mlp(input_tensor)

        if edge_mask is not None:
            trans = trans * edge_mask

        agg = unsorted_segment_sum(trans, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)

        x = x + agg
        return x

    def forward(self, h, x, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        x = self.coord_model(h, x, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            x = x * node_mask
        return x


class EquivariantBlock(nn.Module):
    def __init__(self, input_nf, hidden_nf, output_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2,
                 attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum', hyp=False, manifold_in=None, manifold_out=None,
                 skip_conn=False):
        super(EquivariantBlock, self).__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.input_nf = input_nf
        self.skip_conn = skip_conn
        if skip_conn:
            self.input_nf = self.input_nf * 2
            if hyp:
                self.input_nf = self.input_nf - 1
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
        if hyp:
            self.gcl = HGCLayer(self.input_nf, self.output_nf, manifold_in, manifold_out, 0, act_fn,
                                edge_dim=edge_feat_nf)
            self.apply(weight_init)
        else:
            self.gcl = GCLayer(self.input_nf, self.output_nf, 0, act_fn, edge_dim=edge_feat_nf)
        self.centroid = CentroidDistance(output_nf,output_nf,self.manifold_out)
        self.gcl_equiv = EquivariantUpdate(output_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                           coords_range=self.coords_range_layer,
                                           normalization_factor=self.normalization_factor,
                                           aggregation_method=self.aggregation_method)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None, h_skip=None):
        # edge_index:list[rol,col] (b*n_nodes*n_nodes,)

        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        edge_attr = torch.cat([distances, edge_attr], dim=1)  # (b*n_nodes*n_nodes,2) edge_attr是最初的distance
        if self.skip_conn:
            if self.hyp:
                h_in = self.manifold_in.logmap0(h)
                h_skip_in = self.manifold_in.logmap0(h_skip)
                h_in = torch.cat([h_in, h_skip_in[..., 1:]], dim=-1)
                h_in = self.proj_tan0(h_in)
                h_in = self.manifold_in.expmap0(h_in)
                # h_in = torch.cat([h, h_skip[...,1:]], dim=-1)
                # h_in = self.manifold_in.projx(h_in)
            else:
                h_in = torch.cat([h, h_skip], dim=-1)
        else:
            h_in = h
        input = h_in, edge_attr, edge_index, node_mask, edge_mask
        output = self.gcl(input)
        h, edge_attr, _, _, _ = output

        if self.hyp:
            h_t = self.centroid(h,node_mask)
            # h_t = self.manifold_out.logmap0(h)
        else:
            h_t = h
        x = self.gcl_equiv(h_t, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)
        return h, x

    def proj_tan0(self, u):
        if self.manifold_in.name == 'Lorentz':
            narrowed = u.narrow(-1, 0, 1)
            vals = torch.zeros_like(u)
            vals[:, 0:1] = narrowed
            return u - vals
        else:
            return u


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class UNet(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 normalization_factor=100, aggregation_method='sum', hyp=False):
        super(UNet, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        # self.input_nf =     [256, 128, 64, 32, 16, 16, 32, 64, 128]
        # self.hidden_nf =    [256, 128, 64, 32, 16, 32, 64, 128, 256]
        # self.out_nf =       [128, 64, 32, 16, 16, 32, 64, 128, 256]
        self.input_nf = [128, 128, 128, 128, 128, 128, 128, 128, 128]
        self.hidden_nf = [128, 128, 128, 128, 128, 128, 128, 128, 128]
        self.out_nf = [128, 128, 128, 128, 128, 128, 128, 128, 128]  # 1655103.0
        # concat=(0,7)(1,6)(2,5)(3,4)
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range / n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.hyp = hyp

        edge_feat_nf = 2

        if hyp:
            edge_feat_nf = 3
            self.manifolds = [geoopt.Lorentz(1, learnable=True) for _ in range(n_layers + 1)]
            # self.embedding = HypLinear(in_node_nf, self.input_nf[0], self.manifolds[0])
            self.embedding = nn.Linear(in_node_nf, self.input_nf[0])
        else:
            self.embedding = nn.Linear(in_node_nf, self.input_nf[0])
            self.manifolds = [None for _ in range(n_layers + 1)]
        self.embedding_out = nn.Linear(self.out_nf[-1], out_node_nf)
        for i in range(0, n_layers):
            skip_conn = False
            if i > 4:
                skip_conn = True
            self.add_module("e_block_%d" % i, EquivariantBlock(self.input_nf[i], self.hidden_nf[i], self.out_nf[i],
                                                               edge_feat_nf=edge_feat_nf, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method, hyp=hyp,
                                                               manifold_in=self.manifolds[i],
                                                               manifold_out=self.manifolds[i + 1]
                                                               , skip_conn=skip_conn))
        # if hyp:
        #     self.apply(weight_init)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, t=None):

        distances, _ = coord2diff(x, edge_index)  # (b*n_node*n_node,1)
        emb = self.embedding(h)  # default: (b*n_nodes,hidden_nf=128)
        if self.hyp:
            emb = self.proj_tan0(emb)
            emb = self.manifolds[0].expmap0(emb)
        h0, x = self._modules["e_block_0"](emb, x, edge_index, node_mask, edge_mask, distances)
        h1, x = self._modules["e_block_1"](h0, x, edge_index, node_mask, edge_mask, distances)
        h2, x = self._modules["e_block_2"](h1, x, edge_index, node_mask, edge_mask, distances)
        h3, x = self._modules["e_block_3"](h2, x, edge_index, node_mask, edge_mask, distances)
        h4, x = self._modules["e_block_4"](h3, x, edge_index, node_mask, edge_mask, distances)
        h5, x = self._modules["e_block_5"](h4, x, edge_index, node_mask, edge_mask, distances, h3)
        h6, x = self._modules["e_block_6"](h5, x, edge_index, node_mask, edge_mask, distances, h2)
        h7, x = self._modules["e_block_7"](h6, x, edge_index, node_mask, edge_mask, distances, h1)
        h8, x = self._modules["e_block_8"](h7, x, edge_index, node_mask, edge_mask, distances, h0)
        if self.hyp:
            h8 = self.manifolds[-1].logmap0(h8)
        h = self.embedding_out(h8)
        h = self.proj_tan0(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x

    def proj_tan0(self, u):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals
