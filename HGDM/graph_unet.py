from torch import nn
import torch
import math
from torch.nn import init
import geoopt

from layers.att_layers import DenseAtt


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
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb
class HGCLayer(nn.Module):
    def __init__(self, in_features, out_features, manifold_in, manifold_out, dropout, act,edge_dim=1,skip_conn=None):
        super().__init__()
        self.skip_conn = skip_conn
        if skip_conn:
            in_features = 2*in_features
        self.in_features = in_features
        self.out_features = out_features
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear1 = nn.Linear(out_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.Tensor(1, out_features))
        self.edge_mlp = nn.Sequential(
            nn.Linear(out_features, out_features),
            act,
            nn.Linear(out_features, out_features),
        )
        # self.node_mlp = nn.Sequential(
        #     nn.Linear(out_features*2, out_features),
        #     act,
        #     nn.Linear(out_features, out_features))
        self.dropout = nn.Dropout(dropout)
        self.normalization_factor = 1
        self.aggregation_method = 'sum'
        self.att = DenseAtt(out_features,dropout=dropout, edge_dim=edge_dim)
        self.act = act
        if self.manifold_in.name == 'Lorentz':
            self.ln = nn.LayerNorm(out_features - 1)
        else:
            self.ln = nn.LayerNorm(out_features)
        self.reset_parameters()

    def proj_tan0(self,u):
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

    def forward(self, x, edges, node_mask, edge_mask,temb,x_skip=None):
        # x = self.manifold_in.logmap0(x)
        if self.skip_conn:
            x = torch.cat([x,x_skip],dim=-1)
        x = self.linear(x)
        x = x+temb
        x = self.linear1(x)
        x = self.proj_tan0(x)
        x = self.manifold_in.expmap0(x)
        bias = self.proj_tan0(self.bias.view(1, -1))
        bias = self.manifold_in.transp0(x, bias)
        x = self.manifold_in.expmap(x, bias)

        x_tan = self.manifold_in.logmap0(x)
        row, col = edges  # 0,0,0...0,1 0,1,2..,0

        geodesic = self.manifold_in.dist(x[row], x[col], keepdim=True)  # (b*n_node*n_node,dim)
        edge_attr = geodesic
        att = self.att(x_tan[row], x_tan[col], edge_attr, edge_mask)  # (b*n_node*n_node,dim)
        x_local_tangent = self.manifold_in.logmap(x[row], x[col])  # (b*n_node*n_node,dim)  x_col落在x_row的切空间
        x_local_tangent = self.manifold_in.transp0back(x[row],x_local_tangent)
        agg = self.edge_mlp(torch.cat([x_local_tangent],-1)) * att
        agg = unsorted_segment_sum(agg, row, num_segments=x.size(0),  # num_segments=b*n_nodes
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)  # sum掉第二个n_nodes (b*n_nodes*n_nodes,dim)->(b*n_nodes,dim)
        # agg = torch.cat([x_tan, agg], dim=1)
        # out = self.node_mlp(agg)
        support_t = self.proj_tan0(agg)
        support_t = self.manifold_in.transp0(x,support_t)
        x = self.manifold_in.expmap(x, support_t)  # 类似于残差连接 x+support_t
        x = self.manifold_in.logmap0(x)
        if self.manifold_in.name == 'Lorentz':
            x[..., 1:] = self.ln(x[..., 1:].clone())
        else:
            x = self.ln(x)
        x = self.act(x)
        x = self.proj_tan0(x)
        # x = self.manifold_out.expmap0(x)

        return x

class GCLayer(nn.Module):
    def __init__(self, in_features, out_features,  dropout, act,edge_dim=0,skip_conn=None):
        super().__init__()
        self.skip_conn = skip_conn
        if skip_conn:
            in_features = 2*in_features
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.linear1 = nn.Linear(out_features, out_features, bias=False)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * out_features, out_features),
            act,
            nn.Linear(out_features, out_features)
        )
        self.temb_net = nn.Sequential(
            act,
            nn.Linear(128,128),
        )
        self.normalization_factor = 1
        self.aggregation_method = 'sum'
        self.att = DenseAtt(out_features,dropout=dropout, edge_dim=edge_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*out_features + edge_dim, out_features),
            act,
            nn.Linear(out_features, out_features),
            act
        )


    def forward(self, x, edges, node_mask, edge_mask,temb,x_skip=None):
        if self.skip_conn:
            x = torch.cat([x,x_skip],dim=-1)
        x = self.linear(x)
        x = x + self.temb_net(temb)
        x = self.linear1(x)
        x = self.Agg(x,  edges, node_mask, edge_mask)
        x = x * node_mask
        return x

    def Agg(self, x,  edges, node_mask, edge_mask):

        row, col = edges  # 0,0,0...0,1 0,1,2..,0

        att = self.att(x[row], x[col], None, edge_mask)  # (b*n_node*n_node,dim)
        agg = self.edge_mlp(torch.concat([x[row], x[col]],dim=-1)) * att

        agg = unsorted_segment_sum(agg, row, num_segments=x.size(0),  # num_segments=b*n_nodes
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)  # sum掉第二个n_nodes (b*n_nodes*n_nodes,dim)->(b*n_nodes,dim)

        agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight,gain=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
class UNet(nn.Module):
    def __init__(self, in_node_nf, act_fn=nn.SiLU(), n_layers=3,
                 normalization_factor=1, aggregation_method='sum', hyp=False):
        super(UNet, self).__init__()

        out_node_nf = in_node_nf
        # self.input_nf =     [256, 128, 64, 32, 16, 16, 32, 64, 128]
        # self.hidden_nf =    [256, 128, 64, 32, 16, 32, 64, 128, 256]
        # self.out_nf =       [128, 64, 32, 16, 16, 32, 64, 128, 256]
        self.input_nf = [128, 128, 128, 128, 128, 128, 128, 128, 128]
        self.out_nf = [128, 128, 128, 128, 128, 128, 128, 128, 128]  #1655103.0
        self.temb_dim = 128
        self.temb_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        # concat=(0,7)(1,6)(2,5)(3,4)
        self.n_layers = n_layers
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.hyp = hyp
        if hyp:
            self.manifolds = [geoopt.PoincareBall(0.05, learnable=True) for _ in range(n_layers + 1)]
            # self.embedding = HypLinear(in_node_nf, self.input_nf[0], self.manifolds[0])
            self.embedding = nn.Linear(in_node_nf, self.input_nf[0])
        else:
            self.embedding = nn.Linear(in_node_nf, self.input_nf[0])
            self.manifolds = [None for _ in range(n_layers + 1)]
        self.embedding_out = nn.Linear(self.out_nf[-1], out_node_nf)
        layers = []
        for i in range(0, n_layers):
            skip_conn = False
            if i > 4:
                skip_conn = True
            if hyp:
                layers.append(HGCLayer(self.input_nf[i],self.out_nf[i],manifold_in=self.manifolds[i],
                                   manifold_out=self.manifolds[i+1],dropout=0,act=act_fn
                                   ,skip_conn=skip_conn))
            else:
                layers.append(GCLayer(self.input_nf[i], self.out_nf[i],dropout=0, act=act_fn
                                       , skip_conn=skip_conn))
        self.layers = nn.ModuleList(layers)
        # if hyp:
        #     self.apply(weight_init)

    def forward(self, h, t, node_mask,edge_index, edge_mask,context=None):
        b,n_node,dim = h.size()
        h = h.view(-1,dim)
        node_mask = node_mask.view(b*n_node,1)
        temb = get_timestep_embedding(t,embedding_dim=128)
        temb = temb.repeat(1,n_node,1).view(-1,self.temb_dim)
        temb = self.temb_net(temb)
        emb = self.embedding(h)
        if self.hyp:
            # emb = self.proj_tan0(emb)
            emb = self.manifolds[0].expmap0(emb)
        h0 = self.layers[0](emb, edge_index,node_mask,edge_mask,temb)
        h1 = self.layers[1](h0, edge_index, node_mask, edge_mask,temb)
        h2 = self.layers[2](h1, edge_index, node_mask, edge_mask,temb)
        h3 = self.layers[3](h2, edge_index, node_mask, edge_mask,temb)
        h4 = self.layers[4](h3, edge_index, node_mask, edge_mask,temb)
        h5 = self.layers[5](h4, edge_index, node_mask, edge_mask,temb,h3)
        h6 = self.layers[6](h5, edge_index, node_mask, edge_mask,temb,h2)
        h7 = self.layers[7](h6, edge_index, node_mask, edge_mask,temb,h1)
        h8 = self.layers[8](h7, edge_index, node_mask, edge_mask,temb,h0)
        # if self.hyp:
        #     h8 = self.manifolds[-1].logmap0(h8)
        h = self.embedding_out(h8)
        h = self.proj_tan0(h)
        if node_mask is not None:
            h = h * node_mask
        return h

    def proj_tan0(self, u):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals
