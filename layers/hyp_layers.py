"""Hyperbolic layers."""
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from geoopt import PoincareBall
from geoopt import Lorentz
# from manifolds import Lorentz
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt


def get_dim_act_curv(args, num_layers, enc=True):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """

    act = getattr(nn, args.act)
    acts = [act()] * (num_layers)  # len=args.num_layers
    if enc:
        dims = [args.hidden_dim] * num_layers + [args.dim]  # len=args.num_layers+1
    else:
        dims = [args.hidden_dim] * (num_layers+1)  # len=args.num_layers+1

    manifold_class = {'PoincareBall': PoincareBall, 'Lorentz': Lorentz}

    if args.c is None:
        manifolds = [manifold_class[args.manifold](1, learnable=True) for _ in range(num_layers + 1)]
    else:
        manifolds = [manifold_class[args.manifold](args.c, learnable=False) for _ in range(num_layers + 1)]

    return dims, acts, manifolds


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout, act, edge_dim=2):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(in_dim, out_dim, manifold_in, dropout)
        self.hyp_act = HypAct(manifold_in, manifold_out, act)

    def forward(self, x):
        x = self.linear(x)
        x = self.hyp_act(x)
        return x


class HGCLayer(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout, act, edge_dim=2, normalization_factor=1,
                 aggregation_method='sum', msg_transform=True, sum_transform=True,use_norm='ln'):
        super(HGCLayer, self).__init__()
        self.linear = HypLinear(in_dim, out_dim, manifold_in, dropout)
        self.agg = HypAgg(
            out_dim, manifold_in, dropout, edge_dim, normalization_factor, aggregation_method, act, msg_transform,
            sum_transform
        )
        self.use_norm = use_norm
        if use_norm != 'none':
            self.norm = HypNorm(out_dim, manifold_in,use_norm)
        self.hyp_act = HypAct(manifold_in, manifold_out, act)

    def forward(self, input):
        x, edge_attr, edges, node_mask, edge_mask = input
        x = self.linear(x)
        # print('linear:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        # if self.use_norm != 'none':
        #     x = self.norm(x)
        x = self.agg(x, edge_attr, edges, edge_mask)
        # print('agg:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        if self.use_norm != 'none':
            x = self.norm(x)
            # print('norm:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        x = self.hyp_act(x)
        # print('HypAct:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        output = (x, edge_attr, edges, node_mask, edge_mask)
        return output

class HGCLayerV1(nn.Module):
    """
    HGNN
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout, act, edge_dim=2, normalization_factor=1,
                 aggregation_method='sum', msg_transform=True, sum_transform=True,use_norm='ln'):
        super(HGCLayerV1, self).__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.linear = nn.Linear(in_dim, out_dim)
        self.att_net = DenseAtt(out_dim, dropout=dropout, edge_dim=edge_dim)
        self.act = act
        self.edge_dim = edge_dim
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, input):
        x, edge_attr, edges, node_mask, edge_mask = input
        row, col = edges  # 0,0,0...0,1 0,1,2..,0

        geodesic = self.manifold_in.dist(x[row], x[col], keepdim=True)  # (b*n_node*n_node,dim)
        x = self.manifold_in.logmap0(x)
        if self.edge_dim == 2:
            in_edge_attr = torch.cat([edge_attr, geodesic], dim=-1)
        else:
            in_edge_attr = geodesic
        x = self.linear(x)
        att = self.att_net(x, in_edge_attr, edges, edge_mask)
        msg = x[col] * att
        msg = unsorted_segment_sum(msg, row, num_segments=x.size(0),  # num_segments=b*n_nodes
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        x = x + msg
        x = self.ln(x)
        x = proj_tan0(x,self.manifold_in)
        x = self.manifold_in.expmap0(x)
        x = self.manifold_in.to_poincare(x)
        x = self.act(x)
        x = self.manifold_in.to_lorentz(x)
        # print('x:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        output = (x, edge_attr, edges, node_mask, edge_mask)
        return output

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    input in manifold
    output in manifold
    """

    def __init__(self, in_dim, out_dim, manifold_in, dropout):
        super(HypLinear, self).__init__()
        self.manifold = manifold_in
        self.bias = nn.Parameter(torch.Tensor(1, out_dim))
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.dp = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        # init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        # print('linearin:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        x = self.manifold.logmap0(x)
        # print('linearlogmap0:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        x = self.linear(x)
        # print('linearout:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        x = self.dp(x)
        x = proj_tan0(x, self.manifold)
        x = self.manifold.expmap0(x)
        # print('linearexpmap0:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        bias = proj_tan0(self.bias.view(1, -1), self.manifold)
        bias = self.manifold.transp0(x, bias)
        x = self.manifold.expmap(x, bias)
        return x


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, out_dim, manifold_in, dropout, edge_dim, normalization_factor=1, aggregation_method='sum',
                 act=nn.SiLU(), msg_transform=True, sum_transform=True):
        super(HypAgg, self).__init__()
        self.manifold = manifold_in
        self.dim = out_dim
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.att_net = DenseAtt(out_dim, dropout=dropout, edge_dim=edge_dim)
        self.msg_transform = msg_transform
        self.sum_transform = sum_transform
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        if msg_transform:
            self.msg_net = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                act,
                nn.LayerNorm(out_dim),
                nn.Linear(out_dim, out_dim)
            )

        if sum_transform:
            self.out_net = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                act,
                nn.LayerNorm(out_dim),
                nn.Linear(out_dim, out_dim)
            )

    def forward(self, x, edge_attr, edges, edge_mask):
        x_tangent0 = self.manifold.logmap0(x)  # (b*n_node,dim)

        row, col = edges  # 0,0,0...0,1 0,1,2..,0
        x_row = x[row]
        x_col = x[col]

        geodesic = self.manifold.dist(x_row, x_col, keepdim=True)  # (b*n_node*n_node,dim)
        if self.edge_dim == 2:
            edge_attr = torch.cat([edge_attr, geodesic], dim=-1)
        else:
            edge_attr = geodesic
        att = self.att_net(x_tangent0, edge_attr, edges, edge_mask)  # (b*n_node*n_node,dim)
        msg = self.manifold.logmap(x_row, x_col)  # (b*n_node*n_node,dim)  x_col落在x_row的切空间
        if self.msg_transform:
            msg = self.manifold.transp0back(x_row, msg)
            msg = self.msg_net(msg)
        msg = msg * att
        msg = unsorted_segment_sum(msg, row, num_segments=x_tangent0.size(0),  # num_segments=b*n_nodes
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)  # sum掉第二个n_nodes (b*n_nodes*n_nodes,dim)->(b*n_nodes,dim)
        if self.sum_transform:
            msg = self.out_net(msg)
        if self.msg_transform:
            msg = proj_tan0(msg, self.manifold)
            msg = self.manifold.transp0(x, msg)
        else:
            msg = self.manifold.proju(x, msg)
        output = self.manifold.expmap(x, msg)
        return output


class HypAct(Module):
    """
    Hyperbolic activation layer.
    input in manifold
    output in manifold
    """

    def __init__(self, manifold_in, manifold_out, act):
        super(HypAct, self).__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.act = act

    def forward(self, x):
        x = self.act(self.manifold_in.logmap0(x))
        x = proj_tan0(x, self.manifold_in)
        x = self.manifold_out.expmap0(x)
        return x


class HypNorm(nn.Module):

    def __init__(self, in_features, manifold, method='ln'):
        super(HypNorm, self).__init__()
        self.manifold = manifold
        if self.manifold.name == 'Lorentz':
            in_features = in_features - 1
        if method == 'ln':
            self.norm = nn.LayerNorm(in_features)

    def forward(self, h):
        h = self.manifold.logmap0(h)
        if self.manifold.name == 'Lorentz':
            h[..., 1:] = self.norm(h[..., 1:].clone())
        else:
            h = self.norm(h)
        h = self.manifold.expmap0(h)
        return h


def proj_tan0(u, manifold):
    if manifold.name == 'Lorentz':
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals
    else:
        return u


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
