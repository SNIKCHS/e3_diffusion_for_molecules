"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt


def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers)  # len=args.num_layers
    # dims = [args.dim] * (args.num_layers + 1)  # len=args.num_layers+1
    dims = [args.dim] + [args.hidden_dim] * (args.num_layers - 1) + [args.dim]  # len=args.num_layers+1
    # dims = [args.dim] + [args.hidden_dim] * (args.num_layers)  # len=args.num_layers+1

    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = nn.ParameterList([nn.Parameter(torch.Tensor([1.])) for _ in range(args.num_layers + 1)])
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]).to(args.device) for _ in range(args.num_layers + 1)]

    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias):
        super(HNNLayer, self).__init__()

        self.norm = HypNorm(manifold, in_features, c_in)
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, h):
        # print('h:', torch.any(torch.isnan(h)).item())
        h = self.norm(h)
        h = self.linear.forward(h)
        # print('linear:', torch.any(torch.isnan(h)).item())
        h = self.hyp_act.forward(h)
        # print('hyp_act:', torch.any(torch.isnan(h)).item())
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, local_agg, edge_dim=1):
        super(HyperbolicGraphConvolution, self).__init__()
        self.norm = HypNorm(manifold, in_features, c_in)
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, local_agg=local_agg, edge_dim=edge_dim)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        h, distances, edges, node_mask, edge_mask = input
        # if torch.any(torch.isnan(h)):
        #     print('input nan')
        h = self.norm(h)
        # if torch.any(torch.isnan(h)):
        #     print('norm0 nan')
        h = self.linear.forward(h)
        # if torch.any(torch.isnan(h)):
        #     print('linear nan')
        h = self.agg.forward(h, distances, edges, node_mask, edge_mask)
        # if torch.any(torch.isnan(h)):
        #     print(h[0:5])
        #     print('agg nan')
        h = self.hyp_act.forward(h)
        # if torch.any(torch.isnan(h)):
        #     print('act nan')
        output = (h, distances, edges, node_mask, edge_mask)

        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    input in manifold
    output in manifold
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(1, out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=0.25)
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = self.manifold.mobius_matvec(drop_weight, x, self.c)  # x先log到切空间与drop_weight相乘再exp到manifold
        # res = self.manifold.proj(res, self.c) #hyperbolid的expmap0结束后有proj

        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            # hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            # res = self.manifold.proj(res, self.c)

        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


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


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, local_agg=True, edge_dim=1):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.normalization_factor = 100
        self.aggregation_method = 'sum'
        self.att = DenseAtt(in_features, dropout, edge_dim=edge_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * in_features, in_features),
            nn.SiLU(),
            nn.Linear(in_features, in_features))

    def forward(self, x, distances, edges, node_mask, edge_mask):
        x_tangent = self.manifold.logmap0(x, c=self.c)  # (b*n_node,dim)

        row, col = edges

        if self.local_agg:
            # x_row = x[row]  # 提供切空间 (b*n_node*n_node,dim)
            # x_col = x[col]  # 要映射的向量 (b*n_node*n_node,dim)
            x_local_tangent = self.manifold.logmap(x[row], x[col], c=self.c)  # (b*n_node*n_node,dim)  x_col落在x_row的切空间
            x_local_self_tangent = self.manifold.logmap(x, x, c=self.c)  # (b*n_atom,dim)
            # if torch.any(torch.isnan(x_local_tangent)):
            #     print('x_local_tangent nan')
            # if torch.any(torch.isnan(x_local_self_tangent)):
            #     print('x_local_self_tangent nan')
            edge_feat = self.att(x_local_tangent, x_local_self_tangent[row], distances,
                                 edge_mask)  # (b*n_node*n_node,dim)
            # if torch.any(torch.isnan(edge_feat)):
            #     print('edge_feat nan')
            agg = unsorted_segment_sum(edge_feat, row, num_segments=x_tangent.size(0),  # num_segments=b*n_nodes
                                       normalization_factor=self.normalization_factor,
                                       aggregation_method=self.aggregation_method)  # sum掉第二个n_nodes (b*n_nodes*n_nodes,dim)->(b*n_nodes,dim)
            # if torch.any(torch.isnan(agg)):
            #     print('unsorted_segment_sum nan')
            agg = torch.cat([x_local_self_tangent, agg], dim=1)  # (b*n_nodes,2*dim)
            out = x_local_self_tangent + self.node_mlp(agg)  # residual connect
            # print('out', out)
            # if torch.any(torch.isnan(out)):
            #     print('out nan')
            support_t = self.manifold.proj_tan(out, x, self.c)
            # if torch.any(torch.isnan(support_t)):
            #     print('support_t nan')
            output = self.manifold.expmap(support_t, x, c=self.c)
            # if torch.any(torch.isnan(output)):
            #     print('expmap nan')
            #     print(output)
            output = self.manifold.proj(output, c=self.c)
        else:
            edge_feat = self.att(x_tangent[row], x_tangent[col], distances, edge_mask)  # (b*atom_num*atom_num,dim)
            agg = unsorted_segment_sum(edge_feat, row, num_segments=x_tangent.size(0),  # num_segments=b*n_nodes
                                       normalization_factor=self.normalization_factor,
                                       aggregation_method=self.aggregation_method)
            agg = torch.cat([x_tangent, agg], dim=1)
            out = x_tangent + self.node_mlp(agg)
            support_t = self.manifold.proj_tan0(out, self.c)
            output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    input in manifold
    output in manifold
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        out = self.manifold.expmap0(xt, c=self.c_out)
        # out = self.manifold.proj(out, c=self.c_out)
        return out

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class HypNorm(nn.Module):

    def __init__(self, manifold, in_features, c):
        super(HypNorm, self).__init__()
        self.manifold = manifold
        self.c = c
        if self.manifold.name == 'Hyperboloid':
            self.ln = nn.LayerNorm(in_features - 1)
        else:
            self.ln = nn.LayerNorm(in_features)

    def forward(self, h):
        h = self.manifold.logmap0(h, self.c)
        if self.manifold.name == 'Hyperboloid':
            h[..., 1:] = self.ln(h[..., 1:].clone())
        else:
            h = self.ln(h)
        h = self.manifold.expmap0(h, c=self.c)
        return h
