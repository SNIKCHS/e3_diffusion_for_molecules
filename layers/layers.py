"""Euclidean layers."""
import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt


def get_dim_act(args, num_layers, enc=True):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    act = getattr(nn, args.act)
    acts = [act()] * (num_layers)

    if enc:
        dims = [args.hidden_dim] * num_layers + [args.dim]  # len=args.num_layers+1
    else:
        dims = [args.hidden_dim] * (num_layers + 1)  # len=args.num_layers+1

    return dims, acts


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


class GCLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, act, edge_dim=1, normalization_factor=1,
                 aggregation_method='sum', msg_transform=True, sum_transform=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.att = DenseAtt(out_dim, dropout, edge_dim=edge_dim)
        self.msg_transform = msg_transform
        self.sum_transform = sum_transform
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
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, input):
        h, edge_attr, edges, node_mask, edge_mask = input

        h = self.linear(h)
        h = self.Agg(h, edge_attr, edges, edge_mask)
        h = self.ln(h)

        output = (h, edge_attr, edges, node_mask, edge_mask)
        return output

    def Agg(self, x, distances, edges, edge_mask):
        row, col = edges  # 0,0,0...0,1 0,1,2..,0
        if self.msg_transform:
            x_ = self.msg_net(x)
        else:
            x_ = x
        att = self.att(x, distances, edges, edge_mask)  # (b*n_node*n_node,dim)
        msg = x_[col] * att
        msg = unsorted_segment_sum(msg, row, num_segments=x.size(0),  # num_segments=b*n_nodes
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)  # sum掉第二个n_nodes (b*n_nodes*n_nodes,dim)->(b*n_nodes,dim)
        if self.sum_transform:
            msg = self.out_net(msg)
        x = x + msg
        return x


class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1)
        return probs


'''
InnerProductDecdoer implemntation from:
https://github.com/zfjsail/gae-pytorch/blob/master/gae/model.py
'''


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout=0, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, emb_in, emb_out):
        cos_dist = emb_in * emb_out
        probs = self.act(cos_dist.sum(1))
        return probs
