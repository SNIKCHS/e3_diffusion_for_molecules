"""Euclidean layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from layers.att_layers import DenseAtt


def get_dim_act(args,num_layers,enc = True):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * num_layers

    # dims = [args.dim] * (args.num_layers+1)
    if enc:
        dims = [args.hidden_dim] * num_layers + [args.dim]  # len=args.num_layers+1
    else:
        dims = [args.dim] + [args.hidden_dim] * num_layers  # len=args.num_layers+1
    return dims, acts

def calc_gaussian(x,h):
    molecule = x*x
    demominator = 2*h*h
    left = 1/(math.sqrt(2*math.pi)*h)
    return left * torch.exp(-molecule/demominator )
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
class GraphConvolution(Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features
        self.h_gauss = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.normalization_factor = 100
        self.aggregation_method = 'sum'
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*in_features + 1, in_features),
            nn.SiLU(),
            nn.Linear(in_features, in_features))

        self.node_mlp = nn.Sequential(
            nn.Linear(2*in_features, in_features),
            nn.SiLU(),
            nn.Linear(in_features, in_features))

        self.att_mlp = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid())

    def forward(self, input):
        h, distances, edges, node_mask, edge_mask = input
        gauss_dist = calc_gaussian(distances, F.softplus(self.h_gauss)) * edge_mask
        row, col = edges
        mij = self.edge_mlp(torch.cat([h[row], h[col], gauss_dist], dim=1))  # (b*atom_num*atom_num,dim)
        att = self.att_mlp(mij)
        edge_feat = mij * att * edge_mask
        agg = unsorted_segment_sum(edge_feat, row, num_segments=h.size(0),  # num_segments=b*n_nodes
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        agg = torch.cat([h, agg], dim=1)  # (b*n_nodes,2*dim)
        out = h + self.node_mlp(agg)  # residual connect

        hidden = self.linear.forward(out)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        output = (hidden, distances, edges, node_mask, edge_mask)
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )
class GCLayer(nn.Module):
    def __init__(self, in_features, out_features,  dropout, act,edge_dim=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * out_features, out_features),
            act,
            nn.Linear(out_features, out_features)
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


    def forward(self, input):
        h, edge_attr, edges, node_mask, edge_mask = input
        h = self.linear(h)
        h = self.Agg(h, edge_attr, edges, node_mask, edge_mask)
        h = h * node_mask
        output = (h, edge_attr, edges, node_mask, edge_mask)
        return output

    def Agg(self, x, edge_attr, edges, node_mask, edge_mask):

        row, col = edges  # 0,0,0...0,1 0,1,2..,0

        att = self.att(x[row], x[col], edge_attr, edge_mask)  # (b*n_node*n_node,dim)
        agg = self.edge_mlp(torch.concat([x[row], x[col], edge_attr],dim=-1)) * att

        agg = unsorted_segment_sum(agg, row, num_segments=x.size(0),  # num_segments=b*n_nodes
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)  # sum掉第二个n_nodes (b*n_nodes*n_nodes,dim)->(b*n_nodes,dim)

        agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out

class Linear(Module):
    """
    Simple Linear layer with dropout.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.ln = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x):
        x = self.ln(x)
        hidden = self.linear.forward(x)
        if self.dropout is not None:
            hidden = F.dropout(hidden, self.dropout, training=self.training)
        if self.act is not None:
            hidden = self.act(hidden)
        return hidden

