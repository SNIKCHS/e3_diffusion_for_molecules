"""Attention layers (some modules are copied from https://github.com/Diego999/pyGAT."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_gaussian(x,h):
    molecule = x*x
    demominator = 2*h*h
    left = 1/(math.sqrt(2*math.pi)*h)
    return left * torch.exp(-molecule/demominator )
class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Sequential(
            nn.Linear(2 * in_features+1, 2 * in_features, bias=True),
            nn.SiLU(),
            nn.Linear(2 * in_features, in_features, bias=True),
            nn.SiLU(),
        )
        self.att_mlp = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid())
        self.h_gauss = nn.Parameter(torch.Tensor(1))
        self.in_features = in_features

    def forward (self, x, x_tangent_self, distances, edge_mask):
        """
        Parameters
        ----------
        x (b*n_node*n_node,dim)
        x_tangent_self (b*n_node*n_node,dim)
        distances (b*n_node*n_node,1)
        edge_mask (b*n_node*n_node,1)

        Returns
        -------
        """
        #prepare gauss kernel distance

        gauss_dist = calc_gaussian(distances,F.softplus(self.h_gauss)) * edge_mask

        x_left = x  # (b,n_node,n_node,dim)
        x_right = x_tangent_self  # (b*n_node*n_node,dim)

        x_cat = torch.concat((x_left, x_right,gauss_dist), dim=1)  # (b*n*n,2*dim+1)

        mij = self.linear(x_cat)  # (b*n_node*n_node,dim)

        att = self.att_mlp(mij)  # (b*n_node*n_node,1)

        att_adj = mij * att  # (b*n_node*n_node,dim)

        return att_adj * edge_mask





