"""Attention layers (some modules are copied from https://github.com/Diego999/pyGAT."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# def calc_gaussian(x,h):
#     molecule = x*x
#     demominator = 2*h*h
#     left = 1/(math.sqrt(2*math.pi)*h)
#     return left * torch.exp(-molecule/demominator)
class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout,edge_dim=1):
        super(DenseAtt, self).__init__()
        # self.att_mlp = nn.Sequential(
        #     nn.Linear(2 * in_features + edge_dim, 2 * in_features, bias=True),
        #     nn.SiLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(2 * in_features, in_features, bias=True),
        #     nn.SiLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(in_features, 1),
        #     nn.Sigmoid()
        # )
        # self.h_gauss = nn.Parameter(torch.Tensor(1))
        self.att_mlp = nn.Sequential(
            nn.Linear(2 * in_features + edge_dim, in_features, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )


    def forward (self, x_left, x_right, distances, edge_mask):
        """
        Parameters
        ----------
        x_left (b*n_node*n_node,dim)
        x_right (b*n_node*n_node,dim)
        distances (b*n_node*n_node,edge_dim)
        edge_mask (b*n_node*n_node,1)

        Returns
        -------
        """

        distances = distances * edge_mask
        x_cat = torch.concat((x_left, x_right,distances), dim=1)  # (b*n*n,2*dim+1)
        att = self.att_mlp(x_cat)  # (b*n_node*n_node,1)

        return att * edge_mask





