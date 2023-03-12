from geoopt.tensor import ManifoldParameter
import torch
import torch.nn as nn

from layers.hyp_layers import HypLinear


def init_weight(weight, method):
    """
    Initialize parameters
    Args:
        weight: a Parameter object
        method: initialization method
    """
    if method == 'orthogonal':
        nn.init.orthogonal_(weight)
    elif method == 'xavier':
        nn.init.xavier_uniform_(weight)
    elif method == 'kaiming':
        nn.init.kaiming_uniform_(weight)
    elif method == 'none':
        pass
    else:
        raise Exception('Unknown init method')


def nn_init(nn_module, method='orthogonal'):
    """
    Initialize a Sequential or Module object
    Args:
        nn_module: Sequential or Module
        method: initialization method
    """
    if method == 'none':
        return
    for param_name, _ in nn_module.named_parameters():
        if isinstance(nn_module, nn.Sequential):
            # for a Sequential object, the param_name contains both id and param name
            i, name = param_name.split('.', 1)
            param = getattr(nn_module[int(i)], name)
        else:
            param = getattr(nn_module, param_name)
        if param_name.find('weight') > -1:
            init_weight(param, method)
        elif param_name.find('bias') > -1:
            nn.init.uniform_(param, -1e-4, 1e-4)


class CentroidDistance(nn.Module):
    """
    Implement a model that calculates the pairwise distances between node representations
    and centroids
    """

    def __init__(self, num_centroid, dim, manifold):
        super(CentroidDistance, self).__init__()
        self.manifold = manifold
        self.num_centroid = num_centroid
        self.centroid_embedding = nn.Embedding(num_centroid,dim)
        self.linear = HypLinear(dim,dim,manifold)
        # self.bias = nn.Parameter(torch.Tensor(1, out_features))

    def forward(self, node_repr, mask):

        node_num,embed_size = node_repr.size()

        # broadcast and reshape node_repr to [node_num * num_centroid, embed_size]
        node_repr = node_repr.unsqueeze(1).expand(
            -1,
            self.num_centroid,
            -1).contiguous().view(-1, embed_size)

        # broadcast and reshape centroid embeddings to [node_num * num_centroid, embed_size]

        centroid_repr = self.centroid_embedding.weight.clone()
        centroid_repr = self.proj_tan0(centroid_repr)
        centroid_repr = self.manifold.expmap0(centroid_repr)
        centroid_repr = self.linear(centroid_repr)
        centroid_repr = centroid_repr.unsqueeze(0).expand(
            node_num,
            -1,
            -1).contiguous().view(-1, embed_size)

        # get distance
        node_centroid_dist = self.manifold.dist(node_repr, centroid_repr)
        node_centroid_dist = node_centroid_dist.view(node_num, self.num_centroid) * mask  # (node_num,num_centroid) * (node_num, 1)
        # average pooling over nodes
        # graph_centroid_dist = torch.sum(node_centroid_dist, dim=0) / torch.sum(mask)
        return node_centroid_dist
    def proj_tan0(self,u):
        if self.manifold.name == 'Lorentz':
            narrowed = u.narrow(-1, 0, 1)
            vals = torch.zeros_like(u)
            vals[:, 0:1] = narrowed
            return u - vals
        else:
            return u