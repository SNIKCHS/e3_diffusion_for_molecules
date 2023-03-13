from geoopt.tensor import ManifoldParameter
import torch
import torch.nn as nn



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
        self.centroid_embedding.weight = ManifoldParameter(manifold.random_normal((num_centroid, dim)), manifold)

    def forward(self, node_repr, mask):

        node_num,embed_size = node_repr.size()
        # broadcast and reshape node_repr to [node_num * num_centroid, embed_size]
        node_repr = node_repr.unsqueeze(1).expand(
            -1,
            self.num_centroid,
            -1).contiguous().view(-1, embed_size)

        # broadcast and reshape centroid embeddings to [node_num * num_centroid, embed_size]

        centroid_repr = self.centroid_embedding(torch.arange(self.num_centroid).to(node_repr.device))

        centroid_repr = centroid_repr.unsqueeze(0).expand(
            node_num,
            -1,
            -1).contiguous().view(-1, embed_size)

        # get distance
        node_centroid_dist = self.manifold.dist(node_repr, centroid_repr)
        node_centroid_dist = node_centroid_dist.view(node_num, self.num_centroid) * mask  # (node_num,num_centroid) * (node_num, 1)

        return node_centroid_dist