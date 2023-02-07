import torch
from torch import nn

from AutoEncoder.distributions import DiagonalGaussianDistribution
from layers.hyp_layers import get_dim_act_curv, HNNLayer, HGCLayer
from layers.layers import get_dim_act, GraphConvolution, Linear


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, args):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(args.max_z, args.hidden_dim, padding_idx=0)  # qm9 max_z=6
        self.mean_logvar = nn.Linear(args.dim,2*args.dim)


    def forward(self, x, categories, edges, node_mask, edge_mask):
        h = self.embedding(categories)  # (b,n_atom,n_atom_embed)
        b, n_nodes, _ = h.shape

        x = x.view(b * n_nodes, 3)
        node_mask = node_mask.view(b * n_nodes, 1)
        edge_mask = edge_mask.view(b * n_nodes * n_nodes, 1)
        h = h.view(b * n_nodes, -1) * node_mask  # (b*n_atom,n_atom_embed+1)

        distances, _ = coord2diff(x, edges)  # (b*n_node*n_node,1)

        output, distances, edges, node_mask, edge_mask = self.encode(h, distances, edges, node_mask, edge_mask)
        # if torch.any(torch.isnan(output)):
        #     print('ENCoutput nan')
        parameters = self.mean_logvar(output)
        posterior = DiagonalGaussianDistribution(parameters,self.manifold,node_mask)

        return posterior, distances, edges, node_mask, edge_mask

    def encode(self,h, distances, edges, node_mask, edge_mask):
        if self.message_passing:
            input = (h, distances, edges, node_mask, edge_mask)
            output, distances, edges, node_mask, edge_mask = self.layers(input)
        else:
            output = self.layers(h)
        return output, distances, edges, node_mask, edge_mask


# class MLP(Encoder):
#     """
#     Multi-layer perceptron.
#     """
#
#     def __init__(self, args):
#         super(MLP, self).__init__(args)
#         assert args.num_layers > 0
#         self.manifold = getattr(manifolds, args.manifold)()
#         dims, acts = get_dim_act(args)
#         layers = []
#         for i in range(args.num_layers):
#             in_dim, out_dim = dims[i], dims[i + 1]
#             act = acts[i]
#             layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
#         self.layers = nn.Sequential(*layers)
#         self.message_passing = False
#         self.norm = nn.LayerNorm(args.dim)
#
#
# class HNN(Encoder):
#     """
#     Hyperbolic Neural Networks.
#     """
#
#     def __init__(self, args):
#         super(HNN, self).__init__(args)
#
#         self.manifold = getattr(manifolds, args.manifold)()
#         assert args.num_layers > 1
#         dims, acts, self.curvatures = get_dim_act_curv(args)
#         hnn_layers = []
#         for i in range(args.num_layers):
#             c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
#             in_dim, out_dim = dims[i], dims[i + 1]
#             act = acts[i]
#             hnn_layers.append(
#                 HNNLayer(self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias)
#             )
#         self.layers = nn.Sequential(*hnn_layers)
#         self.message_passing = False
#         if self.manifold.name == 'Hyperboloid':
#             self.norm = nn.LayerNorm(args.dim - 1)
#         else:
#             self.norm = nn.LayerNorm(args.dim)
#
#     def encode(self, h, distances, edges, node_mask, edge_mask):
#         h_hyp = self.manifold.proj(
#             self.manifold.expmap0(
#                 self.manifold.proj_tan0(h, self.curvatures[0]), c=self.curvatures[0]
#             ),
#             c=self.curvatures[0]
#         )
#
#         output, distances, edges, node_mask, edge_mask = super(HNN, self).encode( h_hyp, distances, edges, node_mask, edge_mask)
#
#         output = self.manifold.proj_tan0(
#             self.manifold.logmap0(output, self.curvatures[-1]),
#             c=self.curvatures[-1]
#         )
#         return output, distances, edges, node_mask, edge_mask
#
#
class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, args):
        super(GCN, self).__init__(args)
        self.manifold = None
        self.manifolds = None
        dims, acts = get_dim_act(args,args.enc_layers)
        gc_layers = []
        for i in range(args.enc_layers):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.message_passing = True
        self.norm = nn.LayerNorm(args.dim)


class HGCN(Encoder):
    """
    Hyperbolic Graph Convolutional Auto-Encoders.
    """

    def __init__(self, args):  # , use_cnn
        super(HGCN, self).__init__(args)

        dims, acts, self.manifolds = get_dim_act_curv(args,args.enc_layers)
        self.manifold = self.manifolds[-1]
        hgc_layers = []
        for i in range(args.enc_layers):
            m_in, m_out = self.manifolds[i], self.manifolds[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                HGCLayer(
                    in_dim, out_dim, m_in, m_out, args.dropout, act
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.message_passing = True
        # if self.manifold.name == 'Hyperboloid':
        #     self.norm = nn.LayerNorm(args.dim - 1)
        # else:
        #     self.norm = nn.LayerNorm(args.dim)

    def encode(self, h, distances, edges, node_mask, edge_mask):
        h = self.proj_tan0(h)
        h = self.manifolds[0].expmap0(h)
        output, distances, edges, node_mask, edge_mask = super(HGCN, self).encode(h, distances, edges, node_mask, edge_mask)

        output = self.proj_tan0(self.manifolds[-1].logmap0(output))
        return output, distances, edges, node_mask, edge_mask
    def proj_tan0(self,u):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals
