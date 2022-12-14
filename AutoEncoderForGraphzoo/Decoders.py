import manifolds
import torch.nn as nn
import layers.hyp_layers as hyp_layers
import torch.nn.functional as F
import torch

from layers.att_layers import DenseAtt
from layers.layers import GraphConvolution, Linear, get_dim_act


class Decoder(nn.Module):
    """
    Decoder abstract class
    """

    def __init__(self, manifold, args):
        super(Decoder, self).__init__()
        self.manifold = manifold
        # self.out = nn.Sequential(
        #     nn.Linear(args.hidden_dim, args.max_z),
        #     # nn.Sigmoid()
        # )
        self.pred_edge = args.pred_edge
        # self.link_net = DenseAtt(args.dim,args.dropout)
        # self.link_out = nn.Linear(args.dim,1)

    def decode(self, h, distances, edges, node_mask, edge_mask):
        if self.message_passing:
            input = (h, distances, edges, node_mask, edge_mask)
            output, distances, edges, node_mask, edge_mask = self.decoder.forward(input)
        else:
            output = self.decoder.forward(h)

        # if self.c is not None:
        #     output = self.manifold.logmap0(output, self.curvatures[-1])
        #     output = self.manifold.proj_tan0(output,  self.curvatures[-1])
        # node_pred = self.out(output)


        # if self.pred_edge:
        #     row, col = edges
        #     output = self.manifold.logmap0(output, self.curvatures[-1])
        #     output = self.manifold.proj_tan0(output,  self.curvatures[-1])
        #     edge_pred = self.link_net(output[row], output[col], distances, edge_mask)  # (b*atom*atom,1)
        #     edge_pred = self.link_out(edge_pred)
        # else:
        #     edge_pred = None

        return output


class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c, args)

        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        # dims = dims[::-1]
        # acts = acts[::-1]
        # acts = acts[::-1][:-1] + [lambda x: x]  # Last layer without act
        gc_layers = []
        for i in range(args.num_layers):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.decoder = nn.Sequential(*gc_layers)

        self.message_passing = True


class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c, args)
        self.manifold = getattr(manifolds, args.manifold)()
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(args.num_layers):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.decoder = nn.Sequential(*layers)
        self.message_passing = False

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
            self.input_dim, self.output_dim, self.bias, self.c
        )

class HNNDecoder(Decoder):
    """
    Decoder for HNN
    """

    def __init__(self, manifold, args):
        super(HNNDecoder, self).__init__(manifold, args)

        dims, acts, self.manifolds = hyp_layers.get_dim_act_curv(args)
        self.manifolds.insert(0, self.manifold)

        hnn_layers = []

        for i in range(args.num_layers):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            manifold_in, manifold_out = self.manifolds[i], self.manifolds[i + 1]

            hnn_layers.append(
                hyp_layers.HNNLayer(
                    self.manifold, in_dim, out_dim, manifold_in, manifold_out, args.dropout, act, args.bias
                )
            )

        self.decoder = nn.Sequential(*hnn_layers)
        self.message_passing = False

    def decode(self, h, distances, edges, node_mask, edge_mask):
        h_hyp = self.manifold.expmap0(
                self.manifold.proj_tan0(h, self.curvatures[0]), c=self.curvatures[0]
            )
        # h_hyp = self.manifold.proj(h_hyp,c=self.curvatures[0])
        h = self.manifold.proj(h_hyp, c=self.curvatures[0])
        output = super(HNNDecoder, self).decode(h, distances, edges, node_mask, edge_mask)

        return output

class HGCNDecoder(Decoder):
    """
    Decoder for HGCAE
    """

    def __init__(self, manifold, args):
        super(HGCNDecoder, self).__init__(manifold, args)
        dims, acts, self.manifolds = hyp_layers.get_dim_act_curv(args)
        self.manifolds.insert(0, self.manifold)

        hgc_layers = []
        for i in range(args.num_layers):
            manifold_in, manifold_out = self.manifolds[i], self.manifolds[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.HyperbolicGraphConvolution(
                    in_dim, out_dim, manifold_in, manifold_out, args.dropout, act, args.bias, args.local_agg,
                )
            )

        self.decoder = nn.Sequential(*hgc_layers)
        self.message_passing = True

    def decode(self, h, distances, edges, node_mask, edge_mask):
        # h = self.manifold.expmap0(
        #         self.manifold.proj_tan0(h, self.curvatures[0]), c=self.curvatures[0]
        #     )
        output = super(HGCNDecoder, self).decode(h, distances, edges, node_mask, edge_mask)

        return output





model2decoder = {
    'GCN': GCNDecoder,
    'HNN': HNNDecoder,
    'HGCN': HGCNDecoder,
    'MLP': LinearDecoder,
}
