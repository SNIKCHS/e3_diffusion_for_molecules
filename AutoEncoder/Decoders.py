import manifolds
import torch.nn as nn
import layers.hyp_layers as hyp_layers
import torch.nn.functional as F
import torch
from layers.layers import GraphConvolution, Linear, get_dim_act


class Decoder(nn.Module):
    """
    Decoder abstract class
    """

    def __init__(self, c, args):
        super(Decoder, self).__init__()
        self.c = c
        self.out = nn.Sequential(
            nn.Linear(args.dim,args.max_z),
            # nn.Sigmoid()
        )

    def decode(self, h, distances, edges, node_mask, edge_mask):

        if self.decode_adj:
            input = (h, distances, edges, node_mask, edge_mask)
            output, distances, edges, node_mask, edge_mask = self.decoder.forward(input)
        else:
            output = self.decoder.forward(h)

        # if self.c is not None:
        #     output = self.manifold.logmap0(output, self.curvatures[-1])
        #     output = self.manifold.proj_tan0(output,  self.curvatures[-1])

        output = self.out(output)

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

        self.decode_adj = True


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
        self.decode_adj = False

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
            self.input_dim, self.output_dim, self.bias, self.c
        )


class HGCNDecoder(Decoder):
    """
    Decoder for HGCAE
    """

    def __init__(self, c, args):
        super(HGCNDecoder, self).__init__(c, args)
        self.manifold = getattr(manifolds, args.manifold)()

        assert args.num_layers > 0

        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        # dims = dims[::-1] # 倒序
        # acts = acts[::-1][:-1] + [lambda x: x]  # Last layer without act
        self.curvatures[0] = c[-1]
        if args.encdec_share_curvature:
            self.curvatures = c[::-1]

        hgc_layers = []
        for i in range(args.num_layers):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.HyperbolicGraphConvolution(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.local_agg,
                )
            )

        self.decoder = nn.Sequential(*hgc_layers)
        self.decode_adj = True

    def decode(self, h, distances, edges, node_mask, edge_mask):
        h_hyp = self.manifold.proj(
            self.manifold.expmap0(
                self.manifold.proj_tan0(h, self.curvatures[0]), c=self.curvatures[0]
            ),
            c=self.curvatures[0]
        )

        output = super(HGCNDecoder, self).decode(h_hyp, distances, edges, node_mask, edge_mask)

        return output


class HNNDecoder(Decoder):
    """
    Decoder for HNN
    """

    def __init__(self, c, args):
        super(HNNDecoder, self).__init__(c, args)
        self.manifold = getattr(manifolds, args.manifold)()

        assert args.num_layers > 0

        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        # dims = dims[::-1]
        # acts = acts[::-1]
        self.curvatures[0] = c[-1]  # encoder的最后一个curvature是decoder的第一个curvature
        if args.encdec_share_curvature:
            self.curvatures = c[::-1]

        hnn_layers = []

        for i in range(args.num_layers):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]

            hnn_layers.append(
                hyp_layers.HNNLayer(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias
                )
            )

        self.decoder = nn.Sequential(*hnn_layers)
        self.decode_adj = False

    def decode(self, h, distances, edges, node_mask, edge_mask):
        h_hyp = self.manifold.proj(
            self.manifold.expmap0(
                self.manifold.proj_tan0(h, self.curvatures[0]), c=self.curvatures[0]
            ),
            c=self.curvatures[0]
        )
        output = super(HNNDecoder, self).decode(h_hyp, distances, edges, node_mask, edge_mask)

        return output


model2decoder = {
    'GCN': GCNDecoder,
    'HNN': HNNDecoder,
    'HGCN': HGCNDecoder,
    'MLP': LinearDecoder,
}
