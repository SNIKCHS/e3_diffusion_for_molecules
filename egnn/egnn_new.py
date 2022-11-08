import numpy as np
from torch import nn
import torch
import math

import manifolds
from layers.hyp_layers import HyperbolicGraphConvolution, HypLinear, HypNorm, HypAgg, HypAct


class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:  # edge_attr.shape=(b*n_nodes*n_nodes,2) concat2个distances
            out = torch.cat([source, target, edge_attr], dim=1)  # (b*n_nodes*n_nodes,2*hidden_nf(default:128)+2)
        mij = self.edge_mlp(out)  # (b*n_nodes*n_nodes,hidden_nf(default:128))

        if self.attention:
            att_val = self.att_mlp(mij)  # (b*n_nodes*n_nodes,1)
            out = mij * att_val  # (b*n_nodes*n_nodes,hidden_nf(default:128))
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask

        return out, mij  # (b*n_nodes*n_nodes,hidden_nf(default:128)) (b*n_nodes*n_nodes,hidden_nf(default:128))

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),  # num_segments=b*n_nodes
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)  # sum掉第二个n_nodes (b*n_nodes*n_nodes,hidden_nf)->(b*n_nodes,hidden_nf)

        if node_attr is not None:  # None
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)  # (b*n_nodes,2*hidden_nf)
        out = x + self.node_mlp(agg)  # residual connect
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)  # pairwise的信息 (b*n_nodes*n_nodes,hidden_nf(default:128)) shape都一样，mij不使用
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)  # h.shape=(b*n_nodes,hidden_nf)
        if node_mask is not None:
            h = h * node_mask
        return h, mij

class HGCL(nn.Module):
    def __init__(self, input_nf, output_nf, c_in, c_out, act_fn=nn.SiLU(),manifold='Hyperboloid',edges_in_d=2):
        super(HGCL, self).__init__()
        self.manifold = getattr(manifolds, manifold)()
        # self.hgcl = HyperbolicGraphConvolution(self.manifold, input_nf, output_nf, c_in, c_out, dropout=0, act=act_fn, use_bias=1, local_agg=1,edge_dim=edges_in_d)
        self.norm = HypNorm(self.manifold, input_nf, c_in)
        self.linear = HypLinear(self.manifold, input_nf, output_nf, c_in, dropout=0, use_bias=1)
        self.agg = HypAgg(self.manifold, c_in, output_nf, dropout=0, local_agg=True, edge_dim=edges_in_d)
        self.hyp_act = HypAct(self.manifold, c_in, c_out, act_fn)
        self.norm1 = HypNorm(self.manifold, output_nf, c_in)

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        # if torch.any(torch.isnan(h)):
        #     print('input nan')
        h = self.norm(h)
        # if torch.any(torch.isnan(h)):
        #     print('norm0 nan')
        h = self.linear.forward(h)
        # if torch.any(torch.isnan(h)):
        #     print('linear nan')
        h = self.agg.forward(h, edge_attr, edge_index, node_mask, edge_mask)
        # if torch.any(torch.isnan(h)):
        #     print('agg nan')
        h = self.hyp_act.forward(h)
        # if torch.any(torch.isnan(h)):
        #     print('act nan')

        return h, None

class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        # if torch.any(torch.isnan(input_tensor)):
        #     print('input_tensor nan')
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)

        # if torch.any(torch.isnan(trans)):
        #     print('trans nan')

        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        # if torch.any(torch.isnan(agg)):
        #     print('agg nan')
        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum',hyp=False,c=None,manifold='Hyperboloid'):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.hyp = hyp

        if hyp:
            self.manifold = getattr(manifolds, manifold)()
            self.c = c
            acts = [act_fn] * n_layers  # len=args.num_layers
            dims = [self.hidden_nf] * (n_layers + 1)  # len=args.num_layers+1

            for i in range(0, n_layers):
                self.add_module("gcl_%d" % i,
                                HGCL(
                                    dims[i], dims[i+1], c, c,acts[i],manifold=manifold,edges_in_d=edge_feat_nf
                                ))
        else:
            for i in range(0, n_layers):
                self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                                  act_fn=act_fn, attention=attention,
                                                  normalization_factor=self.normalization_factor,
                                                  aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # edge_index:list[rol,col] (b*n_nodes*n_nodes,)
        # Edit Emiel: Remove velocity as input
        # if torch.any(torch.isnan(x)):
        #     print('place 1 x nan')
        # if torch.any(torch.isnan(h)):
        #     print('place 1 h nan')
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        # if torch.any(torch.isnan(coord_diff)):
        #     print('place 1 coord_diff nan')
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)

        # if torch.any(torch.isnan(distances)):
        #     print('distances nan')
        # if torch.any(torch.isnan(edge_attr)):
        #     print('edge_attr nan')
        edge_attr = torch.cat([distances, edge_attr], dim=1)  # (b*n_nodes*n_nodes,2) edge_attr是最初的distance
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)

        # if torch.any(torch.isnan(h))
        #     print('place 2 h nan')
        # if self.hyp:
        #     h_tan = self.manifold.logmap0(h,self.c)
        # else:
        #     h_tan = h
        # if torch.any(torch.isnan(h_tan)):
        #     print('place 2 h_hyp nan')
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)
        # if torch.any(torch.isnan(x)):
        #     print('place 2 x nan')
        return h, x

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=0.25)
class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum',hyp=False,c=None,manifold='Hyperboloid'):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.hyp = hyp

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        if hyp:
            self.manifold = getattr(manifolds, manifold)()
            self.c = c
            self.embedding = HypLinear(getattr(manifolds, manifold)(),in_node_nf, self.hidden_nf,c,dropout=0,use_bias=1)
            # self.embedding_out = HypLinear(getattr(manifolds, manifold)(),hidden_nf, out_node_nf,c,dropout=0,use_bias=1)
        else:
            self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Sequential(
            nn.Linear(self.hidden_nf, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, out_node_nf)
        )


        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method,hyp=hyp,c=c,manifold=manifold))
        self.to(self.device)
        self.apply(weight_init)


    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)  # (b*n_node*n_node,1)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)


        if self.hyp:
            h0 = h[..., 0].clone()
            h = self.manifold.expmap0(
                self.manifold.proj_tan0(h, self.c), c=self.c
            )

        h = self.embedding(h)  # default: (b*n_nodes,hidden_nf=128)

        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances)
            # if torch.any(torch.isnan(h)):
            #     print('loop'+str(i)+' nan')

        # Important, the bias of the last linear might be non-zero

        if self.hyp:
            h = self.manifold.proj_tan0(
                self.manifold.logmap0(h, self.c),
                c=self.c
            )
            h[...,0] = h0

        h = self.embedding_out(h)

        if node_mask is not None:
            h = h * node_mask
        return h, x


class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 act_fn=nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=1, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                attention=attention))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


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
