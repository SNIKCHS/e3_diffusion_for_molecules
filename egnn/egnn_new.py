import geoopt
import numpy as np

from layers.att_layers import DenseAtt
from manifold.Lorentz import Lorentz
from torch import nn
import torch
import math

from torch.nn import init

from layers.hyp_layers import HGCLayer

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    input in manifold
    output in manifold
    """

    def __init__(self, in_features, out_features, manifold, dropout=0):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self.bias = nn.Parameter(torch.Tensor(1, out_features))
        self.linear = nn.Linear(in_features, out_features, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.linear.weight, gain=0.1)
        init.constant_(self.bias, 0)
    def proj_tan0(self,u):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals
    def forward(self, x):
        x = self.manifold.logmap0(x)
        x = self.linear(x)
        x = self.proj_tan0(x)
        x = self.manifold.expmap0(x)
        bias = self.bias.repeat(x.size(0),1)
        bias = self.proj_tan0(bias)
        bias = self.manifold.transp0(x, bias)
        res = self.manifold.expmap(x, bias)

        return res
class HGCL(nn.Module):
    def __init__(self, in_features, out_features, manifold_in, manifold_out, dropout, act,edge_dim=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.bias = nn.Parameter(torch.Tensor(1, out_features))
        self.linear = nn.Linear(in_features, out_features, bias=False)

        self.normalization_factor = 1
        self.aggregation_method = 'sum'
        self.att = DenseAtt(out_features,dropout=dropout, edge_dim=edge_dim)
        # self.node_mlp = nn.Sequential(
        #     nn.Linear(out_features, out_features),
        #     nn.LayerNorm(out_features),
        #     nn.SiLU(),
        #     nn.Linear(out_features, out_features))

        self.act = act
        if self.manifold_in.name == 'Lorentz':
            self.ln = nn.LayerNorm(out_features - 1)
        else:
            self.ln = nn.LayerNorm(out_features)
        self.reset_parameters()

    def proj_tan0(self,u):
        if self.manifold_in.name == 'Lorentz':
            narrowed = u.narrow(-1, 0, 1)
            vals = torch.zeros_like(u)
            vals[:, 0:1] = narrowed
            return u - vals
        else:
            return u
    def reset_parameters(self):
        # init.xavier_uniform_(self.linear.weight, gain=0.01)
        init.constant_(self.bias, 0)

    def forward(self, input):
        h, edge_attr, edges, node_mask, edge_mask = input

        h = self.HypLinear(h)
        # print('HypLinear',h)
        # if torch.any(torch.isnan(h)):
        #     print('HypLinear nan')
        h,edge_attr = self.HypAgg(h, edge_attr, edges, node_mask, edge_mask)
        # print('HypAgg', h)
        # if torch.any(torch.isnan(h)):
        #     print('HypAgg nan')
        h = self.HNorm(h)
        # print('HNorm', h)
        # if torch.any(torch.isnan(h)):
        #     print('HNorm nan')
        h = self.HypAct(h)
        # print('HypAct', h)
        # if torch.any(torch.isnan(h)):
        #     print('HypAct nan')
        output = (h, edge_attr, edges, node_mask, edge_mask)
        return output

    def HypLinear(self, x):
        x = self.manifold_in.logmap0(x)
        x = self.linear(x)
        x = self.proj_tan0(x)
        x = self.manifold_in.expmap0(x)
        bias = self.proj_tan0(self.bias.view(1, -1))
        bias = self.manifold_in.transp0(x, bias)
        res = self.manifold_in.expmap(x, bias)
        return res

    def HypAgg(self, x, edge_attr, edges, node_mask, edge_mask):
        x_tangent = self.manifold_in.logmap0(x)  # (b*n_node,dim)

        row, col = edges  # 0,0,0...0,1 0,1,2..,0
        x_tangent_row = x_tangent[row]
        x_tangent_col = x_tangent[col]

        # geodesic = self.manifold_in.dist(x[row], x[col],keepdim=True)  # (b*n_node*n_node,dim)
        # edge_attr = torch.cat([edge_attr,geodesic],dim=-1)
        att = self.att(x_tangent_row, x_tangent_col, edge_attr, edge_mask)  # (b*n_node*n_node,dim)
        x_local_tangent = self.manifold_in.logmap(x[row], x[col])  # (b*n_node*n_node,dim)  x_col落在x_row的切空间
        agg = x_local_tangent * att
        out = unsorted_segment_sum(agg, row, num_segments=x_tangent.size(0),  # num_segments=b*n_nodes
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)  # sum掉第二个n_nodes (b*n_nodes*n_nodes,dim)->(b*n_nodes,dim)

        # out = self.node_mlp(out)
        support_t = self.manifold_in.proju(x, out)
        output = self.manifold_in.expmap(x, support_t)
        # print('output:', torch.max(output.view(-1)), torch.min(output.view(-1)))
        return output,edge_attr

    def HypAct(self, x):
        xt = self.act(self.manifold_in.logmap0(x))
        xt = self.proj_tan0(xt)
        out = self.manifold_out.expmap0(xt)
        return out

    def HNorm(self, x):
        h = self.manifold_in.logmap0(x)
        if self.manifold_in.name == 'Lorentz':
            h[..., 1:] = self.ln(h[..., 1:].clone())
        else:
            h = self.ln(h)
        h = self.manifold_in.expmap0(h)
        return h
# class HGCL(nn.Module):
#     def __init__(self, input_nf, output_nf, hidden_nf,manifold_in, manifold_out, normalization_factor, aggregation_method,
#                  edges_in_d=0, act_fn=nn.SiLU()):
#         super(HGCL, self).__init__()
#         input_edge = input_nf * 2
#         self.manifold_in = manifold_in
#         self.manifold_out = manifold_out
#         self.normalization_factor = normalization_factor
#         self.aggregation_method = aggregation_method
#
#         self.edge_mlp = nn.Sequential(
#             nn.Linear(input_nf, hidden_nf),
#             act_fn,
#             nn.Linear(hidden_nf, hidden_nf),
#             )  # tanh
#
#         self.node_mlp = nn.Sequential(
#             # nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
#             nn.Linear(hidden_nf, hidden_nf),
#             act_fn,
#             nn.Linear(hidden_nf, output_nf),
#         )
#
#         self.att_mlp = nn.Sequential(
#             nn.Linear(input_edge + edges_in_d, 1),
#             nn.Sigmoid()
#         )
#         self.ln = nn.LayerNorm(hidden_nf-1)
#         self.apply(weight_init)
#
#     def edge_model(self, source, target, edge_attr, edge_mask,size):
#         b,n_nodes = size
#         # c = self.manifold_in.k.float()
#         # c = c.view(b, n_nodes)
#         # c = c.repeat(1, n_nodes).view(-1, 1)
#
#         # s = self.manifold_in.logmap(source,source,expand_k=True)
#         t = self.manifold_in.logmap(source,target)
#         t = self.manifold_in.transp0back(source,t)
#         out = self.edge_mlp(t)
#         # out = torch.cat([s, t, edge_attr], dim=1)  # (b*n_nodes*n_nodes,2*hidden_nf(default:128)+2)
#
#         # mij = self.edge_mlp(out)  # (b*n_nodes*n_nodes,hidden_nf(default:128))
#         # mij = self.proj_tan0(mij)
#         # mij = self.manifold_in.proju(source, mij,expand_k=True)
#
#         source_att = self.manifold_in.logmap0(source)
#         target_att = self.manifold_in.logmap0(target)
#         att_val = self.att_mlp(torch.cat([source_att, target_att, edge_attr], dim=1))  # (b*n_nodes*n_nodes,1)
#         out = out * att_val  # (b*n_nodes*n_nodes,hidden_nf(default:128))
#
#         out = self.proj_tan0(out)
#         # out = self.manifold_in.proju(source, out)
#         if edge_mask is not None:
#             out = out * edge_mask
#
#         return out, None  # (b*n_nodes*n_nodes,hidden_nf(default:128)) (b*n_nodes*n_nodes,hidden_nf(default:128))
#
#     def node_model(self, x, edge_index, edge_attr, node_attr):
#         # print('xin:', torch.max(x.view(-1)), torch.min(x.view(-1)))
#         row, col = edge_index
#         agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),  # num_segments=b*n_nodes
#                                    normalization_factor=1000,
#                                    aggregation_method=self.aggregation_method)  # sum掉第二个n_nodes (b*n_nodes*n_nodes,hidden_nf)->(b*n_nodes,hidden_nf)
#
#         # agg = self.manifold_in.proju(x, agg)
#         # print('agg:', torch.max(agg.view(-1)), torch.min(agg.view(-1)))
#         # x_tangent = self.manifold_in.logmap(x, x)
#         # out = self.manifold_in.expmap0(agg)  # 相当于x+agg
#         # print('out:', torch.max(out.view(-1)), torch.min(out.view(-1)))
#         agg = self.manifold_in.transp0(x,agg)
#         x = self.manifold_in.expmap(x,agg)
#         # print('x:', torch.max(x.view(-1)), torch.min(x.view(-1)))
#         x_tangent = self.manifold_in.logmap0(x)
#         # out_tangent = self.manifold_in.logmap0(out)
#         # if node_attr is not None:  # None
#         #     agg = torch.cat([x_tangent, agg, node_attr], dim=1)
#         # else:
#         #     agg = torch.cat([x_tangent, agg], dim=1)  # (b*n_nodes,2*hidden_nf)
#         out = x_tangent+self.node_mlp(x_tangent)  # residual connect
#
#         out = self.proj_tan0(out)
#         # print('befnorm:', torch.max(out.view(-1)), torch.min(out.view(-1)))
#         if self.manifold_in.name == 'Lorentz':
#             out[..., 1:] = self.ln(out[..., 1:].clone())
#         # print('norm:', torch.max(out.view(-1)), torch.min(out.view(-1)))
#         out = self.manifold_out.expmap0(out)
#
#         # print('xout:', torch.max(out.view(-1)), torch.min(out.view(-1)))
#         return out, agg
#
#     def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
#         row, col = edge_index
#         b_node,b_node_2 = h.size(0),row.size(0)
#         n_nodes = b_node_2 // b_node
#         size = (b_node//n_nodes,n_nodes)
#         geodesic = self.manifold_in.dist(h[row], h[col], keepdim=True)  # (b*n_node*n_node,dim)
#         edge_attr = torch.cat([edge_attr,geodesic],dim=-1)
#
#
#         edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask,size)  # pairwise的信息 (b*n_nodes*n_nodes,hidden_nf(default:128)) shape都一样，mij不使用
#         h, agg = self.node_model(h, edge_index, edge_feat, node_attr)  # h.shape=(b*n_nodes,hidden_nf)
#         if node_mask is not None:
#             h = h * node_mask
#         return h, edge_attr
#
#     def proj_tan0(self,u):
#         narrowed = u.narrow(-1, 0, 1)
#         vals = torch.zeros_like(u)
#         vals[:, 0:1] = narrowed
#         return u - vals

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
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index

        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)

        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)

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
                 normalization_factor=100, aggregation_method='sum',hyp=False,manifold_in=None,manifold_out=None):
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
            self.manifold = manifold_out
            dims = [self.hidden_nf] * (n_layers + 1)  # len=args.num_layers+1

            for i in range(0, n_layers):
                self.add_module("gcl_%d" % i,
                                HGCL(
                                    dims[i], dims[i+1], manifold_in, manifold_out,0,
                                    act=act_fn,edge_dim=edge_feat_nf)
                                )
                # self.add_module("gcl_%d" % i,
                #                 HGCL(
                #                     dims[i], dims[i + 1],dims[i + 1], manifold_in, manifold_out,
                #                     normalization_factor=self.normalization_factor,aggregation_method=self.aggregation_method,
                #                     act_fn=act_fn, edges_in_d=edge_feat_nf)
                #                 )

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
        # self.to(self.device)
        self.apply(weight_init)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # edge_index:list[rol,col] (b*n_nodes*n_nodes,)
        # Edit Emiel: Remove velocity as input


        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)

        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)

        edge_attr = torch.cat([distances, edge_attr], dim=1)  # (b*n_nodes*n_nodes,2) edge_attr是最初的distance
        for i in range(0, self.n_layers):
            input = h, edge_attr, edge_index, node_mask, edge_mask
            if self.hyp:
                h, edge_attr, _, _, _ = self._modules["gcl_%d" % i](input)
                # h, edge_attr = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
            else:
                h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
          # (b*n_node*n_node,1)
        if self.hyp:
            h_t = self.manifold.logmap0(h)
        else:
            h_t = h
        x = self._modules["gcl_equiv"](h_t, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)
        return h, x

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight,gain=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum',hyp=False,):
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
            edge_feat_nf = 2
            self.manifolds = [geoopt.Lorentz(1,learnable=True) for _ in range(n_layers+1)]
            self.embedding = HypLinear(in_node_nf, self.hidden_nf,self.manifolds[0])
            # self.embedding_out = HypLinear(self.hidden_nf, out_node_nf,self.manifolds[-1])
            # self.curvature_net = nn.Sequential(
            #     nn.Linear(1,64),
            #     nn.SiLU(),
            #     nn.Linear(64, 64),
            #     nn.SiLU(),
            #     nn.Linear(64, n_layers+1), # [b,20][20,n_layers+1]
            #     nn.Softplus()
            # )
        else:
            self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)

        if hyp:
            for i in range(0, n_layers):
                self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                                   act_fn=act_fn, n_layers=inv_sublayers,
                                                                   attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                                   coords_range=coords_range, norm_constant=norm_constant,
                                                                   sin_embedding=self.sin_embedding,
                                                                   normalization_factor=self.normalization_factor,
                                                                   aggregation_method=self.aggregation_method,hyp=hyp,
                                                                   manifold_in=self.manifolds[i],manifold_out=self.manifolds[i+1],
                                                                   ))
        else:
            for i in range(0, n_layers):
                self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                                   act_fn=act_fn, n_layers=inv_sublayers,
                                                                   attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                                   coords_range=coords_range, norm_constant=norm_constant,
                                                                   sin_embedding=self.sin_embedding,
                                                                   normalization_factor=self.normalization_factor,
                                                                   aggregation_method=self.aggregation_method,hyp=hyp))



    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None,t=None):
        # print(t.shape) (b,1)
        # print(h.shape) (b*n_nodes,dim+1)
        # if self.hyp:
        #     k = self.curvature_net(t)
        #     for i in range(self.n_layers+1):
        #         self.manifolds[i].set_k(k[:,i].unsqueeze(1))
            # print('t:%.3f' % (t[0]))
            # print(k[0])
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)  # (b*n_node*n_node,1)

        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)

        if self.hyp:
            h = self.manifolds[0].expmap0(h)
        h = self.embedding(h)  # default: (b*n_nodes,hidden_nf=128)


        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances)
            # if torch.any(torch.isnan(h)):
            #     print('loop'+str(i)+' nan')

        # Important, the bias of the last linear might be non-zero
        if self.hyp:
            h = self.manifolds[-1].logmap0(h)
            h = self.proj_tan0(h)
        h = self.embedding_out(h)

        # print('hout:', h, h[:3])
        if node_mask is not None:
            h = h * node_mask
        return h, x

    def proj_tan0(self, u):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals


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
    return norm, coord_diff


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
