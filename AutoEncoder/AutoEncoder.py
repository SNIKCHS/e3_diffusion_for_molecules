import torch
import AutoEncoder.Decoders as Decoders
import AutoEncoder.Encoders as Encoders
from torch import nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight,gain=0.5)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
class HyperbolicAE(nn.Module):

    def __init__(self, args):
        super(HyperbolicAE, self).__init__()
        self.device = args.device
        self.encoder = getattr(Encoders, args.model)(args)
        manifolds = self.encoder.manifolds
        self.decoder = Decoders.model2decoder[args.model](manifolds, args)
        self.args = args
        self._edges_dict = {}
        self.pred_edge = args.pred_edge
        self.apply(weight_init)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*args.dim,args.dim),
            nn.Dropout(args.dropout),
            nn.SiLU(),
            nn.Linear(args.dim, args.dim),
            nn.Dropout(args.dropout),
            nn.SiLU(),
            nn.Linear(args.dim, 1),
            nn.Softplus()
        )


    def forward(self, x, h, node_mask, edge_mask):

        categories, charges = h  # (b,n_atom)
        batch_size, n_nodes = categories.shape
        edges = self.get_adj_matrix(n_nodes,batch_size)  # [rows, cols] rows=cols=(batch_size*n_nodes*n_nodes) value in [0,batch_size*n_nodes)
        posterior, distances, edges, node_mask, edge_mask = self.encoder(x, categories, edges, node_mask, edge_mask)
        h = posterior.sample()
        edge_loss = self.edge_pred(edges,edge_mask,h,distances)
        # if torch.any(torch.isnan(h)):
        #     print('posterior nan')
        output = self.decoder.decode(h, distances, edges, node_mask, edge_mask)
        # if torch.any(torch.isnan(output)):
        #     print('output nan')
        rec_loss = self.compute_loss(categories, output,node_mask)
        kl_loss = posterior.kl().mean()
        return rec_loss, kl_loss,edge_loss

    def compute_loss(self, x, x_hat,node_mask):
        """
        auto-encoder的损失
        :param x: encoder的输入 原子类别（0~5）
        :param x_hat: decoder的输出 (b*n_nodes,6)
        :return: loss
        """
        b,n_atom = x.size()

        n_type = x_hat.size(-1)
        atom_loss_f = nn.CrossEntropyLoss(reduction='none')
        rec_loss = atom_loss_f(x_hat.view(-1, n_type), x.view(-1)) * node_mask.squeeze()
        rec_loss = rec_loss.sum()/b
        return rec_loss

    def get_adj_matrix(self, n_nodes, batch_size):
        # 对每个n_nodes，batch_size只要算一次
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(self.device),
                         torch.LongTensor(cols).to(self.device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size)

    def edge_pred(self,edges,edge_mask,h,target_edge):
        row, col = edges
        h_row = h[row]
        h_col = h[col]
        pred_edge = self.edge_mlp(torch.concat([h_row,h_col],dim=-1)) * edge_mask
        target_edge = target_edge * edge_mask
        edge_loss_f = nn.MSELoss(reduction='mean')
        # zeros = torch.zeros_like(pred_edge,device=pred_edge.device)
        # ones = torch.ones_like(pred_edge, device=pred_edge.device)
        # edge_cutoff = torch.where(edge>5,zeros,ones)

        loss1 = torch.sqrt_(edge_loss_f(pred_edge,target_edge))
        return loss1
    def show_curvatures(self):
        if self.args.manifold is not 'Euclidean':
            c = [m.k for m in self.encoder.manifolds]
            c.append([m.k for m in self.decoder.manifolds])
            print(c)

