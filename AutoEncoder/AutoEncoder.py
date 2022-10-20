import torch
import AutoEncoder.Decoders as Decoders
import AutoEncoder.Encoders as Encoders
from torch import nn
import manifolds


class HyperbolicAE(nn.Module):

    def __init__(self, args):
        super(HyperbolicAE, self).__init__()
        self.device = args.device
        self.manifold = getattr(manifolds, args.manifold)()  # 选择相应的流形
        self.encoder = getattr(Encoders, args.model)(args)
        c = self.encoder.curvatures if hasattr(self.encoder, 'curvatures') else args.c
        self.decoder = Decoders.model2decoder[args.model](c, args)
        self.args = args
        self._edges_dict = {}
        self.pred_edge = args.pred_edge

    def forward(self, x, h, node_mask, edge_mask):
        # note：将来使用EnVariationalDiffusion的normalize
        categories, charges = h  # (b,n_atom)
        batch_size, n_nodes = categories.shape
        edges = self.get_adj_matrix(n_nodes,batch_size)  # [rows, cols] rows=cols=(batch_size*n_nodes*n_nodes) value in [0,batch_size*n_nodes)
        h, distances, edges, node_mask, edge_mask = self.encoder(x, categories, charges, edges, node_mask, edge_mask)
        # h += 0.5*torch.randn(h.size(),device=h.device)
        output,_ = self.decoder.decode(h, distances, edges, node_mask, edge_mask)

        return self.compute_loss(categories, output)

    def compute_loss(self, x, x_hat):
        """
        auto-encoder的损失
        :param x: encoder的输入 原子类别（0~5）
        :param x_hat: decoder的输出 (b*n_nodes,6)
        :return: loss
        """
        b,n_atom = x.size()

        n_type = x_hat.size(-1)
        atom_loss_f = nn.CrossEntropyLoss(reduction='sum')
        rec_loss = atom_loss_f(x_hat.view(-1, n_type), x.view(-1))/b
        # print(x[0])
        # print(torch.argmax(x_hat.view(b,n_atom, n_type)[0],dim=1))

        # if self.pred_edge:
        #     edge_loss_f = nn.MSELoss(reduction='mean')
        #     zeros = torch.zeros_like(edge,device=edge.device)
        #     ones = torch.ones_like(edge, device=edge.device)
        #     edge_cutoff = torch.where(edge>5,zeros,ones)
        #     edge_hat = edge_hat * edge_cutoff
        #     edge = edge*edge_cutoff
        #     loss1 = torch.sqrt(edge_loss_f(edge_hat,edge))
        # else:
        #     loss1=torch.tensor(0.0,device=loss0.device)

        return rec_loss,torch.tensor(0.0,device=rec_loss.device)

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
