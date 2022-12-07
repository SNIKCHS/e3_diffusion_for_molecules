import torch
from torch import nn
import AutoEncoderForGeoopt.Decoders as Decoders
import AutoEncoderForGeoopt.Encoders as Encoders
from AutoEncoderForGeoopt.CentroidDistance import CentroidDistance
from geoopt.manifolds.lorentz import Lorentz
import graphzoo


class HyperbolicAE(nn.Module):

    def __init__(self, args):
        super(HyperbolicAE, self).__init__()
        self.device = args.device
        self.encoder = getattr(Encoders, args.model)(args)
        manifold = self.encoder.manifolds[-1] if hasattr(self.encoder, 'manifolds') else Lorentz()
        self.decoder = Decoders.model2decoder[args.model](manifold, args)
        self.distance = CentroidDistance(args.num_centroid,args.dim, self.decoder.manifolds[-1])
        self.output_linear = nn.Sequential(
            nn.Linear(args.num_centroid, args.num_centroid),
            nn.ReLU(),
            nn.Linear(args.num_centroid, args.num_centroid),
            nn.ReLU(),
            nn.Linear(args.num_centroid, args.max_z),
        )
            # nn.Linear(args.num_centroid,args.max_z)
        self.args = args
        self._edges_dict = {}
        self.pred_edge = args.pred_edge


    def forward(self, x, h, node_mask, edge_mask):

        categories, charges = h  # (b,n_atom)
        batch_size, n_nodes = categories.shape
        edges = self.get_adj_matrix(n_nodes,batch_size)  # [rows, cols] rows=cols=(batch_size*n_nodes*n_nodes) value in [0,batch_size*n_nodes)
        posterior, distances, edges, node_mask, edge_mask = self.encoder(x, categories, edges, node_mask, edge_mask)
        h = posterior.sample()
        print(h[0])
        output = self.decoder.decode(h, distances, edges, node_mask, edge_mask)

        _, node_centroid_sim = self.distance(output, node_mask)
        scores = self.output_linear(node_centroid_sim.squeeze())
        # return self.compute_loss(categories, output,node_mask,posterior)
        # scores = self.softmax(scores)
        return self.cross_entropy(scores,categories,node_mask),posterior.kl().mean()

    def cross_entropy(self,pred, label, mask):
        b, n_atom = label.size()
        atom_loss_f = nn.CrossEntropyLoss(reduction='none')
        rec_loss = atom_loss_f(pred, label.view(-1)) * mask.squeeze()
        return rec_loss.sum()/b

    def compute_loss(self, x, x_hat,node_mask,posterior):
        """
        auto-encoder的损失
        :param x: encoder的输入 原子类别（0~5）
        :param x_hat: decoder的输出 (b*n_nodes,6)
        :return: loss
        """
        b,n_atom = x.size()

        n_type = x_hat.size(-1)
        atom_loss_f = nn.CrossEntropyLoss(reduction='none')
        # rec_loss = atom_loss_f(x_hat.view(-1, n_type), x.view(-1))/b
        rec_loss = atom_loss_f(x_hat.view(-1, n_type), x.view(-1)) * node_mask.squeeze()
        rec_loss = rec_loss.sum()/b
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
        kl_loss = posterior.kl().mean()

        return rec_loss,kl_loss

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
