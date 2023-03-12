from torch import nn

from HGDM.graph_unet import UNet
from equivariant_diffusion import utils
import numpy as np
import math
import torch
from torch.nn import functional as F
from equivariant_diffusion import utils as diffusion_utils
from equivariant_diffusion.utils import remove_mean_with_mask


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    improved_diffusion cosine
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)
class HyperbolicDiffusion(nn.Module):
    def __init__(self, args, Encoder, Decoder,device,num_classes):
        super().__init__()
        self.T = args.T
        self.num_classes = num_classes
        self.noise_net = UNet(args.dim,n_layers=args.n_layers,hyp=args.hyp)
        self.dim = args.dim
        self.device = device
        betas = betas_for_alpha_bar(
            self.T,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
        self.register_buffer('betas', betas)
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas', torch.sqrt(alphas))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.encoder = Encoder
        self.decoder = Decoder
        self._edges_dict = {}
        self.loss_fn = nn.MSELoss()
        self.atom_decoder = torch.tensor([1, 6, 7, 8, 9], device=self.device)

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):

        categories, charges = h
        batch_size, n_nodes = categories.shape
        edges = self.get_adj_matrix(n_nodes, batch_size)
        posterior, _, _, _ = self.encoder(x, categories, edges, node_mask, edge_mask)
        h = posterior.mode()
        h = h.view(batch_size, n_nodes, -1)

        loss = self.compute_loss(h, node_mask, edge_mask, context, edge=edges)

        return loss

    def compute_loss(self, h, node_mask, edge_mask, context, edge=None):

        t = torch.randint(  # batch_size维度是不同的t 0~T-1
            0, self.T, size=(h.size(0),), device=h.device
        )

        noise = torch.randn_like(h) * node_mask
        h_t = extract(self.sqrt_alphas_cumprod, t, h.shape) * h +\
              extract(self.sqrt_one_minus_alphas_cumprod, t, h.shape) * noise

        net_out = self.noise_net(h_t, t, node_mask,edge,edge_mask, context)
        error = self.compute_error(net_out, noise.view(-1,noise.size(-1)))
        loss = error
        return loss

    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False):
        """
        Draw samples from the generative model.
        """
        edges = self.get_adj_matrix(n_nodes, n_samples)
        z = sample_gaussian_with_mask((n_samples, n_nodes, self.dim), node_mask.device, node_mask)
        for t in reversed(range(0, self.T)):
            Time = torch.ones((z.size(0),), dtype=torch.int64, device=z.device) * t
            beta_t = extract(self.betas, Time, z.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, Time, z.shape)
            sqrt_alphas_t = extract(self.sqrt_alphas, Time, z.shape)
            pred_noise = self.noise_net(z, Time, node_mask,edges, edge_mask, context)
            pred_noise = pred_noise.view(n_samples, n_nodes, self.dim)
            noise = sample_gaussian_with_mask((n_samples, n_nodes, self.dim), node_mask.device, node_mask)
            z = (z-beta_t/sqrt_one_minus_alphas_cumprod_t*pred_noise)/sqrt_alphas_t+beta_t*noise
        x,h = self.decode(z,node_mask,edge_mask)
        return x,h

    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None, outdim=9):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        z = sample_gaussian_with_mask((n_samples, n_nodes, self.dim), node_mask.device, node_mask)
        edges = self.get_adj_matrix(n_nodes, n_samples)
        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames, z.size(0), z.size(1), outdim), device=z.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for t in reversed(range(0, self.T)):
            Time = torch.ones((z.size(0),), dtype=torch.int64, device=z.device) * t
            beta_t = extract(self.betas, Time, z.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, Time, z.shape)
            sqrt_alphas_t = extract(self.sqrt_alphas, Time, z.shape)
            pred_noise = self.noise_net(z, Time, node_mask,edges, edge_mask, context)
            pred_noise = pred_noise.view(n_samples, n_nodes, self.dim)
            noise = sample_gaussian_with_mask((n_samples, n_nodes, self.dim), node_mask.device, node_mask)
            z = (z - beta_t / sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_t + beta_t * noise

            x, h = self.decode(z, node_mask, edge_mask)
            z_chain = torch.cat([x, h['categorical'], h['integer']],dim=2)
            write_index = (t * keep_frames) // self.T
            chain[write_index] = z_chain

        chain_flat = chain.view(n_samples * keep_frames, z.size(1), outdim)

        return chain_flat

    @torch.no_grad()
    def decode(self, z, node_mask, edge_mask):
        batch_size, n_nodes, dim = z.size()
        edges = self.get_adj_matrix(n_nodes, batch_size)
        h,x = self.decoder.decode(z.view(-1, dim), edges, node_mask.view(-1, 1),
                                     edge_mask)  # (b,n_nodes,max_z)  # max_z = 1 padding+5 types
        x = x.view(batch_size, n_nodes, -1)
        x = remove_mean_with_mask(x, node_mask)
        h = h.view(batch_size, n_nodes, -1)
        # torch.argmax(h_pred, dim=2) 0~5 -1 ->-1~4 mask ->0~4
        argmax = torch.argmax(h, dim=2, keepdim=True)
        idx = ((argmax - 1) * node_mask).long() * argmax.bool()

        h = F.one_hot(idx.squeeze(), self.num_classes) * node_mask

        charge = torch.gather(self.atom_decoder.view(1, 1, -1).repeat(batch_size, n_nodes, 1), 2, idx)
        charge = charge * node_mask
        # print(x.shape, charge.shape, h.shape)
        h = {'integer': charge.long(), 'categorical': h}

        return x,h
    def compute_error(self, net_out, eps):

        error = self.loss_fn(net_out,eps)
        return error

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

def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, dim=0, index=t).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked