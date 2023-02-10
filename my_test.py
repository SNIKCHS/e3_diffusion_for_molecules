import torch
from tqdm import tqdm
import numpy as np
from torch import nn
from torch.nn import functional as F
import geoopt

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    input in manifold
    output in manifold
    """

    def __init__(self, in_features, out_features, manifold):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self.bias = nn.Parameter(torch.Tensor(1, out_features))
        self.linear = nn.Linear(in_features, out_features, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=0.25)
        nn.init.constant_(self.bias, 0)

    def proj_tan0(self, u):
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
# class HypAct(nn.Module):
#     """
#     Hyperbolic activation layer.
#     input in manifold
#     output in manifold
#     """
#
#     def __init__(self, manifold, act):
#         super(HypAct, self).__init__()
#         self.manifold = manifold
#         self.act = act
#     def proj_tan0(self, u):
#         narrowed = u.narrow(-1, 0, 1)
#         vals = torch.zeros_like(u)
#         vals[:, 0:1] = narrowed
#         return u - vals
#     def forward(self, x):
#         xt = self.act(self.manifold.logmap0(x))
#         xt = self.proj_tan0(xt)
#         out = self.manifold.expmap0(xt)
#         return out
class HypLayer(nn.Module):
    def __init__(self, in_features, out_features, manifold_in, manifold_out, act):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.bias = nn.Parameter(torch.Tensor(1, out_features))
        self.linear = nn.Linear(in_features, out_features, bias=False)
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
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.bias, 0)

    def forward(self, h):

        h = self.HypLinear(h)
        # if torch.any(torch.isnan(h)):
        #     print('HypLinear nan')
        h = self.HNorm(h)
        # if torch.any(torch.isnan(h)):
        #     print('HNorm nan')
        h = self.HypAct(h)
        # if torch.any(torch.isnan(h)):
        #     print('HypAct nan')
        return h

    def HypLinear(self, x):
        x = self.manifold_in.logmap0(x)
        x = self.linear(x)
        x = self.proj_tan0(x)
        x = self.manifold_in.expmap0(x)

        bias = self.proj_tan0(self.bias.view(1, -1))
        bias = self.manifold_in.transp0(x, bias)
        res = self.manifold_in.expmap(x, bias)
        return res

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

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, dim=0, index=t).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
class HyperbolicDiffusion(nn.Module):

    def __init__(self,manifold,T=1000, beta_1=1e-4, beta_T=0.02):
        super(HyperbolicDiffusion, self).__init__()
        self.manifold =manifold
        self.denoise_net = nn.Sequential(
            HypLayer(11,20,manifold,manifold,nn.SiLU()),
            HypLayer(20,20,manifold,manifold,nn.SiLU()),
            HypLayer(20,20,manifold,manifold,nn.SiLU()),
            HypLayer(20,20,manifold,manifold,nn.SiLU()),
            HypLayer(20,20,manifold,manifold,nn.SiLU()),
            HypLayer(20,20,manifold,manifold,nn.SiLU()),
            HypLinear(20,10,manifold)
        )

        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())

        alphas = 1. - self.betas
        self.register_buffer(
            'sqrt_alphas', torch.sqrt(alphas))
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
    def proj_tan0(self, u):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals
    def forward(self, x):
        loss = self.compute_loss(x)

        return loss

    @torch.no_grad()
    def sample(self, h):
        z_t = torch.randn_like(h)
        z_t = self.proj_tan0(z_t)
        for t in reversed(range(self.T)):
            Time = torch.ones((h.size(0),), dtype=torch.int64, device=h.device) * t
            noise = torch.randn_like(h)
            input = torch.concat([z_t,Time[..., None]/self.T], dim=1)
            input = self.manifold.expmap0(input)
            pred_noise = self.denoise_net(input)

            pred_noise = self.manifold.logmap0(pred_noise)
            pred_noise = self.proj_tan0(pred_noise)

            sqrt_one_minus_alphas_bar = extract(self.sqrt_one_minus_alphas_bar, Time, h.shape)
            sqrt_alphas = extract(self.sqrt_alphas, Time, h.shape)
            betas = extract(self.betas, Time, h.shape)

            n = self.manifold.expmap0(betas / sqrt_one_minus_alphas_bar * pred_noise)
            z_t = self.manifold.expmap0(z_t)
            z_t = self.manifold.logmap(n,z_t)
            z_t = self.manifold.transp0back(n,z_t)/sqrt_alphas
            z_t = self.proj_tan0(z_t)

            n_new = self.manifold.expmap0(betas * noise)
            z_t = self.manifold.transp0(n_new,z_t)
            z_t = self.manifold.expmap(n_new,z_t)
            z_t = self.manifold.logmap0(z_t)
            z_t = self.proj_tan0(z_t)
            print('t:', t, ' z_t:', z_t[0])

    def compute_loss(self, h):
        h = self.manifold.logmap0(h)
        t = torch.randint(self.T, size=(h.shape[0],), device=h.device)
        noise = torch.randn_like(h)
        noise = self.proj_tan0(noise)
        x_t = extract(self.sqrt_alphas_bar, t, h.shape) * h
        noise_t = extract(self.sqrt_one_minus_alphas_bar, t, h.shape) * noise
        noise_t = self.manifold.expmap0(noise_t)
        x_t = self.manifold.transp0(noise_t,x_t)
        x_t = self.manifold.expmap(noise_t,x_t)
        t = t[..., None]/self.T
        x_t = self.manifold.logmap0(x_t)
        input = torch.concat([x_t,t], dim=1)
        input = self.manifold.expmap0(input)

        pred_noise = self.denoise_net(input)

        pred_noise = self.manifold.logmap0(pred_noise)
        pred_noise = self.proj_tan0(pred_noise)
        loss = F.mse_loss(pred_noise, noise, reduction='mean')

        return loss

def proj_tan0(u):
    narrowed = u.narrow(-1, 0, 1)
    vals = torch.zeros_like(u)
    vals[:, 0:1] = narrowed
    return u - vals




# manifold = geoopt.Lorentz(1, learnable=False)
# model = HyperbolicDiffusion(manifold)
#
# optimizer = torch.optim.Adam(model.parameters(), 5e-4)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=2000,
#     gamma=float(0.9)
# )
#
# x = torch.randn((200, 10), dtype=torch.float32)
# x = proj_tan0(x)
# x = manifold.expmap0(x)
# # x = torch.tensor([[1.5794, 2.5078, 0.0000, 0.0000, 1.7158, 1.0340, 0.0000, 1.3385, 0.0000,
# #         0.8585, 0.9584, 0.1076, 0.4893, 0.0000, 0.0000, 0.6972, 1.2082, 2.7626,
# #         0.0000, 0.0000]],device='cuda').repeat(200,1)
#
# x = x.to('cuda')
# model = model.to('cuda')
# for i in tqdm(range(10000)):
#     optimizer.zero_grad()
#     loss = model(x)
#     if i % 1000 == 0:
#         print(loss, lr_scheduler.get_last_lr())
#     loss.backward()
#     optimizer.step()
#     lr_scheduler.step()
#
# model.sample(x)
# print('-----------------------------')
# print(manifold.logmap0(x[0]))
# print('-----------------------------')
# print(x[0])