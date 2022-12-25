import torch
from torch.distributions import Normal
import torch.nn.functional as F


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, manifold=None, node_mask=None):
        self.parameters = parameters
        self.mean, self.std = torch.chunk(parameters, 2, dim=1)
        # self.logvar = torch.clamp(self.logvar, -20.0, 5.0)
        self.node_mask = node_mask
        self.std = F.softplus(self.std)
        # self.var = torch.exp(self.logvar)
        eps = torch.finfo(self.std.dtype).eps
        self.std = torch.clamp(self.std, min=eps, max=4.0)
        self.manifold = manifold
        self.origin = self.manifold.origin((self.mean.size()))
        if manifold is not None:
            self.base = Normal(
                torch.zeros(
                    self.mean.size(),
                    device=self.mean.device
                )[..., 1:],
                self.std[..., 1:]
            )
            #             try:
            #                 self.base = Normal(
            #                     torch.zeros(
            #                         self.mean.size(),
            #                         device=self.mean.device
            #                     )[..., 1:],
            #                     self.std[..., 1:]
            #                 )
            #             except ValueError as e:
            #                 torch.set_printoptions(profile="full")
            #                 print(self.std)  # prints the whole tensor
            #                 torch.set_printoptions(profile="default")
            #                 print(e)
            self.mean = self.manifold.expmap0(self.proj_tan0(self.mean))
        else:
            self.base = Normal(
                torch.zeros(
                    self.mean.size(),
                    device=self.mean.device
                ),
                self.std
            )

    def sample(self):
        if self.manifold is not None:
            std = self.base.sample()
            zeros = torch.zeros((std.size(0), 1), device=std.device)
            std = torch.concat([zeros, std], dim=-1)
            std = self.manifold.transp0(self.mean, std)
            x = self.manifold.expmap(self.mean, std)
        else:
            x = self.mean + self.base.sample()

        x = x * self.node_mask
        return x

    def proj_tan0(self, u):
        return self.manifold.proju(self.origin, u)

    def kl(self, x):
        log_prob_base = self.log_prob(x)
        log_prob_target = self.log_prob_prior(x)
        kl = log_prob_base - log_prob_target

        # kl = 0.5 * torch.mean(torch.pow(self.mean, 2)
        #                       + self.var - 1.0 - self.logvar,
        #                       dim=-1, keepdim=True)
        kl = kl * self.node_mask

        return torch.clamp(kl.squeeze(), min=0)

    def log_prob(self, x):
        u = self.manifold.logmap(self.mean, x)
        v = self.manifold.transp(self.mean, self.origin, u)
        log_prob_v = self.base.log_prob(v[:, 1:]).sum(-1)

        r = self.manifold.norm(u)
        log_det = (u.size(-1) - 1) * (torch.sinh(r).log() - r.log())

        log_prob_z = log_prob_v - log_det
        return log_prob_z

    def log_prob_prior(self, x):
        prior_mean = torch.zeros(
            self.mean.size(),
            device=self.mean.device
        )
        prior_std = torch.ones_like(prior_mean, device=prior_mean.device)
        prior_base = Normal(
            prior_mean[..., 1:],
            prior_std[..., 1:]
        )
        u = self.manifold.logmap0(x)
        log_prob_v = prior_base.log_prob(u[:, 1:]).sum(-1)
        r = self.manifold.norm(u)
        log_det = (u.size(-1) - 1) * (torch.sinh(r).log() - r.log())

        log_prob_z = log_prob_v - log_det
        return log_prob_z

    def mode(self):

        x = self.mean
        x = x * self.node_mask
        return x
