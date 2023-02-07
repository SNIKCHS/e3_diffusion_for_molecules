import torch


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, manifold=None,node_mask=None):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.node_mask = node_mask
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.manifold = manifold

    def proj_tan0(self, u):
        if self.manifold.name == 'Lorentz':
            narrowed = u.narrow(-1, 0, 1)
            vals = torch.zeros_like(u)
            vals[:, 0:1] = narrowed
            return u - vals
        else:
            return u

    def sample(self):
        if self.manifold is None:
            x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
            x = x * self.node_mask
            x = self.proj_tan0(x)
        else:
            mean = self.manifold.expmap0(self.proj_tan0(self.mean))
            std = self.std * torch.randn(mean.shape).to(device=self.parameters.device)
            std_t = self.manifold.transp0(mean,self.proj_tan0(std))
            x = self.manifold.expmap(mean,std_t)
            x = self.manifold.logmap0(x)
        return x

    def kl(self):
        if self.manifold is not None:
            kl = 0.5 * torch.mean(torch.pow(self.mean[...,1:], 2)
                               + self.var[...,1:] - 1.0 - self.logvar[...,1:],
                               dim=-1,keepdim=True)
        else:
            kl = 0.5 * torch.mean(torch.pow(self.mean, 2)
                                 + self.var - 1.0 - self.logvar,
                                 dim=-1, keepdim=True)

        kl = kl * self.node_mask

        return kl.squeeze()

    def mode(self):
        x = self.mean
        x = self.proj_tan0(x.clone())
        x = x * self.node_mask
        return x


