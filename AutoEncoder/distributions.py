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


    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        if self.manifold.name == 'Hyperboloid':
            narrowed = x.narrow(-1, 0, 1)
            vals = torch.zeros_like(x)
            vals[:, 0:1] = narrowed
            x = x - vals

        x = x * self.node_mask
        return x

    def kl(self):
        if self.manifold.name == 'Hyperboloid':
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
        if self.manifold.name == 'Hyperboloid':
            narrowed = x.narrow(-1, 0, 1)
            vals = torch.zeros_like(x)
            vals[:, 0:1] = narrowed
            x = x - vals
        x = x * self.node_mask
        return x


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
