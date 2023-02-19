import torch
import torch.nn
from . import lorentz_math as math
import geoopt
from geoopt.manifolds.base import Manifold, ScalingInfo

__all__ = ["Lorentz"]

_lorentz_ball_doc = r"""
    Lorentz model

    Parameters
    ----------
    k : float|tensor
        manifold negative curvature

    Notes
    -----
    It is extremely recommended to work with this manifold in double precision
"""


class Lorentz:
    __doc__ = r"""{}
    """.format(
        _lorentz_ball_doc
    )

    ndim = 1
    reversible = False
    name = "Lorentz"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self, k=1.0,learnable=False, k_is_vary=False):
        super().__init__()
        k = torch.as_tensor(k)
        self.k_is_vary = k_is_vary
        # if not torch.is_floating_point(k):
        #     k = k.to(torch.get_default_dtype())
        # if k_is_vary:
        #     self.k = k.to(torch.float64)
        # else:
        #     self.k = torch.nn.Parameter(k.to(torch.float64), requires_grad=learnable)
        # self.k = torch.nn.Parameter(k.to(torch.float64), requires_grad=learnable)
        self.k = k.to(torch.float64)




    def set_k(self,k):
        # if self.k_is_vary:
        #     self.k = k.to(torch.float64)
        # else:
        #     self.k.data = k.to(torch.float64)
        # self.k.data = k.to(torch.float64)
        self.k = k.to(torch.float64)
    def dist(
            self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1,expand_k=False
    ) -> torch.Tensor:
        x = x.to(torch.float64)
        y = y.to(torch.float64)
        return math.dist(x, y, k=self.k, keepdim=keepdim, dim=dim,expand_k=expand_k).to(torch.get_default_dtype())

    @__scaling__(ScalingInfo(1))
    def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        x = x.to(torch.float64)
        return math.dist0(x, k=self.k, dim=dim, keepdim=keepdim).to(torch.get_default_dtype())

    def norm(self, u: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        u = u.to(torch.float64)
        return math.norm(u, keepdim=keepdim, dim=dim).to(torch.get_default_dtype())

    def projx(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        x = x.to(torch.float64)
        return math.project(x, k=self.k, dim=dim,k_is_vary=self.k_is_vary).to(torch.get_default_dtype())

    def proju(self, x: torch.Tensor, v: torch.Tensor, *, dim=-1,expand_k=False) -> torch.Tensor:
        x = x.to(torch.float64)
        v = v.to(torch.float64)
        k = self.k
        if expand_k:
            b_n_nodes = k.size(0)
            n_nodes = x.size(0) // b_n_nodes
            b = b_n_nodes // n_nodes
            k = k.view(b, n_nodes)
            k = k.repeat(1, n_nodes).view(-1, 1)

        v = math.project_u(x, v, k=k, dim=dim).to(torch.get_default_dtype())
        return v

    def expmap(
            self, x: torch.Tensor, u: torch.Tensor, *, norm_tan=True, project=True, dim=-1
    ) -> torch.Tensor:
        x = x.to(torch.float64)
        u = u.to(torch.float64)
        if norm_tan is True:
            u = self.proju(x, u, dim=dim)
        res = math.expmap(x, u, k=self.k, dim=dim)
        if project is True:
            return math.project(res, k=self.k, dim=dim,k_is_vary=self.k_is_vary).to(torch.get_default_dtype())
        else:
            return res.to(torch.get_default_dtype())

    @__scaling__(ScalingInfo(u=-1))
    def expmap0(self, u: torch.Tensor, *, project=True, dim=-1) -> torch.Tensor:
        u = u.to(torch.float64)
        res = math.expmap0(u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim,k_is_vary=self.k_is_vary).to(torch.get_default_dtype())
        else:
            return res.to(torch.get_default_dtype())

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1,expand_k=False) -> torch.Tensor:
        x = x.to(torch.float64)
        y = y.to(torch.float64)
        return math.logmap(x, y, k=self.k, dim=dim,expand_k=expand_k).to(torch.get_default_dtype())

    @__scaling__(ScalingInfo(1))
    def logmap0(self, y: torch.Tensor, *, dim=-1,expand_k=False) -> torch.Tensor:
        y = y.to(torch.float64)

        return math.logmap0(y, k=self.k, dim=dim,expand_k=expand_k).to(torch.get_default_dtype())

    def logmap0back(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        x = x.to(torch.float64)
        return math.logmap0back(x, k=self.k, dim=dim).to(torch.get_default_dtype())

    def transp0(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        y = y.to(torch.float64)
        u = u.to(torch.float64)
        return math.parallel_transport0(y, u, k=self.k, dim=dim).to(torch.get_default_dtype())

    def origin(
            self, *size, dtype=None, device=None, seed=42
    ) -> "geoopt.ManifoldTensor":
        """
        Zero point origin.

        Parameters
        ----------
        size : shape
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : int
            ignored

        Returns
        -------
        ManifoldTensor
            zero point on the manifold
        """
        if dtype is None:
            dtype = self.k.dtype
        if device is None:
            device = self.k.device

        zero_point = torch.zeros(*size, dtype=dtype, device=device)
        zero_point[..., 0] = torch.sqrt(self.k).squeeze()
        return geoopt.ManifoldTensor(zero_point, manifold=self)

    retr = expmap
    def to_poincare(self,x, dim=-1):
        x = x.to(torch.float64)
        return math.lorentz_to_poincare(x,k=self.k, dim=dim).to(torch.get_default_dtype())
