import torch
import torch.nn
from typing import Tuple, Optional
from . import math
import geoopt
from geoopt.manifolds.base import Manifold, ScalingInfo
from geoopt.utils import size2shape

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


class Lorentz(Manifold):
    __doc__ = r"""{}
    """.format(
        _lorentz_ball_doc
    )

    ndim = 1
    reversible = False
    name = "Lorentz"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self, k=1.0, k_is_vary=False):
        super().__init__()
        k = torch.as_tensor(k)
        self.k_is_vary = k_is_vary
        if not torch.is_floating_point(k):
            k = k.to(torch.get_default_dtype())
        self.k = k




    def dist(
            self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1,expand_k=False
    ) -> torch.Tensor:
        return math.dist(x, y, k=self.k, keepdim=keepdim, dim=dim,expand_k=expand_k)

    @__scaling__(ScalingInfo(1))
    def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        return math.dist0(x, k=self.k, dim=dim, keepdim=keepdim)

    def norm(self, u: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        return math.norm(u, keepdim=keepdim, dim=dim)

    def projx(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.project(x, k=self.k, dim=dim,k_is_vary=self.k_is_vary)

    def proju(self, x: torch.Tensor, v: torch.Tensor, *, dim=-1) -> torch.Tensor:
        v = math.project_u(x, v, k=self.k, dim=dim)
        return v

    def expmap(
            self, x: torch.Tensor, u: torch.Tensor, *, norm_tan=True, project=True, dim=-1
    ) -> torch.Tensor:
        if norm_tan is True:
            u = self.proju(x, u, dim=dim)
        res = math.expmap(x, u, k=self.k, dim=dim)
        if project is True:
            return math.project(res, k=self.k, dim=dim,k_is_vary=self.k_is_vary)
        else:
            return res

    @__scaling__(ScalingInfo(u=-1))
    def expmap0(self, u: torch.Tensor, *, project=True, dim=-1) -> torch.Tensor:
        res = math.expmap0(u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim,k_is_vary=self.k_is_vary)
        else:
            return res

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1,expand_k=False) -> torch.Tensor:
        return math.logmap(x, y, k=self.k, dim=dim,expand_k=expand_k)

    @__scaling__(ScalingInfo(1))
    def logmap0(self, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap0(y, k=self.k, dim=dim)

    def logmap0back(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap0back(x, k=self.k, dim=dim)

    def inner(
            self,
            x: torch.Tensor,
            u: torch.Tensor,
            v: torch.Tensor = None,
            *,
            keepdim=False,
            dim=-1,
    ) -> torch.Tensor:
        # TODO: x argument for maintaining the support of optims
        if v is None:
            v = u
        return math.inner(u, v, dim=dim, keepdim=keepdim)

    def inner0(
            self,
            v: torch.Tensor = None,
            *,
            keepdim=False,
            dim=-1,
    ) -> torch.Tensor:
        return math.inner0(v, k=self.k, dim=dim, keepdim=keepdim)

    def transp(
            self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        return math.parallel_transport(x, y, v, k=self.k, dim=dim)

    def transp0(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.parallel_transport0(y, u, k=self.k, dim=dim)

    def transp0back(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.parallel_transport0back(x, u, k=self.k, dim=dim)

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
