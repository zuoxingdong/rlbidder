import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HLGaussLoss(nn.Module):
    def __init__(
        self, 
        vmin: float, 
        vmax: float, 
        num_atoms: int, 
        sigma_to_bin_width_ratio: float = 0.75,
        weights: torch.Tensor | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.num_atoms = num_atoms
        self.eps = eps
        
        bin_width = (vmax - vmin) / (num_atoms - 1)
        self.sigma = bin_width * sigma_to_bin_width_ratio

        self.register_buffer(
            "support",
            torch.linspace(
                vmin - bin_width / 2,
                vmax + bin_width / 2,
                num_atoms + 1, 
                dtype=torch.float32,
            )
        )

        # TODO: support weights?
        if weights is None:
            weights = torch.ones(num_atoms)
        self.weights = weights
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        # HACK: Flatten is critical as cross_entropy doesn't handle broadcasting
        return F.cross_entropy(
            logits.flatten(end_dim=-2), 
            self.transform_to_probs(target).flatten(end_dim=-2).detach(), 
            reduction=reduction,
        )
    
    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        if target.ndim <= 1:
            raise ValueError(f"target (shape: {target.shape}) must be at least 2D")
        support = self.support[*(None,) * (target.ndim - 1), ...]
        cdf_evals = torch.erf(
            (support - target.clip(self.vmin, self.vmax))
            / (math.sqrt(2.0) * self.sigma + self.eps)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return (bin_probs / (z[..., None] + self.eps)).reshape(*target.shape[:-1], self.num_atoms)
    
    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)[..., None]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"vmin={self.vmin}, "
            f"vmax={self.vmax}, "
            f"num_atoms={self.num_atoms}, "
            f"sigma={self.sigma})"
        )
