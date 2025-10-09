import itertools
import logging
import math
import warnings
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LinearWarmupConstantCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a three-phase schedule:
    1. Linear warmup from warmup_start_lr to base_lr
    2. Constant learning rate at base_lr 
    3. Cosine annealing from base_lr to eta_min

    This scheduler supports the DeepSeek-V3 style learning rate schedule with warmup, constant, and cosine phases.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupConstantCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example:
        >>> import torch.nn as nn
        >>> from torch.optim import Adam
        >>> #
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> # Three-phase schedule: warmup -> constant -> cosine decay
        >>> scheduler = LinearWarmupConstantCosineAnnealingLR(
        ...     optimizer, 
        ...     warmup_epochs=2000, 
        ...     constant_epochs=100000,
        ...     max_epochs=160000
        ... )
        >>> # Simple case without constant phase (constant_epochs=0)
        >>> scheduler = LinearWarmupConstantCosineAnnealingLR(
        ...     optimizer, warmup_epochs=10, max_epochs=40, constant_epochs=0
        ... )

    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        constant_epochs: int = 0,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Number of epochs for linear warmup
            max_epochs (int): Total number of epochs
            constant_epochs (int): Number of epochs to keep constant lr after warmup. Default: 0.
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate after cosine decay. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.constant_epochs = constant_epochs
        
        # Calculate phase boundaries
        self.constant_start = warmup_epochs
        self.constant_end = warmup_epochs + constant_epochs
        self.cosine_start = self.constant_end
        self.cosine_end = max_epochs
        
        # Validate configuration
        if self.cosine_end <= self.cosine_start:
            raise ValueError("Not enough epochs for cosine decay phase. Increase max_epochs or reduce constant_epochs.")

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            msg = "To get the last learning rate computed by the scheduler; please use `get_last_lr()`."
            logger.warning(msg)
            warnings.warn(
                msg,
                UserWarning,
            )

        # Phase 1: Warmup
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        
        # Phase 2: Constant at base_lr
        if self.last_epoch < self.constant_end:
            return self.base_lrs
        
        # Phase 3: Cosine decay
        cosine_epochs = self.cosine_end - self.cosine_start
        current_cosine_epoch = self.last_epoch - self.cosine_start
        
        if (current_cosine_epoch - 1 - cosine_epochs) % (2 * cosine_epochs) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / cosine_epochs)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        
        return [
            (1 + math.cos(math.pi * current_cosine_epoch / cosine_epochs))
            / (
                1 + math.cos(math.pi * (current_cosine_epoch - 1) / cosine_epochs)
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> list[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        # Phase 1: Warmup
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]
        
        # Phase 2: Constant at base_lr
        if self.last_epoch < self.constant_end:
            return list(self.base_lrs)
        
        # Phase 3: Cosine decay
        cosine_epochs = self.cosine_end - self.cosine_start
        current_cosine_epoch = self.last_epoch - self.cosine_start
        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * current_cosine_epoch / cosine_epochs))
            for base_lr in self.base_lrs
        ]


def clip_grad_norm_fast(
    params: Iterable[torch.nn.Parameter] | torch.nn.Parameter,
    *more_params: Iterable[torch.nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0,
    foreach: bool = True,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    """
    Efficient grad clipping:
      - Accepts either a single parameter iterable (including generators) or multiple iterables.
      - Uses itertools.chain when multiple iterables are provided.
      - Computes total norm once and reuses it for clipping via clip_grads_with_norm_.

    Returns:
      Total grad norm as a Tensor.
    """
    all_params_iter = itertools.chain(params, *more_params) if more_params else params

    total_grad_norm = torch.nn.utils.clip_grad_norm_(
        all_params_iter,
        max_norm=max_norm,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite,
        foreach=foreach,
    )
    return total_grad_norm


def get_decay_and_no_decay_params(
    module: nn.Module, *more_modules: nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """
    Split parameters of one or multiple modules into two groups: decay and no-decay.

    Common practice to exclude from weight decay:
        * Bias terms (name endswith "bias")
        * Parameters with ndim <= 1 (e.g., LayerNorm/RMSNorm/BatchNorm weights, scalar params)
        * Any parameter whose name contains "norm" or "embedding"

    Args:
        module: First nn.Module.
        *more_modules: Optional additional nn.Module instances.

    Returns:
        (decay_params, no_decay_params): Two lists of Parameters without duplicates.
    """
    # Normalize to iterable of modules
    module_iter: Iterable[nn.Module] = itertools.chain((module,), more_modules)

    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    seen_param_ids: set[int] = set()

    for module in module_iter:
        for name, param in module.named_parameters(recurse=True):
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in seen_param_ids:
                continue
            seen_param_ids.add(pid)

            name_lower = name.lower()
            is_bias = name_lower.endswith("bias")
            is_one_dim = (param.ndim <= 1)
            is_norm_or_embed = ("norm" in name_lower) or ("embedding" in name_lower)

            if is_bias or is_one_dim or is_norm_or_embed:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    return decay_params, no_decay_params


def fix_global_step_for_multi_optimizers(trainer: torch.nn.Module, *optimizers: Optimizer) -> None:
    """
    Fix global step tracking with multiple optimizers in PyTorch Lightning.
    
    This is a workaround for the issue where global step is not properly tracked
    when using multiple optimizers with manual optimization.
    
    See: https://github.com/Lightning-AI/pytorch-lightning/issues/17958
    
    Args:
        trainer: The PyTorch Lightning trainer instance
        *optimizers: Variable number of optimizer instances to fix
    """
    for opt in optimizers:
        opt._on_before_step = lambda: trainer.profiler.start("optimizer_step")
        opt._on_after_step = lambda: trainer.profiler.stop("optimizer_step")
