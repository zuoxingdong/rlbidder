import copy
from collections.abc import Iterable
from itertools import zip_longest

import numpy as np
import torch
import torch.nn as nn

from rlbidder.models.distributions import NormalHead
from rlbidder.models.losses import HLGaussLoss


def init_linear_module(
    module: nn.Module,
    orthogonal_init: bool = False,
    gain: float | None = None,
) -> None:
    """Initialize ``nn.Linear`` weights and bias.

    Args:
        module: Module to initialize (ignored when not ``nn.Linear``).
        orthogonal_init: Use orthogonal initialization when ``True``.
        gain: Optional gain factor applied to the initializer.
    """
    if not isinstance(module, nn.Linear):
        return
        
    if orthogonal_init:
        # Orthogonal initialization with default gain of sqrt(2)
        actual_gain = gain if gain is not None else np.sqrt(2)
        nn.init.orthogonal_(module.weight, gain=actual_gain)
    else:
        # Xavier uniform initialization with default gain of 1.0
        actual_gain = gain if gain is not None else 1.0
        nn.init.xavier_uniform_(module.weight, gain=actual_gain)
    
    # Always initialize bias to zero
    nn.init.zeros_(module.bias)


# Recommended by OLMoE Paper
# https://arxiv.org/pdf/2409.02060 - page 15
# https://github.com/allenai/OLMo/blob/main/olmo/initialization.py
def init_trunc_normal(
    module: nn.Linear | nn.Embedding,
    std: float,
    init_cutoff_factor: float | None = 3.0,
) -> None:
    if not isinstance(module, (nn.Linear, nn.Embedding)):
        return
    
    # weights
    if init_cutoff_factor is not None:
        cutoff_value = init_cutoff_factor * std
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
    else:
        nn.init.normal_(module.weight, mean=0.0, std=std)

    # biases
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


class LearnableScalar(nn.Module):
    """A learnable scalar parameter that can be used as a trainable constant.
    
    This module wraps a single scalar value as a learnable parameter, useful for
    hyperparameters that need to be learned during training (e.g., temperature
    parameters, scaling factors, etc.).
    
    Args:
        init_value: Initial value for the scalar parameter. If log_space=True,
            this should be the log of the desired initial value.
        requires_grad: Whether the parameter should be trainable
        device: Device to place the parameter on (e.g., 'cpu', 'cuda')
        log_space: If True, learns in log space and applies exp() to get the actual value.
        min_log_value: Minimum log value when log_space=True (for clipping).
        max_log_value: Maximum log value when log_space=True (for clipping).
    """
    
    def __init__(
        self, 
        init_value: float, 
        requires_grad: bool = True, 
        device: torch.device | str | None = None,
        log_space: bool = False,
        min_log_value: float | None = None,
        max_log_value: float | None = None,
    ) -> None:
        super().__init__()
        self.log_space = log_space
        self.min_log_value = min_log_value
        self.max_log_value = max_log_value
        
        self.value = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32, device=device),
            requires_grad=requires_grad
        )

    def forward(self, return_log: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.log_space:
            log_value = self.value.clip(self.min_log_value, self.max_log_value)
            if return_log:
                return log_value.exp(), log_value
            else:
                return log_value.exp()
        else:
            return self.value


class StochasticActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: list[int] = [256, 256, 256],
        std_module: str = "sigmoid_range",
        std_softplus_bias: float = 1.0,
        std_softplus_min: float = 0.01,
        std_min: float = 1e-4,
        std_max: float = 1.0,
        std_gain_init: float = 1.0,
        std_trainable_gain: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim

        layers = []
        layer_sizes = [state_dim] + hidden_sizes
        for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers += [
                nn.Linear(in_features, out_features),
                nn.RMSNorm(out_features),
                nn.SiLU(),
            ]
        layers += [
            NormalHead(
                hidden_sizes[-1], 
                action_dim,
                bias=True,
                std_module=std_module,
                std_softplus_bias=std_softplus_bias,
                std_softplus_min=std_softplus_min,
                std_min=std_min,
                std_max=std_max,
                std_gain_init=std_gain_init,
                std_trainable_gain=std_trainable_gain,
            )
        ]
        self.actor = nn.Sequential(*layers)
        
    def init_weights(self) -> None:
        self.actor[:-1].apply(lambda m: init_trunc_normal(m, std=1.0))
        self.actor[-1].apply(lambda m: init_trunc_normal(m, std=1e-2))  # small initial ranges of mean/std

    def forward(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.actor(state)
        return mean, std


class ValueNetwork(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        hidden_sizes: list[int] = [256, 256, 256],
        layer_norm: bool = False,
    ):
        super().__init__()
        
        layers = [
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.SiLU(),
            nn.RMSNorm(hidden_sizes[0]) if layer_norm else nn.Identity(),
        ]
        for in_features, out_features in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers += [
                nn.Linear(in_features, out_features),
                nn.SiLU(),
                nn.RMSNorm(out_features) if layer_norm else nn.Identity(),
            ]
        layers += [
            nn.Linear(hidden_sizes[-1], 1)
        ]
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.net[:-1].apply(lambda m: init_linear_module(m, orthogonal_init=True, gain=np.sqrt(2)))
        self.net[-1].apply(lambda m: init_linear_module(m, orthogonal_init=True, gain=1e-2))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_sizes: list[int] = [256, 256, 256],
        output_dim: int = 1,
        layer_norm: bool = False,
        last_layer_gain: float = 1e-2,
    ):
        super().__init__()

        latent_dim = (hidden_sizes[0] // 2)  
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            nn.SiLU(),
            nn.RMSNorm(latent_dim) if layer_norm else nn.Identity(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, latent_dim),
            nn.SiLU(),
            nn.RMSNorm(latent_dim) if layer_norm else nn.Identity(),
        )

        layers = []
        for in_features, out_features in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers += [
                nn.Linear(in_features, out_features),
                nn.SiLU(),
                nn.RMSNorm(out_features) if layer_norm else nn.Identity(),
            ]
        layers += [
            nn.Linear(hidden_sizes[-1], output_dim)
        ]
        self.net = nn.Sequential(*layers)

        self.state_encoder.apply(lambda m: init_linear_module(m, orthogonal_init=True, gain=np.sqrt(2)))
        self.action_encoder.apply(lambda m: init_linear_module(m, orthogonal_init=True, gain=np.sqrt(2)))
        self.net[:-1].apply(lambda m: init_linear_module(m, orthogonal_init=True, gain=np.sqrt(2)))
        self.net[-1].apply(lambda m: init_linear_module(m, orthogonal_init=True, gain=last_layer_gain))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 3:
            # Repeat state along the N dimension to match action shape
            # HACK: expand is faster than repeat!
            state = state.unsqueeze(1).expand(-1, action.shape[1], -1)
        x = torch.cat([self.state_encoder(state), self.action_encoder(action)], dim=-1)
        q_value = self.net(x).squeeze(dim=-1)
        return q_value


class EnsembledQNetwork(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        num_models: int = 10,
        hidden_sizes: list[int] = [256, 256, 256],
        output_dim: int = 1,
        layer_norm: bool = False,
        last_layer_gain: float = 1e-2,
        vmin: float = -5,
        vmax: float = 10,
        sigma_to_bin_width_ratio: float = 0.75,
        prior_scale: float = 40.9,
    ):
        super().__init__()
        
        self.num_models = num_models
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.layer_norm = layer_norm
        self.prior_scale = prior_scale

        # Create individual models
        models = [
            QNetwork(state_dim, action_dim, hidden_sizes, output_dim, layer_norm, last_layer_gain) 
            for _ in range(num_models)
        ]
        
        # Stack parameters and buffers for vmap
        params, buffers = torch.func.stack_module_state(models)
        # Register parameters and buffers so they can be tracked by state_dict
        for name, param in params.items():
            self.register_parameter(self._format_name_for_register(name), nn.Parameter(param, requires_grad=True))
        for name, buffer in buffers.items():
            self.register_buffer(self._format_name_for_register(name), buffer)

        # Create meta model for functional_call
        base_model = copy.deepcopy(models[0]).to('meta')  # noqa: F821
        def fmodel(params, buffers, inputs):
            """Single model forward pass for vmap"""
            return torch.func.functional_call(base_model, (params, buffers), inputs)
        self._fmodel = fmodel
        self.base_model_repr = str(base_model)

        self.use_categorical_critic = output_dim > 1
        if self.use_categorical_critic:
            self.hlg_loss = HLGaussLoss(
                vmin=vmin,
                vmax=vmax,
                num_atoms=output_dim,
                sigma_to_bin_width_ratio=sigma_to_bin_width_ratio,
            )
            # NOTE: trainable prior - zero distribution
            self.prior_logits = nn.Parameter(
                self.hlg_loss.transform_to_probs(torch.zeros(num_models, 1)).squeeze(),  # (num_atoms, )
                requires_grad=True,
            )
        
    @staticmethod
    def _format_name_for_register(name: str) -> str:
        """Format parameter names to be valid for registration"""
        return "ensemble_" + name.replace('.', '__')
    
    @staticmethod
    def _format_name_for_vmap(name: str) -> str:
        """Format parameter names for vmap"""
        return name.removeprefix('ensemble_').replace('__', '.')
    
    def _prepare_vmap_params(self):
        # Filter out HLGauss-related parameters
        params = {
            EnsembledQNetwork._format_name_for_vmap(name): param 
            for name, param in self.named_parameters()
            if not name.startswith('hlg_loss') and not name.startswith('prior_logits')
        }
        return params
    
    def _prepare_vmap_buffers(self):
        # Filter out HLGauss-related parameters
        buffers = {
            EnsembledQNetwork._format_name_for_vmap(name): buffer
            for name, buffer in self.named_buffers()
            if not name.startswith('hlg_loss') and not name.startswith('prior_logits')
        }
        return buffers
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all models using vmap
        
        Returns:
            torch.Tensor: Shape (num_models, batch_size) or (num_models, batch_size, N) 
                         depending on action dimensions
        """
        inputs = (state, action)
        predictions = torch.vmap(self._fmodel, in_dims=(0, 0, None))(
            self._prepare_vmap_params(), self._prepare_vmap_buffers(), inputs
        )
        
        if self.use_categorical_critic:
            n_expand_dims = predictions.ndim - 2
            logits = predictions + self.prior_scale * self.prior_logits[:, *(None,)*n_expand_dims, :]
            values = (
                self.hlg_loss
                .transform_from_probs(
                    logits.flatten(end_dim=-2).softmax(dim=-1)
                )
                .reshape(*logits.shape[:-1], -1)
            )
            return logits, values
        
        return predictions
    
    def __repr__(self):
        return (
            "QNetworkEnsemble(\n"
            f"  ensemble_size={self.num_models},\n"
            f"  base_model={self.base_model_repr}\n"
            ")"
        )


def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


@torch.no_grad()
def polyak_update(
    params: Iterable[torch.Tensor],
    target_params: Iterable[torch.Tensor],
    tau: float,
) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93

    :param params: parameters to use to update the target params
    :param target_params: parameters to update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    # zip does not raise an exception if length of parameters does not match.
    for param, target_param in zip_strict(params, target_params):
        target_param.data.mul_(1 - tau)
        torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


class HLGaussHead(nn.Module):
    def __init__(
        self, 
        in_features: int,
        vmin: float, 
        vmax: float, 
        num_atoms: int, 
        sigma_to_bin_width_ratio: float = 0.75,
        prior_scale: float = 10.0,
    ):
        super().__init__()
        self.prior_scale = prior_scale

        self.hlg_loss = HLGaussLoss(
            vmin=vmin,
            vmax=vmax,
            num_atoms=num_atoms,
            sigma_to_bin_width_ratio=sigma_to_bin_width_ratio,
        )
        self.head = nn.Linear(in_features, num_atoms)
        
        # NOTE: trainable prior -  zero distribution
        self.prior_logits = nn.Parameter(
            self.hlg_loss.transform_to_probs(torch.zeros(1, 1)).squeeze(),  # (num_atoms, )
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.head(x) + self.prior_scale * self.prior_logits

        values = (
            self.hlg_loss.transform_from_probs(
                logits.flatten(end_dim=-2).softmax(dim=-1)
            )
            .reshape(*logits.shape[:-1], -1)
        )
        return logits, values
