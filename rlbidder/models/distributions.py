import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.distributions import (
    Independent,
    Normal,
    TransformedDistribution,
)
from torch.distributions.transforms import AffineTransform, SoftplusTransform, TanhTransform


def inv_softplus(bias: float) -> float:
    """Inverse softplus function using PyTorch with float32.

    Args:
        bias: the value to be softplus-inverted.
        
    Returns:
        The inverse softplus value
    """
    # Convert to torch tensor with float32 dtype
    bias_tensor = torch.as_tensor(bias, dtype=torch.float32)
    
    # Compute inverse softplus: log(exp(bias) - 1)
    # Using expm1 for numerical stability: expm1(x) = exp(x) - 1
    out = (
        bias_tensor
        .expm1()
        .clip(min=1e-6)  # Clip to avoid log(0)
        .log()
    )
    
    # Return scalar if input was scalar
    return out.item()


class BiasedSoftplus(nn.Module):
    """A biased softplus activation module.

    This module implements a softplus transformation with a configurable bias and minimum value.
    The bias parameter determines the output value when the input is zero.

    Args:
        bias (float): The desired output value when input is zero. The module computes
            an internal bias_shift such that softplus(0 + bias_shift) + min_val = bias.
        min_val (float): Minimum value of the transformation output. Defaults to 0.01.

    Example:
        >>> module = BiasedSoftplus(bias=1.0, min_val=0.01)
        >>> zero_input = torch.tensor(0.0)
        >>> output = module(zero_input)  # output will be approximately 1.0
    """

    def __init__(self, bias: float, min_val: float = 0.01) -> None:
        super().__init__()
        
        # Compute the bias shift to achieve desired output for zero input
        # We want: softplus(0 + bias_shift) + min_val = bias
        # So: bias_shift = inv_softplus(bias - min_val)
        self.register_buffer('bias_shift', torch.tensor(inv_softplus(bias - min_val), dtype=torch.float32))
        self.min_val = min_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply biased softplus transformation.
        
        Args:
            x: Input tensor
            
        Returns:
            Transformed tensor with softplus(x + bias_shift) + min_val
        """
        return F.softplus(x + self.bias_shift) + self.min_val

    def extra_repr(self) -> str:
        """String representation of module parameters."""
        return f'bias_shift={self.bias_shift.item():.4f}, min_val={self.min_val}'


def inv_logit(p: torch.Tensor | float, eps: float = 1e-6) -> torch.Tensor:
    p = torch.as_tensor(p, dtype=torch.float32)
    p = p.clamp(eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)


class SigmoidRangeStd(nn.Module):
    """
    DreamerV3-style: std = min + (max-min) * sigmoid(gain * x + bias_logit)
    No exp/log in forward.
    """
    def __init__(
        self,
        min_std: float,
        max_std: float,
        default_std: float,
        out_features: int | None = None,
        gain_init: float = 1.0,
        trainable_gain: bool = False,
    ) -> None:
        super().__init__()
        assert max_std > min_std
        shape = (out_features,) if out_features is not None else (1,)
        frac = (default_std - min_std) / (max_std - min_std)
        self.register_buffer('min_std', torch.full(shape, min_std, dtype=torch.float32))
        self.register_buffer('max_std', torch.full(shape, max_std, dtype=torch.float32))
        self.register_buffer('bias_logit', inv_logit(frac))
        if trainable_gain:
            self.gain = nn.Parameter(torch.full(shape, gain_init, dtype=torch.float32))
        else:
            self.register_buffer('gain', torch.full(shape, gain_init, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(self.gain * logits + self.bias_logit)
        return self.min_std + (self.max_std - self.min_std) * s


class NormalHead(nn.Module):
    """
    This module has a linear layer to learn mean and standard deviation for a normal distribution.
    The standard deviation is processed via a configurable module to ensure positive values.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features (dimension of the distribution).
        bias (bool, optional): Whether to use bias in the linear layer. Defaults to True.
        std_module (str, optional): Type of std module to use. Options: "sigmoid_range" or "biased_softplus".
            Defaults to "sigmoid_range".
        std_softplus_bias (float, optional): For biased_softplus: bias parameter. For sigmoid_range: default_std.
            Defaults to 1.0.
        std_softplus_min (float, optional): For biased_softplus: minimum value after softplus.
            Defaults to 0.01.
        std_min (float, optional): Hard minimum clipping value for standard deviation.
            Defaults to 1e-4.
        std_max (float, optional): Hard maximum clipping value for standard deviation.
            Defaults to 1.0.
        std_gain_init (float, optional): For sigmoid_range: initial gain value. Defaults to 1.0.
        std_trainable_gain (bool, optional): For sigmoid_range: whether gain is trainable. Defaults to False.
    
    Example:
        >>> # Using sigmoid_range (default)
        >>> head = NormalHead(in_features=64, out_features=2)
        >>> x = torch.randn(32, 64)  # batch_size=32, features=64
        >>> mean, std = head(x)
        >>> mean.shape, std.shape
        (torch.Size([32, 2]), torch.Size([32, 2]))
        >>> # Using biased_softplus
        >>> head = NormalHead(in_features=64, out_features=2, std_module="biased_softplus")
    """

    def __init__(
        self,
        in_features: int, 
        out_features: int,
        bias: bool = True,
        std_module: str = "sigmoid_range",
        std_softplus_bias: float = 1.0,
        std_softplus_min: float = 0.01,
        std_min: float = 1e-4,
        std_max: float = 1.0,
        std_gain_init: float = 1.0,
        std_trainable_gain: bool = False,
    ) -> None:
        super().__init__()

        # Store minimum/maximum std values for hard clipping
        self.std_min = std_min
        self.std_max = std_max
        self.std_module_type = std_module

        self.fc = nn.Linear(in_features, out_features * 2, bias=bias)
        
        # Configure the std module based on type
        if std_module == "biased_softplus":
            # Biased softplus for nice gradients
            # https://github.com/tensorflow/probability/issues/751
            self.logits_to_std = BiasedSoftplus(
                bias=std_softplus_bias, 
                min_val=std_softplus_min,
            )
        elif std_module == "sigmoid_range":
            # DreamerV3-style sigmoid range std
            self.logits_to_std = SigmoidRangeStd(
                min_std=self.std_min, 
                max_std=self.std_max, 
                default_std=std_softplus_bias,
                gain_init=std_gain_init,
                trainable_gain=std_trainable_gain,
            )
        else:
            raise ValueError(f"Unknown std_module type: {std_module}. Must be 'sigmoid_range' or 'biased_softplus'")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std_logits = self.fc(x).chunk(2, dim=-1)
        
        # Process std through configured module and apply minimum/maximum clipping
        std = self.logits_to_std(std_logits).clip(min=self.std_min, max=self.std_max)

        return mean, std


class SafeTanhNoEps(autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(input: torch.Tensor):
        output = input.tanh()
        eps = torch.finfo(input.dtype).resolution
        lim = 1.0 - eps
        output = output.clamp(-lim, lim)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, *grad):
        grad = grad[0]
        (output,) = ctx.saved_tensors
        return (grad * (1 - output.pow(2)),)


class SafeaTanhNoEps(autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(tanh_val: torch.Tensor):
        eps = torch.finfo(tanh_val.dtype).resolution
        lim = 1.0 - eps
        output = tanh_val.clamp(-lim, lim)
        # ctx.save_for_backward(output)
        output = output.atanh()
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        tanh_val = inputs[0]
        eps = torch.finfo(tanh_val.dtype).resolution

        # ctx.mark_non_differentiable(ind, ind_inv)
        # # Tensors must be saved via ctx.save_for_backward. Please do not
        # # assign them directly onto the ctx object.
        ctx.save_for_backward(tanh_val)
        ctx.eps = eps

    @staticmethod
    def backward(ctx, *grad):
        grad = grad[0]
        (tanh_val,) = ctx.saved_tensors
        eps = ctx.eps
        lim = 1.0 - eps
        output = tanh_val.clamp(-lim, lim)
        return (grad / (1 - output.pow(2)),)

# Module-level function references 
safetanh_noeps = SafeTanhNoEps.apply
safeatanh_noeps = SafeaTanhNoEps.apply


class SafeTanhTransform(TanhTransform):
    """TanhTransform subclass that ensured that the transformation is numerically invertible."""

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return safetanh_noeps(x)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        return safeatanh_noeps(y)


class TanhNormal(TransformedDistribution):
    """
    TanhNormal distribution with upscaling for numerical stability.
    
    Subclasses PyTorch's TransformedDistribution to add location upscaling:
    - scaled_loc = tanh(loc / upscale) * upscale
    - This keeps tanh operations in the responsive region, avoiding saturation
    - Output is bounded to [-1, 1] by the TanhTransform
    
    Args:
        loc: Location parameter (mean of underlying normal)
        scale: Scale parameter (std of underlying normal)
        upscale: Upscaling factor for location. Default: 5.0
        event_dims: Number of event dimensions. Default: 1
        tanh_loc: Whether to apply upscaling to location. Default: False
        safe_tanh: Whether to use SafeTanhTransform for numerical stability. Default: True
        validate_args: Whether to validate parameters. Default: None (uses PyTorch default).
    
    Example:
        >>> loc = torch.tensor([0.0, 1.0])
        >>> scale = torch.tensor([0.5, 0.2])
        >>> dist = TanhNormal(loc, scale, upscale=5.0, tanh_loc=True)
        >>> sample = dist.sample()
        >>> log_prob = dist.log_prob(sample)
        >>> print(f"Sample: {sample}, Log prob: {log_prob}")
    """

    # arg_constraints = {
    #     "loc": constraints.real,
    #     "scale": constraints.greater_than(1e-6),
    # }

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        upscale: float = 5.0,
        event_dims: int = 1,
        tanh_loc: bool = False,
        safe_tanh: bool = True,
        validate_args: bool = None,
        safe_bound_eps: float = 1e-6,
    ):
        self.upscale = upscale
        self.tanh_loc = tanh_loc
        self._event_dims = event_dims
        self.safe_tanh = safe_tanh
        self.safe_bound_eps = safe_bound_eps
        self.safe_tanh_bound = 1.0 - safe_bound_eps

        # Apply upscaling to location
        if tanh_loc:
            scaled_loc = torch.tanh(loc / upscale) * upscale
        else:
            scaled_loc = loc
            
        # Create base normal distribution
        if event_dims > 0:
            base_dist = Independent(Normal(scaled_loc, scale), event_dims)
        else:
            base_dist = Normal(scaled_loc, scale)
        
        # Create TanhTransform (output bounded to [-1, 1])
        if safe_tanh:
            transform = SafeTanhTransform(cache_size=1)
        else:
            transform = TanhTransform(cache_size=1)
            
        # Initialize parent TransformedDistribution
        super().__init__(base_dist, transform, validate_args=validate_args)
        
        # Store original parameters for updates
        self._original_loc = loc
        self._original_scale = scale
        
    def update(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        """Update distribution parameters efficiently."""
        # Apply upscaling
        if self.tanh_loc:
            scaled_loc = torch.tanh(loc / self.upscale) * self.upscale
        else:
            scaled_loc = loc
            
        # Update base distribution parameters directly (more efficient than recreating)
        if hasattr(self.base_dist, 'base_dist'):  # Independent wrapper
            self.base_dist.base_dist.loc = scaled_loc
            self.base_dist.base_dist.scale = scale
        else:  # Direct Normal
            self.base_dist.loc = scaled_loc
            self.base_dist.scale = scale
            
        # Store for reference
        self._original_loc = loc
        self._original_scale = scale

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(value.clip(-self.safe_tanh_bound, self.safe_tanh_bound))

    def rsample_and_log_prob(
        self, 
        sample_shape: torch.types._size = torch.Size(),
        batch_first: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample and log probability."""
        # Sample from the distribution
        samples = self.rsample(sample_shape)

        log_probs = self.log_prob(samples.clip(-self.safe_tanh_bound, self.safe_tanh_bound))
        
        # Handle sample_shape swapping
        if len(sample_shape) > 0 and batch_first:
            # Swap axes to make batch dimension first
            samples = samples.swapaxes(0, 1)
            log_probs = log_probs.swapaxes(0, 1)
        
        return samples, log_probs

    @property
    def original_loc(self) -> torch.Tensor:
        """Return the original (unscaled) location parameter."""
        return self._original_loc
        
    @property 
    def original_scale(self) -> torch.Tensor:
        """Return the original scale parameter."""
        return self._original_scale
        
    @property
    def scaled_loc(self) -> torch.Tensor:
        """Return the scaled location parameter actually used in the distribution."""
        if hasattr(self.base_dist, 'base_dist'):
            return self.base_dist.base_dist.loc
        else:
            return self.base_dist.loc
    
    @property
    def root_dist(self):
        """Get the root normal distribution (unwrapping Independent if present)."""
        if hasattr(self.base_dist, 'base_dist'):
            return self.base_dist.base_dist  # Unwrap Independent
        else:
            return self.base_dist  # Direct Normal
    
    @property
    def deterministic_sample(self) -> torch.Tensor:
        """
        Deterministic sample by applying all transforms to the mean of the root distribution.
        
        This property computes what would be the output if we took the mean of the underlying
        normal distribution and passed it through the tanh transform.
        """
        m = self.root_dist.mean
        for t in self.transforms:
            m = t(m)
        return m

    def __repr__(self) -> str:
        parts = [
            f'upscale={self.upscale}',
            f'tanh_loc={self.tanh_loc}',
            f'base_dist={self.base_dist}',
            f'transforms={self.transforms}'
        ]
        return f'{self.__class__.__name__}({", ".join(parts)})'


class BiasedSoftplusNormal(TransformedDistribution):
    """
    y = softplus(x + bias_shift) + min_val,  x ~ Normal(loc, scale).
    - bias/min_val let you set the deterministic output at zero input: softplus(bias_shift) + min_val = bias.
    - Positive, unbounded, pathwise gradients; avoids exp() on the action path.
    """
    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        event_dims: int = 1,
        bias: float | None = None,         # desired output at x=0 (after adding min_val)
        min_val: float = 0.0,              # additive floor after softplus
        validate_args: bool | None = None,
        safe_bound_eps: float = 1e-6,
    ):
        self._event_dims = int(event_dims)
        self.safe_bound_eps = float(safe_bound_eps)
        self.min_val = float(min_val)
        self.bias = None if bias is None else float(bias)

        # Base Normal (optionally wrapped in Independent)
        if self._event_dims > 0:
            base = Independent(Normal(loc, scale), self._event_dims)
        else:
            base = Normal(loc, scale)

        # Compose transforms: (x + bias_shift) -> softplus -> (+ min_val)
        transforms = []
        if self.bias is not None or self.min_val != 0.0:
            # If bias is provided, compute shift so that softplus(shift) + min_val = bias
            bias_shift = 0.0
            if self.bias is not None:
                bias_shift = inv_softplus(self.bias - self.min_val)
            if bias_shift != 0.0:
                transforms.append(AffineTransform(loc=bias_shift, scale=1.0))

        transforms.append(SoftplusTransform(cache_size=1))

        if self.min_val != 0.0:
            transforms.append(AffineTransform(loc=self.min_val, scale=1.0))

        super().__init__(base, transforms, validate_args=validate_args)

        self._original_loc = loc
        self._original_scale = scale

    def update(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        if hasattr(self.base_dist, "base_dist"):          # Independent wrapper
            self.base_dist.base_dist.loc = loc
            self.base_dist.base_dist.scale = scale
        else:
            self.base_dist.loc = loc
            self.base_dist.scale = scale
        self._original_loc = loc
        self._original_scale = scale

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(value.clamp_min(self.safe_bound_eps))

    def rsample_and_log_prob(
        self,
        sample_shape: torch.types._size = torch.Size(),
        batch_first: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        samples = self.rsample(sample_shape)
        log_probs = self.log_prob(samples)
        if len(sample_shape) > 0 and batch_first:
            samples = samples.swapaxes(0, 1)
            log_probs = log_probs.swapaxes(0, 1)
        return samples, log_probs

    @property
    def original_loc(self) -> torch.Tensor:
        return self._original_loc

    @property
    def original_scale(self) -> torch.Tensor:
        return self._original_scale

    @property
    def root_dist(self):
        return self.base_dist.base_dist if hasattr(self.base_dist, "base_dist") else self.base_dist

    @property
    def deterministic_sample(self) -> torch.Tensor:
        m = self.root_dist.mean
        for t in self.transforms:
            m = t(m)
        return m

    def __repr__(self) -> str:
        parts = [
            f'base_dist={self.base_dist}',
            f'transforms={self.transforms}',
            f'bias={self.bias}',
            f'min_val={self.min_val}',
        ]
        return f'{self.__class__.__name__}({", ".join(parts)})'
