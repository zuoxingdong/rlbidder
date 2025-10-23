from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_is_fitted


class BaseScaler(ABC, BaseEstimator, TransformerMixin):
    @abstractmethod
    def fit(self, data: np.ndarray) -> Self:  # pragma: no cover - abstract interface
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def _safe_tanh(x: np.ndarray, eps: float | None = None) -> np.ndarray:
    """
    Applies tanh to x and clamps the result to (-1 + eps, 1 - eps).
    If eps is None, use a default based on x's dtype.
    """
    if eps is None:
        # Use machine epsilon for the dtype of x
        eps = np.finfo(x.dtype).eps if np.issubdtype(x.dtype, np.floating) else 1e-6
    lim = 1.0 - eps
    y = np.tanh(x)
    return np.clip(y, -lim, lim)


def _safe_atanh(x: np.ndarray, eps: float | None = None) -> np.ndarray:
    """
    Applies arctanh to x after clamping to (-1 + eps, 1 - eps).
    If eps is None, use a default based on x's dtype.
    """
    if eps is None:
        eps = np.finfo(x.dtype).eps if np.issubdtype(x.dtype, np.floating) else 1e-6
    lim = 1.0 - eps
    return np.arctanh(np.clip(x, -lim, lim))


def _safe_log1p(x: np.ndarray, eps: float | None = None) -> np.ndarray:
    """
    Applies log1p to x and clamps the result to (eps, inf).
    If eps is None, use a default based on x's dtype.
    """
    if eps is None:
        # Use machine epsilon for the dtype of x
        eps = np.finfo(x.dtype).eps if np.issubdtype(x.dtype, np.floating) else 1e-6
    return np.log1p(np.clip(x, eps, None))


def _safe_expm1(x: np.ndarray, eps: float | None = None) -> np.ndarray:
    if eps is None:
        # Use machine epsilon for the dtype of x
        eps = np.finfo(x.dtype).eps if np.issubdtype(x.dtype, np.floating) else 1e-6
        
    # Lower bound for z, derived from the inverse log1p(eps)
    x_min = np.log1p(eps)

    # Upper bound for z, to prevent exp(z) from overflowing
    # np.expm1(z) is essentially np.exp(z) for large z.
    # The largest z for which np.exp(z) is not infinity.
    # np.finfo(dtype).max is the max representable float. exp(z) should not exceed this.
    # So, z should not exceed log(np.finfo(dtype).max).
    # (Actually, expm1(z) = exp(z) - 1, so exp(z) = expm1(z) + 1.
    # If expm1(z) can go up to np.finfo(dtype).max, then exp(z) can go up to np.finfo(dtype).max + 1.
    # So z_max could be np.log(np.finfo(dtype).max + 1).
    # However, np.log(big_num + 1) is very close to np.log(big_num).
    # And numpy's expm1 itself handles large inputs gracefully by returning inf.
    # Let's use the threshold where exp(z) itself is on the verge of inf.
    x_max = np.log(np.finfo(x.dtype).max) # Approx 709.78 for float64

    x_clipped = np.clip(x, x_min, x_max)
    return np.expm1(x_clipped)


class TanhTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for tanh and safe arctanh.
    """
    def __init__(self, eps: float = float(np.finfo(np.float32).eps)) -> None:
        self.eps = eps

    def fit(self, X: np.ndarray, y: Any | None = None) -> Self:
        # No fitting necessary for tanh/arctanh
        self.is_fitted_ = True  # convention: trailing underscore
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return _safe_tanh(X, eps=self.eps)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return _safe_atanh(X, eps=self.eps)
    
    def fit_transform(self, X: np.ndarray, y: Any | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for log1p and expm1 with safe handling for non-negative input.
    """
    def __init__(self, eps: float = float(np.finfo(np.float32).eps)) -> None:
        self.eps = eps

    def fit(self, X: np.ndarray, y: Any | None = None) -> Self:
        # No fitting necessary for log1p/expm1
        self.is_fitted_ = True  # convention: trailing underscore
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        # Ensure all values are >= 0 (within a small epsilon)
        if np.any(X < -self.eps):
            raise ValueError("Log1pTransformer input contains negative values.")
        return _safe_log1p(X, eps=self.eps)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return _safe_expm1(X, eps=self.eps)
    
    def fit_transform(self, X: np.ndarray, y: Any | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class AffineTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for general affine transformation: X * scale + bias.
    """
    def __init__(self, scale: float | np.ndarray, bias: float | np.ndarray) -> None:
        self.scale = scale
        self.bias = bias

    def fit(self, X: np.ndarray, y: Any | None = None) -> Self:
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return X * self.scale + self.bias

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return (X - self.bias) / self.scale
    
    def fit_transform(self, X: np.ndarray, y: Any | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class ClipTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that clips values to a specified min and/or max.
    If min or max is None, that bound is not applied.
    """
    def __init__(self, min_value: float | None = None, max_value: float | None = None) -> None:
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, X: np.ndarray, y: Any | None = None) -> Self:
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.clip(X, self.min_value, self.max_value)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        # Clipping is not invertible; return X clipped to same bounds
        X = np.asarray(X)
        return np.clip(X, self.min_value, self.max_value)

    def fit_transform(self, X: np.ndarray, y: Any | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that applies winsorization (clipping) based on quantiles.
    Clips each feature to the specified lower and upper quantiles.
    """
    def __init__(self, quantile_range: tuple[float, float] = (0.01, 0.99)) -> None:
        self.quantile_range = quantile_range

    def fit(self, X: np.ndarray, y: Any | None = None) -> Self:
        X = np.asarray(X)
        lower, upper = self.quantile_range
        self.lower_bounds_ = np.quantile(X, lower, axis=0)
        self.upper_bounds_ = np.quantile(X, upper, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        # Inverse transform is not well-defined for winsorization; return X unchanged
        return np.asarray(X)
    
    def fit_transform(self, X: np.ndarray, y: Any | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class SymlogTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for symlog and symexp transformations.
    Symlog is defined as sign(x) * log(1 + abs(x)), and its inverse is sign(y) * (exp(abs(y)) - 1).
    This is used in DreamerV3 and similar RL algorithms for compressing large-magnitude values.
    """
    def __init__(self, eps: float = float(np.finfo(np.float32).eps)) -> None:
        self.eps = eps

    def fit(self, X: np.ndarray, y: Any | None = None) -> Self:
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.sign(X) * np.log1p(np.abs(X) + self.eps)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.sign(X) * (np.expm1(np.abs(X)) - self.eps)

    def fit_transform(self, X: np.ndarray, y: Any | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class FeatureWiseJitter(BaseEstimator, TransformerMixin):
    """
    Randomly jitters each feature in each sample.
    For each feature j, jitter scale = scale_factor * (min positive gap of unique sorted values in feature j).
    If a feature is constant, its jitter scale is 0.

    Parameters
    - scale_factor: float, default 0.5
        Multiplier applied to the per-feature smallest positive gap.
    - distribution: {'uniform','normal'}, default 'uniform'
        Jitter distribution per feature:
          * 'uniform': noise ~ U(-scale_j, +scale_j)
          * 'normal' : noise ~ N(0, scale_j)
    - random_state: int | np.random.Generator | None, default None
        Controls reproducibility.
    - copy: bool, default True
        Whether to copy X before adding noise.
    - min_gap_clip: float, default 1e-4
        Minimum value to clip the computed minimum gaps.

    Attributes
    - min_gap_: np.ndarray of shape (n_features,)
        Smallest positive difference per feature (0 for constant features).
    - jitter_scale_: np.ndarray of shape (n_features,)
        scale_factor * min_gap_.
    """

    def __init__(
        self,
        scale_factor: float = 0.5,
        distribution: str = "uniform",
        random_state: int | np.random.Generator | None = None,
        copy: bool = True,
        min_gap_clip: float = 1e-4,
    ) -> None:
        self.scale_factor = float(scale_factor)
        self.distribution = distribution
        self.random_state = random_state
        self.copy = bool(copy)
        self.min_gap_clip = float(min_gap_clip)

        self._rng_ = np.random.Generator(np.random.PCG64(self.random_state))

    def fit(self, X: np.ndarray, y: Any | None = None) -> Self:
        X = check_array(X, copy=False, dtype=FLOAT_DTYPES, ensure_all_finite=True)
        _, D = X.shape
        min_gap = np.zeros(D, dtype=np.float64)

        for j in range(D):
            u = np.unique(X[:, j])
            if u.size >= 2:
                d = np.diff(u)
                # strictly positive gaps (guard against floating noise)
                d = d[d > 0]
                min_gap[j] = d.min() if d.size else 0.0
            else:
                min_gap[j] = 0.0

        self.min_gap_ = min_gap.clip(min=self.min_gap_clip)
        self.jitter_scale_ = self.scale_factor * self.min_gap_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("jitter_scale_", "_rng_"))
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES, ensure_all_finite=True)
        N, D = X.shape
        if D != self.jitter_scale_.shape[0]:
            raise ValueError(f"X has {D} features, expected {self.jitter_scale_.shape[0]}.")

        scales = self.jitter_scale_[None, :]  # shape (1, D)
        if self.distribution == "uniform":
            # noise in [-scale, +scale]
            noise = (self._rng_.random((N, D)) * 2.0 - 1.0) * scales
        elif self.distribution == "normal":
            # std = scale
            noise = self._rng_.standard_normal((N, D)) * scales
        else:
            raise ValueError("distribution must be 'uniform' or 'normal'.")

        X += noise
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        # Not invertible; return input unchanged (consistent with non-invertible noise transforms).
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES, ensure_all_finite=True)
        return X


# Return-based Scaling: Yet Another Normalisation Trick for Deep RL
# https://arxiv.org/abs/2105.05347
class ReturnScaledRewardTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_std: float | np.ndarray,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        eps: float = float(np.finfo(np.float32).eps),
    ) -> None:
        self.return_std = return_std
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        self.eps = eps

    def fit(self, X: np.ndarray, y: Any | None = None) -> Self:
        # No fitting necessary for tanh/arctanh
        self.is_fitted_ = True  # convention: trailing underscore
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        X = X / (self.return_std + self.eps)
        X = X * self.reward_scale + self.reward_bias
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        X = (X - self.reward_bias) / self.reward_scale
        X = X * (self.return_std + self.eps)
        return X
    
    def fit_transform(self, X: np.ndarray, y: Any | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


