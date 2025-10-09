"""General utilities shared across rlbidder."""

import hashlib
import logging
import sys
from typing import Any, Iterable, Optional

try:
    from rich.logging import RichHandler
except Exception:  # pragma: no cover - optional dependency
    RichHandler = None

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import TQDMProgressBar
from sklearn.metrics import max_error, mean_absolute_percentage_error, r2_score

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def is_interactive_stdout() -> bool:
    """Return ``True`` if stdout appears to be an interactive TTY."""

    try:
        return sys.stdout.isatty()
    except Exception:  # pragma: no cover - defensive
        return False


def get_progress_bar_callback(*, refresh_rate: int = 100, **kwargs: Any) -> TQDMProgressBar:
    """Return a Lightning progress bar suited for the current environment."""

    if is_interactive_stdout():
        try:  # pragma: no cover - optional dependency
            from lightning.pytorch.callbacks import RichProgressBar
        except ImportError:  # pragma: no cover - Rich not installed
            pass
        else:
            return RichProgressBar(refresh_rate=refresh_rate, **kwargs)

    return TQDMProgressBar(refresh_rate=refresh_rate, **kwargs)


def configure_logging(
    level: int = logging.INFO,
    *,
    force: bool = True,
    extra_handlers: Optional[Iterable[logging.Handler]] = None,
    rich_tracebacks: bool = True,
    show_time: bool = True,
    show_path: bool = True,
) -> None:
    """Configure global logging with Rich when available.

    Args:
        level: Logging level passed to :func:`logging.basicConfig`.
        force: Force reconfiguration even if the root logger already has handlers.
        extra_handlers: Optional additional handlers to append.
        rich_tracebacks: Enable Rich tracebacks when Rich is present.
        show_time: Show timestamps in Rich handler output.
        show_path: Show path information in Rich handler output.
    """

    handlers: list[logging.Handler] = []
    if RichHandler is not None:
        handlers.append(
            RichHandler(
                rich_tracebacks=rich_tracebacks,
                show_time=show_time,
                show_path=show_path,
            )
        )

    if extra_handlers:
        handlers.extend(extra_handlers)

    logging.basicConfig(
        level=level,
        format="%(message)s" if handlers else "%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers or None,
        force=force,
    )


def generate_seeds(
    base_seed: int, 
    num_seeds: int, 
    max_value: int = 2**32 - 1,
) -> list[int]:
    """Derive deterministic child seeds from a base seed.

    The function hashes ``"{base_seed}-{i}"`` using SHA-256 for each index ``i``
    and folds the digest into the ``[0, max_value]`` range. This allows us to
    deterministically expand a single seed into a reproducible sequence while
    still being compatible with ``numpy.random.default_rng`` and Lightning's
    ``L.seed_everything``.

    Args:
        base_seed: Starting seed (normally the value passed to
            ``L.seed_everything``).
        num_seeds: Number of derived seeds to generate.
        max_value: Upper bound for generated seeds (defaults to ``2**32 - 1`` to
            match NumPy's uint32 range).

    Returns:
        List of ``num_seeds`` integers suitable for NumPy/PyTorch RNGs.
    """

    seeds: list[int] = []
    for i in range(num_seeds):
        key = f"{base_seed}-{i}".encode("utf-8")
        digest = hashlib.sha256(key).digest()
        # Use first 8 bytes for a 64-bit integer seed, then mod to fit numpy's uint32 range
        seed = int.from_bytes(digest[:8], "big") % (max_value + 1)
        seeds.append(seed)
    return seeds


def regression_report(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    return {
        f"{prefix}r2": r2_score(y_true=y_true, y_pred=y_pred),
        f"{prefix}mape": mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred),
        f"{prefix}max_err": max_error(y_true=y_true, y_pred=y_pred),
    }


def log_distribution(
    logger: Any,
    data: torch.Tensor | np.ndarray,
    name: str,
    step: int,
    context: dict[str, Any] | str,
    bins: int = 64,
    percentile_range: tuple[float, float] = (0.5, 99.5),
) -> None:
    """Log a histogram to Aim when available, otherwise emit a warning once.

    Args:
        logger: Lightning logger instance. Supported values are instances of
            ``AimLogger`` or objects exposing ``experiment.track`` with Aim-like
            semantics. Passing ``None`` disables logging.
        data: Array-like payload (PyTorch tensors are converted to NumPy).
        name: Name of the distribution metric (e.g., ``"critic/q_values"``).
        step: Global step at which to record the metric.
        context: Aim context dictionary or subset label.
        bins: Number of histogram bins (default 64).
        percentile_range: Percentiles used to bound the histogram range and
            ignore outliers.
    """
    # No-op if logger is disabled or missing
    if not logger:
        return

    # Attempt to import aim lazily to avoid hard dependency
    aim_module = None
    try:
        import aim as aim_module  # type: ignore
    except Exception:
        aim_module = None

    # Only proceed for Aim logger. Prefer a type check; fallback to a duck-typed attribute check.
    is_aim_logger = False
    try:
        from aim.pytorch_lightning import AimLogger  # local import to avoid hard dependency
        is_aim_logger = isinstance(logger, AimLogger)
    except Exception:
        is_aim_logger = hasattr(logger, "experiment") and hasattr(getattr(logger, "experiment", None), "track")

    if not is_aim_logger or aim_module is None:
        return

    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    # Use percentile-bounded range to ignore extreme outliers
    range_bounds = np.nanpercentile(data, percentile_range)

    # Accept dict contexts passed by callers; otherwise wrap as subset
    ctx = context if isinstance(context, dict) else {"subset": context}

    # Create histogram with bounded range
    histogram = np.histogram(data, bins=bins, range=range_bounds)

    # Log histogram using Aim
    logger.experiment.track(
        aim_module.Distribution.from_histogram(*histogram),
        name=name,
        step=step,
        context=ctx,
    )


class CustomValidationCallback(L.Callback):
    """Custom callback to run validation at specific initial steps."""
    
    def __init__(self, initial_validation_steps: list[int], val_check_interval: int) -> None:
        super().__init__()
        self.initial_validation_steps = sorted(initial_validation_steps)
        self.val_check_interval = val_check_interval
        self.completed_initial_steps = set()
        self.last_interval_change = -1
        
    def on_train_batch_start(self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int) -> None:
        current_step = trainer.global_step
        # Only change val_check_interval if we haven't changed it recently
        if current_step != self.last_interval_change:
            # Check if we're about to hit an initial validation step
            if current_step + 1 in self.initial_validation_steps:
                trainer.val_check_batch = 1
            else:
                trainer.val_check_batch = self.val_check_interval
            self.last_interval_change = current_step
