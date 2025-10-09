import logging

from rlbidder import agents, data, envs, evaluation, models, viz
from rlbidder.constants import (
    DEFAULT_NUM_EVAL_SEEDS,
    DEFAULT_SEED,
    NUM_TICKS,
    STATE_DIM,
)
from rlbidder.utils import CustomValidationCallback, generate_seeds, log_distribution, regression_report

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # submodules
    "agents",
    "data",
    "envs",
    "evaluation",
    "models",
    "viz",
    # constants
    "DEFAULT_SEED",
    "DEFAULT_NUM_EVAL_SEEDS",
    "NUM_TICKS",
    "STATE_DIM",
    # utilities
    "generate_seeds",
    "regression_report",
    "log_distribution",
    "CustomValidationCallback",
]


# Silence library logs by default; apps/scripts can configure handlers as needed.
logging.getLogger(__name__).addHandler(logging.NullHandler())
