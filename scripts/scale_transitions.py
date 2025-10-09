import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import draccus
import numpy as np
import polars as pl
from rlbidder.constants import (
    CAMPAIGN_KEYS,
    DEFAULT_SEED,
    REWARD_DENSE_COL,
    REWARD_SPARSE_COL,
    TIME_STEP_COL,
    TRAIN_FILE_PREFIX,
)
from rlbidder.data.io import display_file_summary
from rlbidder.data.preprocess import fit_and_scale_offline_rl_transitions
from rlbidder.data.scalers import (
    AffineTransformer,
    ClipTransformer,
    FeatureWiseJitter,
    ReturnScaledRewardTransformer,
    TanhTransformer,
    WinsorizerTransformer,
)
from rlbidder.utils import configure_logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


PolicyType = Literal["tanh_normal", "biased_softplus"]


@dataclass
class Config:
    data_dir: Path = Path(__file__).parent.parent / "data"
    output_dir: str = "scaled_transitions"
    policy_type: PolicyType = "biased_softplus"


@draccus.wrap()
def main(cfg: Config = Config()) -> None:
    logger.info("RL transition data scaling process started.")
    
    data_dir = cfg.data_dir
    save_dir = data_dir / cfg.output_dir
    if not save_dir.exists():
        logger.info("Creating directory: %s", save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info("Directory exists: %s", save_dir)

    processed_dir = data_dir / "processed"
    logger.info("Loading training data from: %s (prefix: '%s')", processed_dir, TRAIN_FILE_PREFIX)
    display_file_summary(processed_dir, f"{TRAIN_FILE_PREFIX}*.parquet", max_display=10)

    lf_train = pl.scan_parquet(processed_dir / f"{TRAIN_FILE_PREFIX}*.parquet")
    n_rows = int(lf_train.select(pl.len()).collect().item())
    logger.info("Training rows to process: %d", n_rows)

    logger.info("Setting up state scaler.")
    state_scaler = Pipeline(
        steps=[
            ('feature_jitter', FeatureWiseJitter(  # tiny jitter to smooth quantile estimation
                scale_factor=0.1,
                distribution='uniform',
                random_state=DEFAULT_SEED,
                copy=True,
            )),
            ('quantile', QuantileTransformer(
                output_distribution='normal', 
                n_quantiles=100_000,  # HACK: critical to be sufficiently large otherwise performance degrades
                subsample=None,  # Use all data points
                random_state=DEFAULT_SEED,
            )),
        ],
        verbose=True,
    )
    
    logger.info("Setting up action scaler for policy type '%s'.", cfg.policy_type)

    if cfg.policy_type == "tanh_normal":
        action_scaler = Pipeline(
            steps=[
                ("clip", ClipTransformer(min_value=0.1, max_value=None)),
                ("winsor", WinsorizerTransformer(quantile_range=(0.005, 0.995))),
                (
                    "feature_jitter",
                    FeatureWiseJitter(
                        scale_factor=0.1,
                        distribution="uniform",
                        random_state=DEFAULT_SEED,
                        copy=True,
                    ),
                ),  # feature jitter to smooth quantile estimation
                (
                    "quantile",
                    QuantileTransformer(
                        output_distribution="normal",
                        n_quantiles=50_000,
                        subsample=None,  # use all samples for quantile estimation
                        random_state=DEFAULT_SEED,
                    ),
                ),
                ("affine", AffineTransformer(scale=0.7, bias=0.0)),  # sharper gradient
                ("tanh", TanhTransformer(eps=1e-6)),  # match tanh-normal actor distribution
            ],
            verbose=True,
        )
    elif cfg.policy_type == "biased_softplus":
        # NOTE: do not use SymlogTransformer in this pipeline (hurts performance)
        action_scaler = Pipeline(
            steps=[
                ("clip", ClipTransformer(min_value=0.1, max_value=None)),
                ("winsor", WinsorizerTransformer(quantile_range=(0.001, 0.999))),
                (
                    "robust",
                    RobustScaler(
                        with_centering=False,  # do NOT center the data to ensure non-negativity
                        with_scaling=True,
                        quantile_range=(25, 75),
                        unit_variance=False,
                    ),
                ),
            ],
            verbose=True,
        )
    else:
        raise ValueError(
            "Invalid policy_type '%s'. Expected one of: 'tanh_normal', 'biased_softplus'"
            % cfg.policy_type
        )

    logger.info("Setting up reward scalers.")
    lf_return_stds= (
        lf_train
        .group_by(["deliveryPeriodIndex", "advertiserNumber"])
        .agg(
            pl.col(TIME_STEP_COL).n_unique().alias("n_samples"),
            pl.col(REWARD_SPARSE_COL).sum().alias("return_sparse"),
            pl.col(REWARD_DENSE_COL).sum().alias("return_dense"),
        )
        .select(
            pl.col("return_sparse").std().alias("return_sparse_std"),
            pl.col("return_dense").std().alias("return_dense_std"),
        )
        .collect()
    )
    return_dense_std = lf_return_stds.select("return_dense_std").item()
    logger.info("return_dense_std: %s", return_dense_std)
    reward_dense_scaler = ReturnScaledRewardTransformer(
        return_std=return_dense_std,
        reward_scale=1.0,
        reward_bias=0.0,
        eps=np.finfo(np.float32).eps,
    )
    return_sparse_std = lf_return_stds.select("return_sparse_std").item()
    logger.info("return_sparse_std: %s", return_sparse_std)
    reward_sparse_scaler = ReturnScaledRewardTransformer(
        return_std=return_sparse_std,
        reward_scale=1.0,
        reward_bias=0.0,
        eps=np.finfo(np.float32).eps,
    )

    logger.info("Fitting and scaling offline RL transitions.")
    fit_and_scale_offline_rl_transitions(
        lf=lf_train,
        campaign_keys=CAMPAIGN_KEYS,
        time_step_col=TIME_STEP_COL,
        state_scaler=state_scaler,
        action_scaler=action_scaler,
        reward_dense_scaler=reward_dense_scaler,
        reward_sparse_scaler=reward_sparse_scaler,
        save_dir=save_dir,
    )

    display_file_summary(save_dir, "*", max_display=5)

    logger.info("RL transition data scaling process finished.")


if __name__ == "__main__":
    configure_logging(level=logging.DEBUG)
    main()
