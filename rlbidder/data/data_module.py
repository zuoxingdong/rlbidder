from pathlib import Path
from typing import Any

import logging
import joblib
import lightning as L
import polars as pl
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    TensorDataset,
    WeightedRandomSampler,
)

from rlbidder.data.traj_dataset import TrajDataset
from rlbidder.data.replay_buffer import OfflineReplayBuffer
from rlbidder.evaluation import OnlineCampaignEvaluator
from rlbidder.utils import generate_seeds
from rlbidder.constants import DEFAULT_SEED, DEFAULT_NUM_EVAL_SEEDS, TRAIN_FILE_PREFIX

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class OfflineDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        scaler_dir: str | Path,
        batch_size: int = 256,
        reward_type: str = "dense",
        val_periods: list[int] = [7, 8],
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.scaler_dir = Path(scaler_dir)
        self.batch_size = batch_size
        self.reward_type = reward_type  # "dense" or "sparse"
        self.val_periods = val_periods

        # placeholders; will be set in prepare_data()
        self.train_periods = None
        self.val_period = None

        # Placeholder for scalers
        # These will be set during prepare_data
        self.state_scaler = None
        self.action_scaler = None
        self.reward_scaler = None
        
        self.load_scalers(self.scaler_dir)

    def prepare_data(self) -> None:
        # use things that must happen only once ever (e.g., downloading, tokenizers).
        # called on 1 GPU/process only
        self.train_file_prefix = TRAIN_FILE_PREFIX

        train_files = sorted(self.data_dir.glob(f"{self.train_file_prefix}*.parquet"))
        logger.info("Found %s train files:", len(train_files))
        for f in train_files:
            logger.debug("  - %s", f.name)

        # use configured validation periods
        self.val_period = self.val_periods
        logger.info("Validation period: %s", self.val_period)

        self.lf_train_transitions = pl.scan_parquet(self.scaler_dir / "scaled_transitions.parquet")
        
        # Compute episode statistics for training data
        logger.info("Computing episode statistics for training data...")
        reward_col = f"reward_{self.reward_type}"
        episode_stats = (
            pl.scan_parquet(self.data_dir / f"{self.train_file_prefix}*.parquet")
            .select(["deliveryPeriodIndex", "advertiserNumber", reward_col])
            .group_by(["deliveryPeriodIndex", "advertiserNumber"])
            .agg(
                episode_return=pl.col(reward_col).sum(),
                episode_length=pl.col(reward_col).count(),
            )
            .describe()
        )
        self.min_epi_ret = episode_stats.filter(pl.col("statistic") == "min")["episode_return"][0]
        self.max_epi_ret = episode_stats.filter(pl.col("statistic") == "max")["episode_return"][0]
        self.max_epi_len = episode_stats.filter(pl.col("statistic") == "max")["episode_length"][0]
        logger.info("Reward Type: %s", reward_col)
        logger.info("Min/Max episode return: %s / %s", self.min_epi_ret, self.max_epi_ret)
        logger.info("Max episode length: %s", self.max_epi_len)

        self.evaluator = OnlineCampaignEvaluator(
            data_dir=self.data_dir,
            min_remaining_budget=0.1,
            delivery_period_indices=self.val_periods,
            seeds=generate_seeds(DEFAULT_SEED, DEFAULT_NUM_EVAL_SEEDS),
            verbose=True,
        )
        logger.info("OnlineCampaignEvaluator initialized.")

    def _prepare_train_replay_buffer(self) -> OfflineReplayBuffer:
        df_train_transitions = self.lf_train_transitions.collect()
        # Collect and normalize validation data, then create and fill the replay buffer
        normalized_state = df_train_transitions["state"].to_numpy()
        normalized_action = df_train_transitions["action"].to_numpy()
        # Use the appropriate scaled reward column based on reward_type
        reward_col_scaled = f"reward_{self.reward_type}_scaled"
        normalized_reward = df_train_transitions[reward_col_scaled].to_numpy()
        normalized_next_state = df_train_transitions["next_state"].to_numpy()
        done = df_train_transitions["done"].to_numpy()
        cpa_compliance_ratio = df_train_transitions["cpa_compliance_ratio"].to_numpy()
        buffer = OfflineReplayBuffer(
            buffer_size=len(normalized_state),
            obs_shape=normalized_state.shape[1:],
            action_shape=(1,),
        )
        buffer.push(
            state=normalized_state,
            action=normalized_action[:, None],
            reward=normalized_reward,
            next_state=normalized_next_state,
            done=done,
            cpa_compliance_ratio=cpa_compliance_ratio,
        )
        logger.info("Training replay buffer created: %s samples using %s.", len(normalized_state), reward_col_scaled)
        
        return buffer

    def setup(self, stage: str) -> None:
        # loading files, fitting and applying scalers, constructing and splitting Datasets
        # called on every GPU/process in DDP
        if stage == "fit":  # load both training and validation data
            self.train_replay_buffer = self._prepare_train_replay_buffer()
        elif stage == "validate":  # load validation data
            pass
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        batch_sampler = BatchSampler(
            RandomSampler(self.train_replay_buffer),
            batch_size=self.batch_size,
            drop_last=False,
        )
        return DataLoader(
            self.train_replay_buffer,
            batch_sampler=batch_sampler,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=32,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            TensorDataset(torch.tensor(self.val_period)), 
            batch_size=len(self.val_period), 
            shuffle=False, 
            num_workers=1, 
            persistent_workers=True,
        )

    @property
    def scalers(self) -> dict[str, Any]:
        # Return a dictionary of all scalers
        return {
            "state_scaler": self.state_scaler,
            "action_scaler": self.action_scaler,
            "reward_scaler": self.reward_scaler,
        }

    def dump_scalers(self, scaler_dir: str | Path) -> None:
        # Save the scalers to disk
        Path(scaler_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.state_scaler, scaler_dir / "state_scaler.joblib")
        joblib.dump(self.action_scaler, scaler_dir / "action_scaler.joblib")
        # Save with the appropriate reward scaler filename
        reward_scaler_file = f"reward_{self.reward_type}_scaler.joblib"
        joblib.dump(self.reward_scaler, scaler_dir / reward_scaler_file)

    def load_scalers(self, scaler_dir: str | Path) -> None:
        # Load the scalers from disk
        scaler_dir = Path(scaler_dir)
        self.state_scaler = joblib.load(scaler_dir / "state_scaler.joblib")
        self.action_scaler = joblib.load(scaler_dir / "action_scaler.joblib")
        # Load the appropriate reward scaler based on reward_type
        reward_scaler_file = f"reward_{self.reward_type}_scaler.joblib"
        self.reward_scaler = joblib.load(scaler_dir / reward_scaler_file)
        logger.info("Scalers loaded successfully. Using %s for reward scaling.", reward_scaler_file)


class OfflineDTDataModule(L.LightningDataModule):
    def __init__(
        self,
        dt_traj_path: str | Path,
        data_dir: str | Path,
        scaler_dir: str | Path,
        seq_len: int,
        seed: int,
        batch_size: int = 256,
        reward_type: str = "dense",
        val_periods: list[int] = [7, 8],
    ) -> None:
        super().__init__()
        self.dt_traj_path = dt_traj_path
        self.data_dir = Path(data_dir)
        self.scaler_dir = Path(scaler_dir)
        self.seq_len = seq_len
        self.seed = seed
        self.batch_size = batch_size
        self.reward_type = reward_type  # "dense" or "sparse"
        self.val_periods = val_periods
        
        # Placeholder for scalers
        # These will be set during prepare_data
        self.state_scaler = None
        self.action_scaler = None
        self.reward_scaler = None
        
        self.load_scalers(self.scaler_dir)

    def prepare_data(self) -> None:
        # use things that must happen only once ever (e.g., downloading, tokenizers).
        # called on 1 GPU/process only
        
        # use configured validation periods
        self.val_period = self.val_periods
        logger.info("Validation period: %s", self.val_period)

        self.evaluator = OnlineCampaignEvaluator(
            data_dir=self.data_dir,
            min_remaining_budget=0.1,
            delivery_period_indices=self.val_periods,
            seeds=generate_seeds(DEFAULT_SEED, DEFAULT_NUM_EVAL_SEEDS),
            verbose=True,
        )
        logger.info("OnlineCampaignEvaluator initialized.")

    def setup(self, stage: str) -> None:

        if stage == "fit":
            self.train_dataset = TrajDataset(
                traj_path=self.dt_traj_path,
                seq_len=self.seq_len,
                normalize_states=False,
                seed=self.seed,
                sample_offset=5,  # TODO: configurable
            )
            self.val_dataset = TensorDataset(torch.tensor(self.val_period))
        elif stage == "validate":
            # # Optionally implement validation dataset
            # self.val_dataset = None
            pass
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        sampler = WeightedRandomSampler(
            weights=self.train_dataset.sample_prob,
            num_samples=len(self.train_dataset),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=32,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=len(self.val_period), 
            shuffle=False, 
            num_workers=1, 
            persistent_workers=True,
        )

    @property
    def scalers(self) -> dict[str, Any]:
        # Return a dictionary of all scalers
        return {
            "state_scaler": self.state_scaler,
            "action_scaler": self.action_scaler,
            "reward_scaler": self.reward_scaler,
        }

    def dump_scalers(self, scaler_dir: str | Path) -> None:
        # Save the scalers to disk
        Path(scaler_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.state_scaler, scaler_dir / "state_scaler.joblib")
        joblib.dump(self.action_scaler, scaler_dir / "action_scaler.joblib")
        # Save with the appropriate reward scaler filename
        reward_scaler_file = f"reward_{self.reward_type}_scaler.joblib"
        joblib.dump(self.reward_scaler, scaler_dir / reward_scaler_file)

    def load_scalers(self, scaler_dir: str | Path) -> None:
        # Load the scalers from disk
        scaler_dir = Path(scaler_dir)
        self.state_scaler = joblib.load(scaler_dir / "state_scaler.joblib")
        self.action_scaler = joblib.load(scaler_dir / "action_scaler.joblib")
        # Load the appropriate reward scaler based on reward_type
        reward_scaler_file = f"reward_{self.reward_type}_scaler.joblib"
        self.reward_scaler = joblib.load(scaler_dir / reward_scaler_file)
        print(f"Scalers loaded successfully. Using {reward_scaler_file} for reward scaling.")
