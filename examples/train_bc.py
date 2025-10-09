from pathlib import Path
from dataclasses import dataclass, field
import logging

import draccus
import lightning as L
from aim.pytorch_lightning import AimLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
try:
    from lightning.pytorch.callbacks import RichModelSummary
except ImportError:  # pragma: no cover
    from lightning.pytorch.callbacks import ModelSummary as RichModelSummary

import torch

torch.set_float32_matmul_precision("high")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

from rlbidder.agents.bc import BCModel  # noqa: E402
from rlbidder.constants import STATE_DIM, DEFAULT_SEED
from rlbidder.data.data_module import OfflineDataModule  # noqa: E402
from rlbidder.utils import CustomValidationCallback, get_progress_bar_callback, configure_logging  # noqa: E402


@dataclass
class DataConfig:
    data_dir: str = str(PROJECT_ROOT / "data" / "processed")
    scaler_dir: str = str(PROJECT_ROOT / "data" / "scaled_transitions")
    batch_size: int = 512


@dataclass
class ModelConfig:
    state_dim: int = STATE_DIM
    lr: float = 1e-4
    hidden_sizes: list[int] = field(default_factory=lambda: [512, 512, 512])
    val_metric: str = "mean_score"


@dataclass
class CheckpointConfig:
    monitor: str | None = None  # if None, use the monitor key from ModelConfig
    mode: str = "max"
    save_top_k: int = 1
    filename: str = "best"
    enable_version_counter: bool = False
    every_n_train_steps: int | None = None  # Don't save based on training steps but when metric improves
    every_n_epochs: int = 10


@dataclass
class EarlyStoppingConfig:
    monitor: str | None = None  # if None, use the monitor key from ModelConfig
    mode: str = "max"
    patience: int = 50
    strict: bool = False


@dataclass
class TrainConfig:
    seed: int = DEFAULT_SEED
    max_steps: int = 500_000
    val_check_interval: int = 10_000  # run val every X steps
    initial_validation_steps: list[int] = field(default_factory=lambda: [100, 1_000, 5_000])  # Custom initial validation steps
    model_dir: str = str(CURRENT_DIR / "checkpoints" / "bc")
    accelerator: str = "gpu"
    devices: list[int] | str | int = 1
    log_every_n_steps: int = 1_000
    enable_aim_logger: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    train_cfg: TrainConfig = field(default_factory=TrainConfig)
    checkpoint_cfg: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping_cfg: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@draccus.wrap()
def main(
    cfg: Config = Config(),
) -> None:
    """Main training routine for BC."""
    L.seed_everything(cfg.train_cfg.seed, workers=True)

    datamodule = OfflineDataModule(
        data_dir=Path(cfg.data.data_dir),
        scaler_dir=Path(cfg.data.scaler_dir),
        batch_size=cfg.data.batch_size,
    )

    model = BCModel(
        state_dim=cfg.model_cfg.state_dim,
        lr=cfg.model_cfg.lr,
        hidden_sizes=cfg.model_cfg.hidden_sizes,
        val_metric=cfg.model_cfg.val_metric,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train_cfg.model_dir,
        filename=cfg.checkpoint_cfg.filename,
        monitor=f"val/{cfg.model_cfg.val_metric}" if cfg.checkpoint_cfg.monitor is None else cfg.checkpoint_cfg.monitor,
        mode=cfg.checkpoint_cfg.mode,
        auto_insert_metric_name=False,
        save_top_k=cfg.checkpoint_cfg.save_top_k,
        enable_version_counter=cfg.checkpoint_cfg.enable_version_counter,
        every_n_train_steps=cfg.checkpoint_cfg.every_n_train_steps,
        save_last=True,
        verbose=True,
    )
    checkpoint_epoch_cb = ModelCheckpoint(
        dirpath=cfg.train_cfg.model_dir,
        filename="{epoch}",
        every_n_epochs=cfg.checkpoint_cfg.every_n_epochs,
        save_on_train_epoch_end=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor=f"val/{cfg.model_cfg.val_metric}" if cfg.early_stopping_cfg.monitor is None else cfg.early_stopping_cfg.monitor,
        mode=cfg.early_stopping_cfg.mode,
        patience=cfg.early_stopping_cfg.patience,
        strict=cfg.early_stopping_cfg.strict,
    )

    # Conditionally create Aim logger
    logger = None
    if cfg.train_cfg.enable_aim_logger:
        logger = AimLogger(
            experiment="bc",
            context_prefixes=dict(subset={"train": "train/", "val": "val/", "test": "test/"}),
        )

    # Custom validation callback for initial steps
    custom_validation_callback = CustomValidationCallback(
        initial_validation_steps=cfg.train_cfg.initial_validation_steps,
        val_check_interval=cfg.train_cfg.val_check_interval,
    )

    trainer = L.Trainer(
        logger=logger,
        accelerator=cfg.train_cfg.accelerator,
        devices=cfg.train_cfg.devices,
        max_steps=cfg.train_cfg.max_steps,
        val_check_interval=cfg.train_cfg.val_check_interval,
        check_val_every_n_epoch=None,  # NOTE: triggered solely by the initial steps and interval
        num_sanity_val_steps=0,  # skip sanity check
        callbacks=[
            custom_validation_callback,
            checkpoint_callback,
            checkpoint_epoch_cb,
            early_stopping_callback,
            get_progress_bar_callback(refresh_rate=100),
            RichModelSummary(max_depth=2),
        ],
        log_every_n_steps=cfg.train_cfg.log_every_n_steps,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    main()
