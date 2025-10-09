from pathlib import Path
from dataclasses import dataclass, field
import logging

import draccus
import lightning as L
from aim.pytorch_lightning import AimLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
try:
    from lightning.pytorch.callbacks import RichModelSummary
except ImportError:  # pragma: no cover - fallback when Rich is unavailable
    from lightning.pytorch.callbacks import ModelSummary as RichModelSummary

import torch

torch.set_float32_matmul_precision("high")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

from rlbidder.constants import NUM_TICKS, STATE_DIM, DEFAULT_SEED
from rlbidder.agents.dt import DTModel  # noqa: E402
from rlbidder.data.data_module import OfflineDTDataModule  # noqa: E402
from rlbidder.utils import CustomValidationCallback, get_progress_bar_callback, configure_logging  # noqa: E402


@dataclass
class DataConfig:
    dt_traj_path: str = str(PROJECT_ROOT / "data" / "processed" / "trajectories.npz")
    data_dir: str = str(PROJECT_ROOT / "data" / "processed")
    scaler_dir: str = str(PROJECT_ROOT / "data" / "scaled_transitions")
    batch_size: int = 256
    seq_len: int = 20
    seed: int = DEFAULT_SEED


@dataclass
class ModelConfig:
    state_dim: int = STATE_DIM
    action_dim: int = 1
    seq_len: int = 20
    episode_len: int = NUM_TICKS
    embedding_dim: int = 512
    intermediate_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    lr: float = 1e-4
    weight_decay: float = 0.0
    use_lr_scheduler: bool = True
    lr_warmup_steps: int = 50_000
    lr_constant_steps: int = 300_000
    lr_min: float = 1e-8
    rtg_scale: float = 46  # Return to Go scale factor
    target_rtg: float = 2.0  # Target return-to-go for evaluation
    target_entropy: float | None = -0.5  # Target entropy for SAC-style entropy regularization
    alpha_init_value: float = 0.5
    actor_grad_clip_norm: float = 10.0
    mean_l2_coeff: float = 0.00001
    std_l2_coeff: float = 0.0001
    bc_alpha: float = 0.1
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
    model_dir: str = str(CURRENT_DIR / "checkpoints" / "dt")
    accelerator: str = "gpu"
    devices: int | str | list[int] = 1
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
    """Main training routine for Decision Transformer (DT)."""
    L.seed_everything(cfg.train_cfg.seed, workers=True)
    
    datamodule = OfflineDTDataModule(
        dt_traj_path=cfg.data.dt_traj_path,
        data_dir=Path(cfg.data.data_dir),
        scaler_dir=Path(cfg.data.scaler_dir),
        seq_len=cfg.data.seq_len,
        seed=cfg.data.seed,
        batch_size=cfg.data.batch_size,
        reward_type="dense",  # Use dense rewards for DT
    )

    model = DTModel(
        state_dim=cfg.model_cfg.state_dim,
        action_dim=cfg.model_cfg.action_dim,
        seq_len=cfg.model_cfg.seq_len,
        episode_len=cfg.model_cfg.episode_len,
        embedding_dim=cfg.model_cfg.embedding_dim,
        intermediate_size=cfg.model_cfg.intermediate_size,
        num_layers=cfg.model_cfg.num_layers,
        num_heads=cfg.model_cfg.num_heads,
        attention_dropout=cfg.model_cfg.attention_dropout,
        residual_dropout=cfg.model_cfg.residual_dropout,
        embedding_dropout=cfg.model_cfg.embedding_dropout,
        lr=cfg.model_cfg.lr,
        weight_decay=cfg.model_cfg.weight_decay,
        use_lr_scheduler=cfg.model_cfg.use_lr_scheduler,
        lr_warmup_steps=cfg.model_cfg.lr_warmup_steps,
        lr_constant_steps=cfg.model_cfg.lr_constant_steps,
        lr_min=cfg.model_cfg.lr_min,
        rtg_scale=cfg.model_cfg.rtg_scale,
        target_rtg=cfg.model_cfg.target_rtg,
        target_entropy=cfg.model_cfg.target_entropy,
        alpha_init_value=cfg.model_cfg.alpha_init_value,
        actor_grad_clip_norm=cfg.model_cfg.actor_grad_clip_norm,
        mean_l2_coeff=cfg.model_cfg.mean_l2_coeff,
        std_l2_coeff=cfg.model_cfg.std_l2_coeff,
        bc_alpha=cfg.model_cfg.bc_alpha,
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
    logger = False  # set False to disable logger
    if cfg.train_cfg.enable_aim_logger:
        logger = AimLogger(
            experiment="dt",
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
        check_val_every_n_epoch=None,  # validate based on the total training batches
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
