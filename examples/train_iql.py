from pathlib import Path
from dataclasses import dataclass, field
import logging

import draccus
import torch
import lightning as L
from aim.pytorch_lightning import AimLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
try:
    from lightning.pytorch.callbacks import RichModelSummary
except ImportError:  # pragma: no cover - optional dependency missing
    from lightning.pytorch.callbacks import ModelSummary as RichModelSummary

torch.set_float32_matmul_precision("high")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

from rlbidder.agents.iql import IQLModel  # noqa: E402
from rlbidder.constants import STATE_DIM, DEFAULT_SEED
from rlbidder.data.data_module import OfflineDataModule  # noqa: E402
from rlbidder.utils import CustomValidationCallback, get_progress_bar_callback, configure_logging  # noqa: E402


@dataclass
class DataConfig:
    data_dir: str = str(PROJECT_ROOT / "data" / "processed")
    scaler_dir: str = str(PROJECT_ROOT / "data" / "scaled_transitions")
    batch_size: int = 512
    reward_type: str = "dense"  # "dense" or "sparse"


@dataclass
class ModelConfig:
    state_dim: int = STATE_DIM
    action_dim: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    cpa_penalty_beta: float = 2.0
    actor_hidden_sizes: list[int] = field(default_factory=lambda: [512, 512, 512])
    critic_hidden_sizes: list[int] = field(default_factory=lambda: [512, 512, 512])
    value_hidden_sizes: list[int] = field(default_factory=lambda: [512, 512, 512])
    num_q_models: int = 5
    # Critic HLG (Q-network)
    hlg_q_num_atoms: int = 101
    hlg_q_vmin: float = 0.0
    hlg_q_vmax: float = 20.0
    hlg_q_sigma_to_bin_width_ratio: float = 0.75
    hlg_q_prior_scale: float = 40.9
    # Optimization
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_value: float = 3e-4
    use_actor_scheduler: bool = True
    bc_alpha: float = 0.01  # BC coefficient in DDPG+BC
    alpha_init_value: float = 0.1  # Initial value for learnable alpha
    target_entropy: float = -0.5  # Target entropy for SAC-style entropy regularization
    enable_q_normalization: bool = True  # Enable Q value normalization
    mean_l2_coeff: float = 0.0001  # Coefficient on E[mean^2] regularizer
    std_l2_coeff: float = 0.001   # Coefficient on E[std^2] regularizer
    actor_grad_clip_norm: float = 1.0
    critic_grad_clip_norm: float = 1.0
    value_grad_clip_norm: float = 1.0
    expectile_tau: float = 0.8
    val_metric: str = "mean_score"


@dataclass
class CheckpointConfig:
    monitor: str | None = None  # if None, use the monitor key from ModelConfig
    mode: str = "max"
    save_top_k: int = 1
    filename: str = "best"
    enable_version_counter: bool = False
    every_n_train_steps: int | None = None  # Don't save based on training steps but when metric improves


@dataclass
class EarlyStoppingConfig:
    monitor: str | None = None  # if None, use the monitor key from ModelConfig
    mode: str = "max"
    patience: int = 200
    strict: bool = False


@dataclass
class TrainConfig:
    seed: int = DEFAULT_SEED
    max_steps: int = 500_000
    val_check_interval: int = 10_000  # run val every X steps
    initial_validation_steps: list[int] = field(default_factory=lambda: [100, 1_000, 5_000])  # Custom initial validation steps
    model_dir: str = str(CURRENT_DIR / "checkpoints" / "iql")
    accelerator: str = "gpu"
    devices: list[int] | str | int = 1
    log_every_n_steps: int = 1_000
    enable_aim_logger: bool = True  # Enable/disable Aim logger


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
    """Main training routine for IQL."""
    L.seed_everything(cfg.train_cfg.seed, workers=True)

    datamodule = OfflineDataModule(
        data_dir=Path(cfg.data.data_dir),
        scaler_dir=Path(cfg.data.scaler_dir),
        batch_size=cfg.data.batch_size,
        reward_type=cfg.data.reward_type,
    )

    model = IQLModel(
        state_dim=cfg.model_cfg.state_dim,
        action_dim=cfg.model_cfg.action_dim,
        gamma=cfg.model_cfg.gamma,
        tau=cfg.model_cfg.tau,
        reward_scale=cfg.model_cfg.reward_scale,
        reward_bias=cfg.model_cfg.reward_bias,
        cpa_penalty_beta=cfg.model_cfg.cpa_penalty_beta,
        actor_hidden_sizes=cfg.model_cfg.actor_hidden_sizes,
        critic_hidden_sizes=cfg.model_cfg.critic_hidden_sizes,
        value_hidden_sizes=cfg.model_cfg.value_hidden_sizes,
        num_q_models=cfg.model_cfg.num_q_models,
        hlg_q_num_atoms=cfg.model_cfg.hlg_q_num_atoms,
        hlg_q_vmin=cfg.model_cfg.hlg_q_vmin,
        hlg_q_vmax=cfg.model_cfg.hlg_q_vmax,
        hlg_q_sigma_to_bin_width_ratio=cfg.model_cfg.hlg_q_sigma_to_bin_width_ratio,
        hlg_q_prior_scale=cfg.model_cfg.hlg_q_prior_scale,
        lr_actor=cfg.model_cfg.lr_actor,
        lr_critic=cfg.model_cfg.lr_critic,
        lr_value=cfg.model_cfg.lr_value,
        use_actor_scheduler=cfg.model_cfg.use_actor_scheduler,
        bc_alpha=cfg.model_cfg.bc_alpha,
        alpha_init_value=cfg.model_cfg.alpha_init_value,
        target_entropy=cfg.model_cfg.target_entropy,
        enable_q_normalization=cfg.model_cfg.enable_q_normalization,
        mean_l2_coeff=cfg.model_cfg.mean_l2_coeff,
        std_l2_coeff=cfg.model_cfg.std_l2_coeff,
        expectile_tau=cfg.model_cfg.expectile_tau,
        actor_grad_clip_norm=cfg.model_cfg.actor_grad_clip_norm,
        critic_grad_clip_norm=cfg.model_cfg.critic_grad_clip_norm,
        value_grad_clip_norm=cfg.model_cfg.value_grad_clip_norm,
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
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor=f"val/{cfg.model_cfg.val_metric}" if cfg.early_stopping_cfg.monitor is None else cfg.early_stopping_cfg.monitor,
        mode=cfg.early_stopping_cfg.mode,
        patience=cfg.early_stopping_cfg.patience,
        strict=cfg.early_stopping_cfg.strict,
    )

    # Conditionally create Aim logger
    logger = False
    if cfg.train_cfg.enable_aim_logger:
        logger = AimLogger(
            experiment="iql",
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
