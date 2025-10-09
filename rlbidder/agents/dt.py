import math
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import polars as pl
import torch
import torch.optim as optim

from rlbidder.agents.base import BaseBiddingAgent
from rlbidder.agents.get_baseline_agent_configs import get_baseline_agent_configs
from rlbidder.constants import DEFAULT_SEED, NUM_TICKS, STATE_DIM
from rlbidder.evaluation import summarize_agent_scores
from rlbidder.models.distributions import BiasedSoftplusNormal
from rlbidder.models.networks import LearnableScalar, NormalHead, init_trunc_normal
from rlbidder.models.optim import (
    LinearWarmupConstantCosineAnnealingLR,
    clip_grad_norm_fast,
    fix_global_step_for_multi_optimizers,
    get_decay_and_no_decay_params,
)
from rlbidder.models.transformers import DecisionTransformer, DTInferenceBuffer
from rlbidder.models.utils import extract_state_features, masked_mean
from rlbidder.utils import log_distribution, regression_report


class DTModel(L.LightningModule):
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = 1,
        seq_len: int = 20,
        episode_len: int = NUM_TICKS,
        embedding_dim: int = 512,
        intermediate_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        rtg_scale: float = 46.0,  # scale for returns-to-go
        target_rtg: float = 2.0,  # target return-to-go for evaluation
        target_entropy: float | None = -0.5,  # Target entropy for SAC-style entropy regularization
        alpha_init_value: float = 0.5,
        use_lr_scheduler: bool = True,
        lr_warmup_steps: int = 50_000,  # Number of warmup steps
        lr_constant_steps: int = 300_000,  # Number of constant steps
        lr_min: float = 1e-8,  # Minimum learning rate
        actor_grad_clip_norm: float = 10.0,
        mean_l2_coeff: float = 0.00001,
        std_l2_coeff: float = 0.0001,
        bc_alpha: float = 0.1,
        val_metric: str = "mean_score",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = DecisionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            seq_len=seq_len,
            episode_len=episode_len,
            embedding_dim=embedding_dim,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            residual_dropout=residual_dropout,
            embedding_dropout=embedding_dropout,
        )
        self.action_head = NormalHead(
            in_features=embedding_dim,
            out_features=action_dim,
            bias=True,
            std_softplus_min=1e-2,
            std_softplus_bias=0.1,
            std_min=1e-2,
            std_max=1.0,
        )
        self.apply(lambda m: init_trunc_normal(m, std=0.02))

        # Learnable alpha for entropy regularization
        self.alpha = LearnableScalar(
            init_value=math.log(alpha_init_value),  # Initial value in log space
            log_space=True,
            min_log_value=math.log(1e-6),  # Minimum log alpha value
            max_log_value=math.log(1e9),   # Maximum log alpha value
        )

        # Placeholder for scalers
        # These will be set during training or loaded from checkpoint
        self.scalers = None

        # Enable manual optimization for custom update order
        self.automatic_optimization = False
        
    def configure_optimizers(self):
        # NOTE: Beta2=0.95, see https://github.com/facebookresearch/mae/issues/184#issuecomment-1861673795
        # we follow Qwen style
        # - https://arxiv.org/pdf/2308.12966v1#page=4.79
        # - https://arxiv.org/pdf/2309.16609#page=10.03
        # Build param groups via reusable helper
        decay_params, no_decay_params = get_decay_and_no_decay_params(self.model, self.action_head)

        opt_dt = optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=(0.9, 0.995),
            weight_decay=0.0,  # decay is set in param groups
            fused=True,
        )
        opt_alpha = optim.AdamW(
            self.alpha.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            fused=True,
        )
        if not self.hparams.use_lr_scheduler:
            return (
                {"optimizer": opt_dt},
                {"optimizer": opt_alpha},
            )
        
        # NOTE: we use WSD style (https://arxiv.org/pdf/2410.05192#page=5.35)
        sch_dt = LinearWarmupConstantCosineAnnealingLR(
            optimizer=opt_dt,
            warmup_epochs=self.hparams.lr_warmup_steps,
            max_epochs=self.trainer.max_steps,
            constant_epochs=self.hparams.lr_constant_steps,
            warmup_start_lr=self.hparams.lr_min,
            eta_min=self.hparams.lr_min,
            last_epoch=-1,
        )
        return (
            {"optimizer": opt_dt, "lr_scheduler": {"scheduler": sch_dt}},
            {"optimizer": opt_alpha},
        )
    
    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        opt_dt, opt_alpha = self.optimizers()
        sch_dt = self.lr_schedulers() if self.hparams.use_lr_scheduler else None
        fix_global_step_for_multi_optimizers(self.trainer, opt_alpha)

        states, actions, rewards, penalties, dones, rtgs, time_steps, mask = batch

        # set padding timesteps to 0
        time_steps = time_steps.masked_fill(~mask.bool(), 0)
        scaled_rtgs = rtgs / self.hparams.rtg_scale
        
        r_out, s_out, a_out = self.model(
            states=states,
            actions=actions,
            returns_to_go=scaled_rtgs[..., :-1],
            time_steps=time_steps,
        )
        mean, std = self.action_head(s_out)
        
        dist = BiasedSoftplusNormal(
            loc=mean,
            scale=std,
            # bias=1.0,
            safe_bound_eps=1e-6,
        )
        bc_logprob = dist.log_prob(actions).squeeze(-1)  # (batch_size, seq_len)
        # # NOTE: analytic LogNormal entropy depends on μ -> pushes μ upward (bigger actions)
        # entropy = dist.entropy().squeeze(-1)
        pi_actions, pi_logprobs = dist.rsample_and_log_prob()  # (batch_size, seq_len)
        pi_logprobs = pi_logprobs.squeeze(-1)
        # MC estimate of entropy
        entropy = -pi_logprobs
        alpha, log_alpha = self.alpha(return_log=True)

        actor_loss = -(self.hparams.bc_alpha * bc_logprob + alpha.detach() * entropy)
        actor_loss = masked_mean(mask, actor_loss)

        # Separate mean/std magnitude regularization
        mean_l2 = masked_mean(mask, (mean ** 2).squeeze(-1))
        std_l2 = masked_mean(mask, (std ** 2).squeeze(-1))
        mean_l2_loss = self.hparams.mean_l2_coeff * mean_l2
        std_l2_loss = self.hparams.std_l2_coeff * std_l2

        # Total actor loss
        actor_total_loss = actor_loss + mean_l2_loss + std_l2_loss

        # Entropy loss (update alpha)
        # TODO: optimization in log_space of alpha to adapt faster, use raw alpha has too small loss scale
        alpha_loss = alpha * (entropy - self.hparams.target_entropy).detach()
        # alpha_loss = log_alpha * (entropy - self.hparams.target_entropy).detach()
        alpha_loss = masked_mean(mask, alpha_loss)

        # HACK: avoid incorrect gradient accumulation with multiple optimizers
        for opt in self.optimizers():
            opt.zero_grad(set_to_none=True)
        # Update actor parameters
        # single backward (disjoint grad params) -> slightly more efficient and memory-friendly
        self.manual_backward(actor_total_loss + alpha_loss)
        actor_grad_norm = clip_grad_norm_fast(
            self.model.parameters(),
            self.action_head.parameters(),
            max_norm=self.hparams.actor_grad_clip_norm,
            foreach=True,  # GPU efficient
        )
        opt_dt.step()
        opt_alpha.step()
        if sch_dt is not None:
            sch_dt.step()

        # Logging
        with torch.no_grad():
            # Lightweight logs
            log_args = dict(on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log_dict(
                {
                    "train/alpha": alpha.item(),
                    "train/alpha_loss": alpha_loss.item(),
                    "train/entropy": masked_mean(mask, entropy).item(),
                    "train/mu": masked_mean(mask, mean.squeeze(-1)).item(),
                    "train/std": masked_mean(mask, std.squeeze(-1)).item(),
                    "train/actor_loss": actor_total_loss.item(),
                    "train/mean_l2_loss": mean_l2_loss.item(),
                    "train/std_l2_loss": std_l2_loss.item(),
                    "train/bc_log_prob": masked_mean(mask, bc_logprob).item(),
                    "train/actor_grad_norm": actor_grad_norm.item(),
                    "train/lr": float(opt_dt.param_groups[0]["lr"]),
                    "global_step": self.global_step,
                },
                **log_args,
            )

            # Heavy logs only every log_every_n_steps on rank 0
            if (
                self.logger is not None
                and self.trainer.is_global_zero
                and (self.global_step % self.trainer.log_every_n_steps == 0)
                and (self.scalers is not None)
            ):
                # Convert to numpy and inverse transform to raw scale
                batch_size, seq_len, action_dim = actions.shape
                flat_mask = mask.detach().cpu().numpy().reshape(-1).astype(bool)

                pred_actions = pi_actions.detach().cpu().numpy().reshape(-1, action_dim)
                target_actions = actions.detach().cpu().numpy().reshape(-1, action_dim)

                pred_actions_raw = self.scalers["action_scaler"].inverse_transform(pred_actions)
                target_actions_raw = self.scalers["action_scaler"].inverse_transform(target_actions)

                # Keep only valid (non-padding) positions
                pred_actions_raw = pred_actions_raw[flat_mask]
                target_actions_raw = target_actions_raw[flat_mask]

                # Mu and Std distributions (actor outputs in policy space)
                mu = mean.detach().cpu().numpy().reshape(-1, action_dim)[flat_mask]
                std_vals = std.detach().cpu().numpy().reshape(-1, action_dim)[flat_mask]

                # Log regression metrics in raw action space
                self.log_dict(
                    regression_report(
                        y_true=target_actions_raw,
                        y_pred=pred_actions_raw,
                        prefix="train/raw_action/",
                    ),
                    **log_args,
                )

                # Distribution logging with contexts
                distributions = [
                    ("actor", "pred_actions_raw", pred_actions_raw),
                    ("actor", "mu", mu),
                    ("actor", "std", std_vals),
                ]
                for name, group, data in distributions:
                    log_distribution(
                        logger=self.logger,
                        data=data,
                        name=name,
                        step=self.global_step,
                        context={"group": group},
                        percentile_range=(5, 95),
                    )

                # Additional raw-scale stats
                self.log_dict(
                    {
                        "train/target_actions_raw/mean": float(target_actions_raw.mean()),
                        "train/target_actions_raw/std": float(target_actions_raw.std()),
                        "train/pred_actions_raw/mean": float(pred_actions_raw.mean()),
                        "train/pred_actions_raw/std": float(pred_actions_raw.std()),
                        # Mu/Std summary stats (actor space)
                        "train/mu/mean": float(mu.mean()),
                        "train/mu/std": float(mu.std()),
                        "train/std/mean": float(std_vals.mean()),
                        "train/std/std": float(std_vals.std()),
                    },
                    **log_args,
                )
            
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        # No-op: ensure validation loop runs so on_validation_epoch_end is called
        return None

    def on_validation_epoch_end(self) -> None:
        # Run expensive evaluation on rank 0, broadcast results to all ranks, then log everywhere.
        mean_returns = 0.0
        mean_score = 0.0
        mean_budget_spent_ratio = 0.0

        if self.trainer.is_global_zero:
            agent_config = (
                DTBiddingAgent,
                "DT",
                {
                    "model": self,
                    "state_dim": self.hparams.state_dim,
                    "target_rtg": self.hparams.target_rtg,
                },
                {},
            )
            all_agent_configs = get_baseline_agent_configs(seed=DEFAULT_SEED)
            with torch.inference_mode():
                (
                    df_campaign_reports,
                    df_agent_summaries,
                    auction_histories,
                    agents,
                ) = self.trainer.datamodule.evaluator.evaluate(
                    control_agent_configs=agent_config,
                    all_agent_configs=all_agent_configs,
                    budget_ratio=1.0,
                    cpa_ratio=1.0,
                )
                df_val = (
                    summarize_agent_scores(df_agent_summaries, use_best_advertiser=False)
                    .filter(pl.col("agent_name") == "DT")
                )
                mean_returns = float(df_val.select("mean_returns").item())
                mean_score = float(df_val.select("mean_score").item())
                mean_budget_spent_ratio = float(df_val.select("mean_budget_spent_ratio").item())

        # Broadcast metrics from rank 0 to all ranks
        vals = torch.tensor(
            [mean_returns, mean_score, mean_budget_spent_ratio], dtype=torch.float32, device=self.device
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            vals = self.trainer.strategy.broadcast(vals, src=0)
        mean_returns, mean_score, mean_budget_spent_ratio = [float(x) for x in vals.tolist()]

        # Log on all ranks (no further dist sync needed since values are identical)
        self.log_dict(
            {
                "val/mean_returns": mean_returns,
                "val/mean_score": mean_score,
                "val/mean_budget_spent_ratio": mean_budget_spent_ratio,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
    
    @torch.inference_mode()
    def predict(
        self, 
        states: np.ndarray,
        actions: np.ndarray,
        rtgs: np.ndarray,
        time_steps: np.ndarray,
        mask: np.ndarray,
        curr_pointer: int,
    ) -> np.ndarray:
        # Prevent accidental training-mode inference
        if self.training:
            raise RuntimeError("Cannot call predict in training mode. Use model.eval() before inference.")

        # Scale states using the scalers
        batch_size, seq_len, state_dim = states.shape
        states = (
            self.scalers["state_scaler"]
            .transform(states.reshape(-1, state_dim))
            .reshape(batch_size, seq_len, state_dim)
        )
        states = states * mask[..., None]

        # Scale actions similarly to states
        _, _, action_dim = actions.shape
        actions = (
            self.scalers["action_scaler"]
            .transform(actions.reshape(-1, action_dim))
            .reshape(batch_size, seq_len, action_dim)
        )
        actions = actions * mask[..., None]
        # NOTE: do not use reward scaler as RTGs already been scaled in bidding step
        
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rtgs = torch.as_tensor(rtgs, dtype=torch.float32, device=self.device)
        time_steps = torch.as_tensor(time_steps, dtype=torch.int64, device=self.device)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        
        r_out, s_out, a_out = self.model(
            states=states,
            actions=actions,
            returns_to_go=rtgs,
            time_steps=time_steps,
        )
        mean, std = self.action_head(s_out)
        dist = BiasedSoftplusNormal(
            loc=mean,
            scale=std,
            # bias=1.0,
            safe_bound_eps=1e-6,
        )
        pred_actions = dist.deterministic_sample

        # take out the action prediction based on the current pointer
        if curr_pointer < self.hparams.seq_len:
            pred_actions = pred_actions[:, curr_pointer, :]
        else:
            # If pointer exceeds seq_len, take the last valid action
            pred_actions = pred_actions[:, -1, :]
            
        pred_action = pred_actions.detach().cpu().numpy()
        action = (
            self.scalers["action_scaler"]
            .inverse_transform(pred_action)
        )

        return action.squeeze(-1)
    
    def on_fit_start(self) -> None:
        self.scalers = self.trainer.datamodule.scalers
    
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.scalers is None:
            raise ValueError("Scalers missing: fit or assign scalers before training to enable checkpoint saving.")
        checkpoint["scalers"] = self.scalers
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.scalers = checkpoint["scalers"]


class DTBiddingAgent(BaseBiddingAgent):
    def __init__(
        self,
        adv_indicies: list[int] | np.ndarray,
        budget: list[float] | np.ndarray,
        cpa: list[float] | np.ndarray,
        category: list[int] | np.ndarray,
        name: str = "DT",
        model_dir: str | Path | None = None,
        checkpoint_file: str = "best.ckpt",
        state_dim: int | None = None,
        model: DTModel | None = None,
        target_rtg: float = 4.0,
    ) -> None:
        super().__init__(adv_indicies, budget, cpa, category, name)

        if not ((model_dir is None) ^ (model is None)):
            raise ValueError("Provide exactly one of model_dir or model.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is not None:
            self.model = model
        else:
            self.model = DTModel.load_from_checkpoint(
                Path(model_dir) / checkpoint_file,
                map_location=device,
                state_dim=state_dim,
                action_dim=1,
            )
        self.model.eval()
        
        self.target_rtg = target_rtg
        
    def reset_buffers(self) -> None:
        """Reset sequence buffers for decision transformer inference."""
        batch_size = self.num_advertisers
        seq_len = self.model.hparams.seq_len
        state_dim = self.model.hparams.state_dim
        action_dim = self.model.hparams.action_dim

        # Use DTInferenceBuffer instead of manual numpy ring buffers
        self.buffer = DTInferenceBuffer(
            B=batch_size,
            L=seq_len,
            state_dim=state_dim,
            action_dim=action_dim,
            dtype=np.float32,
        )

        # Track last predicted action for teacher forcing
        self.last_action = np.zeros((batch_size, action_dim), dtype=np.float32)

    def reset(self, budget=None, cpa=None, budget_ratio=None, cpa_ratio=None) -> None:
        super().reset(
            budget=budget, 
            cpa=cpa, 
            budget_ratio=budget_ratio, 
            cpa_ratio=cpa_ratio,
        )
        
        self.reset_buffers()

    def bidding(self, timeStepIndex: int, pValues: np.ndarray, pValueSigmas: np.ndarray, auction_history) -> np.ndarray:
        states = extract_state_features(
            adv_indices=self.adv_indicies,
            timeStepIndex=timeStepIndex,
            budget=self.budget,
            remaining_budget=self.remaining_budget,
            auction_history=auction_history,
            pValues=pValues,
            total_ticks=NUM_TICKS,
        )

        if timeStepIndex > 0:
            penalties, conversions_step = auction_history.compute_cpa_penalty_series(
                adv_indices=self.adv_indicies,
                penalty_power=2.0,  # TODO: hard-coded for now
            )
            # Penalized sum-so-far matches training: sum_{k<=t} penalty[k] * conversions_step[k]
            penalized_so_far = (penalties * conversions_step).sum(axis=0)
            current_rtg = self.target_rtg - penalized_so_far / self.model.hparams.rtg_scale
        else:
            current_rtg = np.full((self.num_advertisers,), self.target_rtg, dtype=np.float32)

        # Append this step to the inference buffer; only provide action_prev when t > 0
        self.buffer.append(
            state_t=states,
            rtg_t=current_rtg,
            t_t=timeStepIndex,
            action_prev=self.last_action if timeStepIndex > 0 else None,
        )

        # Pack into fixed-length arrays
        seq_states, seq_actions, seq_rtgs, seq_timesteps, seq_mask = self.buffer.pack()
        curr_pointer = min(len(self.buffer) - 1, self.model.hparams.seq_len - 1)

        # Use model's predict for normalization and inference
        alpha = self.model.predict(
            states=seq_states,
            actions=seq_actions,
            rtgs=seq_rtgs,
            time_steps=seq_timesteps,
            mask=seq_mask,
            curr_pointer=curr_pointer,
        )

        assert alpha.ndim == 1, f"Expected alpha to be 1D, got {alpha.ndim}D."
        # Store the current action prediction
        self.last_action[:] = alpha[:, None]

        alpha = alpha[None, :].clip(min=0.0)  # pValues: [opportunities, adv_indicies]
        return alpha * pValues
