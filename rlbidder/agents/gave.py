import math
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from rlbidder.models.transformers import DecisionTransformer as BaseDecisionTransformer
from rlbidder.models.transformers import DTInferenceBuffer
from rlbidder.models.utils import extract_state_features, masked_mean
from rlbidder.utils import log_distribution, regression_report


class GAVEDecisionTransformer(BaseDecisionTransformer):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 20,
        episode_len: int = NUM_TICKS,
        embedding_dim: int = 512,
        intermediate_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        # GAVE-specific
        beta_temperature: float = 5.0,
        gate_amp: float = 0.2,
        hlg_rtg_num_atoms: int = 251,
        hlg_rtg_vmin: float = 0.0,
        hlg_rtg_vmax: float = 1.0,
        hlg_rtg_sigma_to_bin_width_ratio: float = 0.75,
        hlg_rtg_prior_scale: float = 40.9,
    ) -> None:
        super().__init__(
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

        self.beta_temperature = beta_temperature
        self.gate_amp = gate_amp

        # Heads
        self.action_head = NormalHead(
            in_features=embedding_dim,
            out_features=action_dim,
            bias=True,
            std_softplus_min=1e-2,
            std_softplus_bias=0.1,
            std_min=1e-2,
            std_max=1.0,
        )
        # Scalar RTG value head (Huber/MSE style)
        self.rtg_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Linear(128, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.beta_head = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

        # Init heads
        self.rtg_head[-1].apply(lambda module: init_trunc_normal(module, std=1.0))
        self.beta_head[-1].apply(lambda module: init_trunc_normal(module, std=0.02))

    def _explore_tanh_space(
        self,
        actions: torch.Tensor,
        beta_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exploration in pre-tanh space (for tanh-bounded actions).

        Returns gate s in [1 - gate_amp, 1 + gate_amp] and explored actions.
        """
        safe_bound = 1.0 - 1e-6
        # Pre-squash delta in latent space: z_prev = atanh(a)
        z_prev = torch.atanh(actions.detach().clamp(-safe_bound, safe_bound))
        # Sigmoid gate in [1 - gate_amp, 1 + gate_amp], neutral at 1
        s = (1.0 - self.gate_amp) + (2.0 * self.gate_amp) * torch.sigmoid(
            beta_logits / self.beta_temperature
        )
        # Apply multiplicative gate in pre-squash space
        explored_action = torch.tanh(s * z_prev)
        return s, explored_action

    def _explore_scaled_sigmoid(
        self,
        actions: torch.Tensor,
        beta_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exploration via scaled-sigmoid multiplicative gate for positive actions.

        With BiasedSoftplusNormal, action space is positive and unbounded. We use a
        multiplicative gate s in [1 - gate_amp, 1 + gate_amp] (neutral at 1) and scale
        the previous action in action space directly.
        """
        s = (1.0 - self.gate_amp) + (2.0 * self.gate_amp) * torch.sigmoid(
            beta_logits / self.beta_temperature
        )
        explored_action = actions.detach() * s
        return s, explored_action

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> torch.FloatTensor:
        r_out, s_out, a_out = super().forward(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            time_steps=time_steps,
        )

        action_mean, action_std = self.action_head(s_out)

        if self.training:
            rtg_values = self.rtg_head(a_out)

            beta_logits = self.beta_head(s_out)
            # Main path: scaled-sigmoid exploration on positive action space
            s, explored_action = self._explore_scaled_sigmoid(actions=actions, beta_logits=beta_logits)

            # Re-run with explored actions to obtain explored RTG predictions
            _, _, a_out_explored = super().forward(
                states=states,
                actions=explored_action,
                returns_to_go=returns_to_go,
                time_steps=time_steps,
            )
            rtg_values_explored = self.rtg_head(a_out_explored)

            return (
                action_mean,
                action_std,
                rtg_values,
                rtg_values_explored,
                s,
                explored_action,
            )

        # Inference: return deterministic sample
        return BiasedSoftplusNormal(action_mean, action_std).deterministic_sample
    
    
class GAVEModel(L.LightningModule):
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
        max_action: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        rtg_scale: float = 46.0,  # scale for returns-to-go
        target_rtg: float = 2.0,  # target return-to-go for evaluation
        expectile: float = 0.7,
        time_dim: int = 512,
        beta_temperature: float = 50.0,
        gate_amp: float = 0.01,
        # HLG hyperparameters
        hlg_rtg_num_atoms: int = 101,
        hlg_rtg_vmin: float = 0.0,
        hlg_rtg_vmax: float = 20.0,
        hlg_rtg_sigma_to_bin_width_ratio: float = 0.75,
        hlg_rtg_prior_scale: float = 40.9,
        use_lr_scheduler: bool = True,
        lr_warmup_steps: int = 50_000,  # Number of warmup steps
        lr_constant_steps: int = 300_000,  # Number of constant steps
        lr_min: float = 1e-8,  # Minimum learning rate
        loss_coeff_action: float = 1.0,
        loss_coeff_return: float = 10.0,
        loss_coeff_exploration: float = 0.01,
        alpha_r: float = 10.0,  # Alpha parameter for exploration weighting
        bc_alpha: float = 0.1,
        target_entropy: float = -0.5,
        actor_grad_clip_norm: float = 10.0,
        mean_l2_coeff: float = 0.00001,
        std_l2_coeff: float = 0.0001,
        val_metric: str = "mean_score",
        alpha_init_value: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.model = GAVEDecisionTransformer(
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
            beta_temperature=beta_temperature,
            gate_amp=gate_amp,
            hlg_rtg_num_atoms=hlg_rtg_num_atoms,
            hlg_rtg_vmin=hlg_rtg_vmin,
            hlg_rtg_vmax=hlg_rtg_vmax,
            hlg_rtg_sigma_to_bin_width_ratio=hlg_rtg_sigma_to_bin_width_ratio,
            hlg_rtg_prior_scale=hlg_rtg_prior_scale,
        )
        self.model.apply(lambda m: init_trunc_normal(m, std=0.02))

        self.alpha = LearnableScalar(
            init_value=math.log(alpha_init_value),  # Initial value in log space
            log_space=True,
            min_log_value=math.log(1e-6),  # Minimum log alpha value
            max_log_value=math.log(1e9),   # Maximum log alpha value
        )
        
        # Placeholder for scalers
        # These will be set during training or loaded from checkpoint
        self.scalers = None

        # Enable manual optimization for custom update order (match DT best practices)
        self.automatic_optimization = False
        
    def configure_optimizers(self) -> tuple[dict[str, Any], dict[str, Any]]:
        # Match DT best practices: separate optimizers for model and alpha with scheduler on model
        # Use decay/no-decay param groups (exclude bias and norm/embedding from weight decay)
        decay_params, no_decay_params = get_decay_and_no_decay_params(self.model)
        opt_model = optim.AdamW(
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
                {"optimizer": opt_model},
                {"optimizer": opt_alpha},
            )

        sch_model = LinearWarmupConstantCosineAnnealingLR(
            optimizer=opt_model,
            warmup_epochs=self.hparams.lr_warmup_steps,
            max_epochs=self.trainer.max_steps,
            constant_epochs=self.hparams.lr_constant_steps,
            warmup_start_lr=self.hparams.lr_min,
            eta_min=self.hparams.lr_min,
            last_epoch=-1,
        )
        return (
            {"optimizer": opt_model, "lr_scheduler": {"scheduler": sch_model}},
            {"optimizer": opt_alpha},
        )
    
    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        # Match DT's manual optimization and masked losses (no fbgemm jagged ops)
        opt_model, opt_alpha = self.optimizers()
        sch_model = self.lr_schedulers() if self.hparams.use_lr_scheduler else None
        fix_global_step_for_multi_optimizers(self.trainer, opt_alpha)

        # Unpack batch (align with OfflineDTDataModule as used by DT)
        states, actions, rewards, penalties, dones, rtgs, time_steps, mask = batch
        time_steps = time_steps.masked_fill(~mask.to(torch.bool), 0)  # mask out padding timesteps explicitly

        scaled_rtgs = rtgs / self.hparams.rtg_scale
        target_rtgs = scaled_rtgs[:, 1:]  # [batch_size, seq_len]

        (
            action_mean,
            action_std,
            rtg_values,
            rtg_values_explored,
            beta_preds,
            explored_action,
        ) = self.model(
            states=states,
            actions=actions,
            returns_to_go=scaled_rtgs[..., :-1],
            time_steps=time_steps,
        )

        dist = BiasedSoftplusNormal(loc=action_mean, scale=action_std, safe_bound_eps=1e-6)
        bc_logprob = dist.log_prob(actions.detach()).squeeze(-1)  # [B, L]
        bc_logprob_explored = dist.log_prob(explored_action.detach()).squeeze(-1)  # [B, L]
        # # NOTE: analytic LogNormal entropy depends on μ -> pushes μ upward (bigger actions)
        # entropy = dist.entropy().squeeze(-1)
        pi_actions, pi_logprobs = dist.rsample_and_log_prob()  # (batch_size, seq_len)
        pi_logprobs = pi_logprobs.squeeze(-1)
        # MC estimate of entropy
        entropy = -pi_logprobs
        alpha, log_alpha = self.alpha(return_log=True)

        # Apply stable GAVE loss w/o value function
        # see https://github.com/Applied-Machine-Learning-Lab/GAVE/blob/master/code/bidding_train_env/baseline/dt/dt.py#L275-L282
        # Exploration weighting: encourage actions that improve predicted RTG
        wo = torch.sigmoid(self.hparams.alpha_r * (rtg_values_explored - rtg_values.detach())).squeeze(-1)  # [B, L]
        wo_frozen = wo.detach()

        # Actor loss: weighted BC + entropy
        weighted_bc_logprob = (1.0 - wo_frozen) * bc_logprob + wo_frozen * bc_logprob_explored  # [B, L]
        actor_loss_terms = self.hparams.bc_alpha * weighted_bc_logprob + alpha.detach() * entropy
        loss_actor = -masked_mean(mask, actor_loss_terms)

        # Mean/Std magnitude regularization (on action head outputs)
        mean_l2 = masked_mean(mask, (action_mean ** 2).squeeze(-1))
        std_l2 = masked_mean(mask, (action_std ** 2).squeeze(-1))
        mean_l2_loss = self.hparams.mean_l2_coeff * mean_l2
        std_l2_loss = self.hparams.std_l2_coeff * std_l2

        # Target RTG loss: scalar regression (Huber)
        loss_target_rtg_per = F.huber_loss(
            rtg_values.squeeze(-1),
            target_rtgs.detach(),
            delta=0.5,
            reduction="none",
        )
        loss_target_rtg = masked_mean(mask, loss_target_rtg_per)

        # Simple exploration regularizer
        loss_explore = masked_mean(mask, (1.0 - wo))

        # Total losses
        loss_main = (
            self.hparams.loss_coeff_action * loss_actor
            + self.hparams.loss_coeff_return * loss_target_rtg
            + self.hparams.loss_coeff_exploration * loss_explore
            + mean_l2_loss
            + std_l2_loss
        )
        alpha_loss = masked_mean(mask, alpha * (entropy - self.hparams.target_entropy).detach())

        # Optimize
        for opt in self.optimizers():
            opt.zero_grad(set_to_none=True)

        self.manual_backward(loss_main)
        actor_grad_norm = clip_grad_norm_fast(
            self.model.parameters(),
            max_norm=self.hparams.actor_grad_clip_norm,
            foreach=True,
        )
        opt_model.step()

        self.manual_backward(alpha_loss)
        opt_alpha.step()

        # Step LR scheduler when enabled
        if sch_model is not None:
            sch_model.step()

        # Logging
        with torch.no_grad():
            log_args = dict(on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log_dict(
                {
                    "train/loss": loss_main.item(),
                    "train/loss_actor": loss_actor.item(),
                    "train/loss_target_rtg": loss_target_rtg.item(),
                    "train/loss_explore": loss_explore.item(),
                    "train/loss_mean_l2": mean_l2_loss.item(),
                    "train/loss_std_l2": std_l2_loss.item(),
                    "train/entropy": masked_mean(mask, entropy).item(),
                    "train/alpha": alpha.item(),
                    "train/alpha_loss": alpha_loss.item(),
                    "train/bc_log_prob": masked_mean(mask, bc_logprob).item(),
                    "train/actor_grad_norm": actor_grad_norm.item(),
                    "train/lr": float(opt_model.param_groups[0]["lr"]),
                    "global_step": self.global_step,
                },
                **log_args,
            )

            # Heavy logs only periodically on rank 0
            if (
                self.trainer.is_global_zero
                and (self.global_step % self.trainer.log_every_n_steps == 0)
                and (self.scalers is not None)
            ):
                batch_size, seq_len, action_dim = actions.shape
                flat_mask = mask.detach().cpu().numpy().reshape(-1).astype(bool)

                pred_actions = dist.deterministic_sample.detach().cpu().numpy().reshape(-1, action_dim)
                target_actions = actions.detach().cpu().numpy().reshape(-1, action_dim)
                explored_actions = explored_action.detach().cpu().numpy().reshape(-1, action_dim)

                pred_actions_raw = self.scalers["action_scaler"].inverse_transform(pred_actions)
                target_actions_raw = self.scalers["action_scaler"].inverse_transform(target_actions)
                explored_actions_raw = self.scalers["action_scaler"].inverse_transform(explored_actions)
                explore_action_diff_raw = explored_actions_raw - target_actions_raw

                pred_actions_raw = pred_actions_raw[flat_mask]
                target_actions_raw = target_actions_raw[flat_mask]
                explore_action_diff_raw = explore_action_diff_raw[flat_mask]

                self.log_dict(
                    regression_report(
                        y_true=target_actions_raw,
                        y_pred=pred_actions_raw,
                        prefix="train/raw_action/",
                    ),
                    **log_args,
                )

                # Prepare flattened/filtered tensors for distribution logging
                beta_flat = beta_preds.detach().cpu().numpy().reshape(-1)[flat_mask]
                bc_logprob_flat = bc_logprob.detach().cpu().numpy().reshape(-1)[flat_mask]
                bc_logprob_explored_flat = bc_logprob_explored.detach().cpu().numpy().reshape(-1)[flat_mask]
                rtg_values_flat = rtg_values.detach().cpu().numpy().reshape(-1)[flat_mask]
                rtg_values_explored_flat = rtg_values_explored.detach().cpu().numpy().reshape(-1)[flat_mask]
                wo_flat = wo.detach().cpu().numpy().reshape(-1)[flat_mask]
                target_rtgs_flat = target_rtgs.detach().cpu().numpy().reshape(-1)[flat_mask]
                mu_flat = action_mean.detach().cpu().numpy().reshape(-1, action_dim)[flat_mask]
                std_flat = action_std.detach().cpu().numpy().reshape(-1, action_dim)[flat_mask]

                distributions = [
                    ("actor", "pred_actions_raw", pred_actions_raw),
                    ("actor", "mu", mu_flat),
                    ("actor", "std", std_flat),
                    ("actor", "beta", beta_flat),
                    ("actor", "explore_action_diff_raw", explore_action_diff_raw.squeeze(-1)),
                    ("actor", "bc_logprob", bc_logprob_flat),
                    ("actor", "bc_logprob_explored", bc_logprob_explored_flat),
                    ("value", "rtg_values", rtg_values_flat),
                    ("value", "rtg_values_explored", rtg_values_explored_flat),
                    ("exploration", "wo", wo_flat),
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

                # Additional stats
                # Compute RTG gap statistics (explored - predicted)
                rtg_gap_flat = (rtg_values_explored_flat - rtg_values_flat)

                self.log_dict(
                    {
                        "train/target_actions_raw/mean": float(target_actions_raw.mean()),
                        "train/target_actions_raw/std": float(target_actions_raw.std()),
                        "train/pred_actions_raw/mean": float(pred_actions_raw.mean()),
                        "train/pred_actions_raw/std": float(pred_actions_raw.std()),
                        "train/target_rtgs/mean": float(target_rtgs_flat.mean()),
                        "train/target_rtgs/std": float(target_rtgs_flat.std()),
                        "train/rtg_values/mean": float(rtg_values_flat.mean()),
                        "train/rtg_values/std": float(rtg_values_flat.std()),
                        "train/rtg_gap/mean": float(rtg_gap_flat.mean()),
                        "train/rtg_gap/std": float(rtg_gap_flat.std()),
                        "train/wo/mean": float(wo_flat.mean()),
                        "train/wo/std": float(wo_flat.std()),
                        "train/beta/mean": float(beta_flat.mean()),
                        "train/beta/std": float(beta_flat.std()),
                        "train/mu/mean": float(mu_flat.mean()),
                        "train/mu/std": float(mu_flat.std()),
                        "train/std/mean": float(std_flat.mean()),
                        "train/std/std": float(std_flat.std()),
                    },
                    **log_args,
                )

        # No return for manual optimization
        return None

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        # No-op: ensure validation loop runs so on_validation_epoch_end is called
        return None

    def on_validation_epoch_end(self) -> None:
        # Evaluate like DT and broadcast metrics
        mean_returns = 0.0
        mean_score = 0.0
        mean_budget_spent_ratio = 0.0

        if self.trainer.is_global_zero:
            agent_config = (
                GAVEBiddingAgent,
                "GAVE",
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
                    .filter(pl.col("agent_name") == "GAVE")
                )
                mean_returns = float(df_val.select("mean_returns").item())
                mean_score = float(df_val.select("mean_score").item())
                mean_budget_spent_ratio = float(df_val.select("mean_budget_spent_ratio").item())

        vals = torch.tensor(
            [mean_returns, mean_score, mean_budget_spent_ratio], dtype=torch.float32, device=self.device
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            vals = self.trainer.strategy.broadcast(vals, src=0)
        mean_returns, mean_score, mean_budget_spent_ratio = [float(x) for x in vals.tolist()]

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
        batch_size, seq_len, action_dim = actions.shape
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
        
        pred_actions = self.model(
            states=states,
            actions=actions,
            returns_to_go=rtgs,
            time_steps=time_steps,
        )
        
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
    

class GAVEBiddingAgent(BaseBiddingAgent):
    def __init__(
        self,
        adv_indicies: list[int] | np.ndarray,
        budget: list[float] | np.ndarray,
        cpa: list[float] | np.ndarray,
        category: list[int] | np.ndarray,
        name: str = "GAVE",
        model_dir: str | Path | None = None,
        checkpoint_file: str = "best.ckpt",
        state_dim: int | None = None,
        model: GAVEModel | None = None,
        target_rtg: float = 4.0,
    ) -> None:
        super().__init__(adv_indicies, budget, cpa, category, name)

        if not ((model_dir is None) ^ (model is None)):
            raise ValueError("Provide exactly one of model_dir or model.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is not None:
            self.model = model
        else:
            self.model = GAVEModel.load_from_checkpoint(
                Path(model_dir) / checkpoint_file,
                map_location=device,
                state_dim=state_dim,
                action_dim=1,
            )
        self.model.eval()
        
        self.target_rtg = target_rtg
        
    def reset_buffers(self) -> None:
        """Reset sequence buffers for decision transformer inference (use DTInferenceBuffer)."""
        batch_size = self.num_advertisers
        seq_len = self.model.hparams.seq_len
        state_dim = self.model.hparams.state_dim
        action_dim = self.model.hparams.action_dim

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

        # RTG update consistent with DT: apply CPA penalty to conversions
        if timeStepIndex > 0:
            penalties, conversions_step = auction_history.compute_cpa_penalty_series(
                adv_indices=self.adv_indicies,
                penalty_power=2.0,
            )
            penalized_so_far = (penalties * conversions_step).sum(axis=0)
            current_rtg = self.target_rtg - penalized_so_far / self.model.hparams.rtg_scale
        else:
            current_rtg = np.full((self.num_advertisers,), self.target_rtg, dtype=np.float32)

        # Append to buffer; provide prev action only when t > 0
        self.buffer.append(
            state_t=states,
            rtg_t=current_rtg,
            t_t=timeStepIndex,
            action_prev=self.last_action if timeStepIndex > 0 else None,
        )

        # Pack into fixed-length arrays
        seq_states, seq_actions, seq_rtgs, seq_timesteps, seq_mask = self.buffer.pack()
        curr_pointer = min(len(self.buffer) - 1, self.model.hparams.seq_len - 1)

        # Predict action
        alpha = self.model.predict(
            states=seq_states,
            actions=seq_actions,
            rtgs=seq_rtgs,
            time_steps=seq_timesteps,
            mask=seq_mask,
            curr_pointer=curr_pointer,
        )

        assert alpha.ndim == 1, f"Expected alpha to be 1D, got {alpha.ndim}D."
        self.last_action[:] = alpha[:, None]

        alpha = alpha[None, :].clip(min=0.0)
        return alpha * pValues
