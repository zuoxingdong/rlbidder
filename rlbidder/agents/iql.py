import copy
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
from rlbidder.constants import DEFAULT_SEED, NUM_TICKS
from rlbidder.evaluation import summarize_agent_scores
from rlbidder.models.distributions import BiasedSoftplusNormal
from rlbidder.models.networks import (
    EnsembledQNetwork,
    LearnableScalar,
    StochasticActor,
    ValueNetwork,
    polyak_update,
)
from rlbidder.models.optim import (
    LinearWarmupConstantCosineAnnealingLR,
    clip_grad_norm_fast,
    fix_global_step_for_multi_optimizers,
)
from rlbidder.models.utils import extract_state_features, grad_total_norm
from rlbidder.utils import log_distribution, regression_report


class IQLModel(L.LightningModule):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        cpa_penalty_beta: float = 2.0,
        actor_hidden_sizes: list[int] = [512, 512, 512],
        critic_hidden_sizes: list[int] = [512, 512, 512],
        value_hidden_sizes: list[int] = [512, 512, 512],
        num_q_models: int = 5,
        hlg_q_num_atoms: int = 101,
        hlg_q_vmin: float = 0.0,
        hlg_q_vmax: float = 20.0,
        hlg_q_sigma_to_bin_width_ratio: float = 0.75,
        hlg_q_prior_scale: float = 40.9,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_value: float = 3e-4,
        use_actor_scheduler: bool = True,
        bc_alpha: float = 0.01,  # BC coefficient in DDPG+BC
        alpha_init_value: float = 0.1,  # Initial value for learnable alpha
        target_entropy: float = -0.5,  # Target entropy for SAC-style entropy regularization
        enable_q_normalization: bool = True,  # Enable Q value normalization
        mean_l2_coeff: float = 0.0001,  # Coefficient on E[mean^2] regularizer
        std_l2_coeff: float = 0.001,  # Coefficient on E[std^2] regularizer
        expectile_tau: float = 0.8,
        actor_grad_clip_norm: float = 1.0,
        critic_grad_clip_norm: float = 1.0,
        value_grad_clip_norm: float = 1.0,
        val_metric: str = "mean_score",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Initialize networks
        self.actor = StochasticActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=actor_hidden_sizes,
            std_softplus_min=1e-2,
            std_softplus_bias=0.1,
            std_min=1e-2,
            std_max=1.0,
        )
        self.actor.init_weights()
        
        self.qf = EnsembledQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            num_models=num_q_models,
            hidden_sizes=critic_hidden_sizes,
            output_dim=self.hparams.hlg_q_num_atoms,
            layer_norm=True,
            vmin=self.hparams.hlg_q_vmin,
            vmax=self.hparams.hlg_q_vmax,
            sigma_to_bin_width_ratio=self.hparams.hlg_q_sigma_to_bin_width_ratio,
            prior_scale=self.hparams.hlg_q_prior_scale,
        )
        self.target_qf = copy.deepcopy(self.qf)
        self.target_qf.eval()
        for p in self.target_qf.parameters():
            p.requires_grad_(False)

        self.vf = ValueNetwork(
            state_dim=state_dim,
            hidden_sizes=value_hidden_sizes,
            layer_norm=True,
        )
        
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

    def configure_optimizers(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        # NOTE: Beta2=0.95, see https://github.com/facebookresearch/mae/issues/184#issuecomment-1861673795
        opt_actor = optim.AdamW(
            self.actor.parameters(),
            lr=self.hparams.lr_actor,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            fused=True,  # Use fused AdamW for more efficient training
        )
        opt_alpha = optim.AdamW(
            self.alpha.parameters(),
            lr=self.hparams.lr_actor,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            fused=True,
        )
        # Actor scheduler
        # https://arxiv.org/pdf/2406.04534#page=17.85
        sch_actor = LinearWarmupConstantCosineAnnealingLR(
            optimizer=opt_actor,
            warmup_epochs=0,  # No warmup for actor
            max_epochs=self.trainer.max_steps,
            warmup_start_lr=self.hparams.lr_actor,
            eta_min=1e-4,  # HACK: seems too low min actor lr hurt performance
            last_epoch=-1,
        )
        opt_qf = optim.AdamW(
            self.qf.parameters(),
            lr=self.hparams.lr_critic,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            fused=True,  # Use fused AdamW for more efficient training
        )
        opt_vf = optim.AdamW(
            self.vf.parameters(),
            lr=self.hparams.lr_value,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            fused=True,  # Use fused AdamW for more efficient training
        )
        
        return (
            {"optimizer": opt_actor, "lr_scheduler": {"scheduler": sch_actor}},
            {"optimizer": opt_alpha},
            {"optimizer": opt_qf},
            {"optimizer": opt_vf},
        )

    def update_value(
        self,
        opt_vf: optim.Optimizer,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Update value function using expectile regression."""
        with torch.no_grad():
            q = (
                self.target_qf(states, actions)[1].squeeze(-1)  # returns (logits, values) -> (num_models, batch_size)
                .topk(k=2, dim=0, largest=False).values.mean(dim=0)  # conservative aggregator -> (batch_size,)
                .detach()
            )

        v_values = self.vf(states).squeeze(-1)
        assert q.ndim == 1 and v_values.ndim == 1, "Q-values and values must be 1D tensors"

        tau = float(self.hparams.expectile_tau)
        diff = q - v_values
        weights = torch.where(diff < 0, 1.0 - tau, tau)
        value_loss = (weights * diff.square()).mean()

        # HACK: avoid incorrect gradient accumulation with multiple optimizers
        for opt in self.optimizers():
            opt.zero_grad(set_to_none=True)
        self.manual_backward(value_loss)
        value_grad_norm = clip_grad_norm_fast(
            self.vf.parameters(),
            max_norm=self.hparams.value_grad_clip_norm,
            foreach=True,  # GPU efficient
        )
        opt_vf.step()

        extra_info = {
            "Vs": v_values.detach(),
            "Advs": (q - v_values).detach(),
            "value_grad_norm": value_grad_norm.item(),
        }
        
        return value_loss, extra_info

    def update_critic(
        self,
        opt_qf: optim.Optimizer,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Update critic using TD learning."""
        with torch.no_grad():
            next_v = self.vf(next_states).squeeze(-1)
            assert next_v.ndim == 1, "Values must be 1D tensors"
            td_target = rewards + (1.0 - dones) * self.hparams.gamma * next_v
        
        q_logits, qs = self.qf(states, actions)  # [num_models, batch_size]
        
        critic_loss = self.qf.hlg_loss(
            q_logits, 
            td_target[..., None].expand_as(qs).detach(), 
            reduction="mean"
        )
        
        # HACK: avoid incorrect gradient accumulation with multiple optimizers
        for opt in self.optimizers():
            opt.zero_grad(set_to_none=True)
        self.manual_backward(critic_loss)
        critic_grad_norm = clip_grad_norm_fast(
            self.qf.parameters(),
            max_norm=self.hparams.critic_grad_clip_norm,
            foreach=True,  # GPU efficient
        )
        opt_qf.step()
        
        extra_info = {
            "Qs": qs.detach().flatten(),
            "td_targets": td_target.detach().flatten(),
            "critic_grad_norm": critic_grad_norm.item(),
        }
        
        return critic_loss, extra_info

    def update_actor(
        self,
        opt_actor: optim.Optimizer,
        opt_alpha: optim.Optimizer,
        sch_actor: optim.lr_scheduler._LRScheduler,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        extra_info = {}

        """Update actor using DDPG+BC with learnable entropy regularization."""
        alpha, log_alpha = self.alpha(return_log=True)
        
        mean, std = self.actor(states)
        dist = BiasedSoftplusNormal(mean, std)
        pi_actions, pi_logprobs = dist.rsample_and_log_prob()
        pi_logprobs = pi_logprobs.squeeze(-1)
        # # NOTE: analytic LogNormal entropy depends on μ -> pushes μ upward (bigger actions)
        # entropy = dist.entropy()
        # MC estimate of entropy
        entropy = -pi_logprobs
        assert (
            pi_actions.ndim == 2 and entropy.ndim == 1 and pi_logprobs.ndim == 1
        ), "Actions and log probabilities must be 2D and 1D tensors"

        # Policy Extraction: DDPG+BC loss >> AWR
        # see paper: Is Value Learning Really the Main Bottleneck in Offline RL?
        q = (
            self.qf(states, pi_actions)[1].squeeze(-1)  # returns (logits, values) -> (num_models, bs) -> (bs,)
            .topk(k=2, dim=0, largest=False).values  # bottom-k conservative aggregator
            .mean(dim=0)  # Take mean across ensemble models
        )
        assert q.ndim == 1 and pi_logprobs.ndim == 1, "Q-values and log probabilities must be 1D tensors"
        
        # Normalize Q values by the absolute mean to make the loss scale invariant.
        if self.hparams.enable_q_normalization:
            q_normalizer = q.abs().mean().detach()
            q = q / (q_normalizer + 1e-6)

        # Actor loss with learnable entropy regularization
        # avoid gradient flow to alpha
        actor_loss = -(q + alpha.detach() * entropy).mean()
        
        # Entropy loss (update alpha)
        # TODO: optimization in log_space of alpha to adapt faster, use raw alpha has too small loss scale
        # Increase alpha when entropy < target, decrease when > target
        alpha_loss = log_alpha * (entropy - self.hparams.target_entropy).detach().mean()
        
        # BC loss using dataset actions (tanh-bounded in [-1, 1])
        bc_logprob = dist.log_prob(actions).squeeze(-1)
        bc_loss = -(self.hparams.bc_alpha * bc_logprob).mean()

        # Mean/Std magnitude regularization with separate coefficients
        mean_l2_loss = self.hparams.mean_l2_coeff * (mean ** 2).mean()
        std_l2_loss = self.hparams.std_l2_coeff * (std ** 2).mean()

        # Only actor and BC loss for actor optimizer (no alpha loss)
        actor_total_loss = actor_loss + bc_loss + mean_l2_loss + std_l2_loss
        
        # NOTE: In manual optimization, global_step increments after the step;
        # Using (global_step + 1) aligns with the heavy logging cadence
        if ((self.global_step + 1) % self.trainer.log_every_n_steps == 0):
            g_Q_norm = grad_total_norm(-q.mean(), self.actor.parameters())
            g_E_norm = grad_total_norm(pi_logprobs.mean(), self.actor.parameters())
            # Define α_c ≈ ||g_Q|| / ||g_E||, and α_ratio = α / α_c
            # When α_ratio > 1, the entropy term dominates; expect σ to increase and BC logprob to drop. 
            alpha_ratio = alpha * g_E_norm / g_Q_norm
            extra_info.update({
                "g_Q_norm": g_Q_norm,
                "g_E_norm": g_E_norm,
                "alpha_ratio": alpha_ratio,
            })

        # HACK: avoid incorrect gradient accumulation with multiple optimizers
        for opt in self.optimizers():
            opt.zero_grad(set_to_none=True)
        
        # Update actor parameters
        self.manual_backward(actor_total_loss)
        actor_grad_norm = clip_grad_norm_fast(
            self.actor.parameters(),
            max_norm=self.hparams.actor_grad_clip_norm,
            foreach=True,  # GPU efficient
        )
        opt_actor.step()
        
        # Update alpha parameters separately
        self.manual_backward(alpha_loss)
        opt_alpha.step()
        
        if self.hparams.use_actor_scheduler:
            sch_actor.step()

        # Total loss for logging (combination of both losses)
        total_loss = actor_total_loss + alpha_loss

        extra_info.update({
            "alpha": alpha.detach(),
            "alpha_loss": alpha_loss.detach(),
            "entropy": entropy.detach(),
            "actor_loss": actor_loss.detach(),
            "bc_loss": bc_loss.detach(),
            "normed_q": q.detach(),
            "pi_actions": pi_actions.detach(),
            "pi_logprobs": pi_logprobs.detach(),
            "bc_logprob": bc_logprob.detach(),
            "temp_error": (entropy - self.hparams.target_entropy).detach(),
            "mu": mean.detach(),
            "std": std.detach(),
            "mean_l2_loss": mean_l2_loss.detach(),
            "std_l2_loss": std_l2_loss.detach(),
            "actor_grad_norm": actor_grad_norm.item(),
        })
            
        return total_loss, extra_info

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        opt_actor, opt_alpha, opt_qf, opt_vf = self.optimizers()
        sch_actor = self.lr_schedulers()
        fix_global_step_for_multi_optimizers(self.trainer, opt_alpha, opt_qf, opt_vf)

        states, actions, rewards, next_states, dones, cpa_compliance_ratios = batch

        # CPA-penalty reward shaping
        penalty = torch.where(
            cpa_compliance_ratios > 0,
            (cpa_compliance_ratios.clip(0, 1) ** self.hparams.cpa_penalty_beta),
            1,
        )
        rewards = rewards * penalty
        # Scale rewards
        rewards = rewards * self.hparams.reward_scale + self.hparams.reward_bias

        # === Value Network ===
        value_loss, value_extra_info = self.update_value(
            opt_vf=opt_vf,
            states=states,
            actions=actions,
        )

        # === Critic Network ===
        critic_loss, critic_extra_info = self.update_critic(
            opt_qf=opt_qf,
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=rewards,
            dones=dones,
        )

        # === Actor Network ===
        actor_loss, actor_extra_info = self.update_actor(
            opt_actor=opt_actor,
            opt_alpha=opt_alpha,
            sch_actor=sch_actor,
            states=states,
            actions=actions,
        )

        # Soft update target networks
        polyak_update(
            params=self.qf.parameters(),
            target_params=self.target_qf.parameters(),
            tau=self.hparams.tau,
        )

        # lightweight logs (aggregate to a single log_dict)
        log_args = dict(on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log_dict(
            {
                "train/actor_loss": actor_loss,
                "train/value_loss": value_loss,
                "train/critic_loss": critic_loss,
                "train/bc_loss": actor_extra_info["bc_loss"],
                "train/alpha_loss": actor_extra_info["alpha_loss"],
                "train/mean_l2_loss": actor_extra_info["mean_l2_loss"],
                "train/std_l2_loss": actor_extra_info["std_l2_loss"],
                "train/alpha": actor_extra_info["alpha"],
                "train/entropy": actor_extra_info["entropy"].mean(),
                "train/temp_error": actor_extra_info["temp_error"].mean(),
                "train/actor_grad_norm": actor_extra_info["actor_grad_norm"],
                "train/value_grad_norm": value_extra_info["value_grad_norm"],
                "train/critic_grad_norm": critic_extra_info["critic_grad_norm"],
                "train/actor_lr": float(opt_actor.param_groups[0]["lr"]),
                "global_step": self.global_step,
            },
            **log_args,
        )

        # heavy logs only every log_every_n_steps on rank 0 and when scalers are available
        if (
            self.logger is not None
            and self.trainer.is_global_zero
            and (self.global_step % self.trainer.log_every_n_steps == 0)
            and (self.scalers is not None)
        ):
            with torch.no_grad():
                target_actions_raw = self.scalers["action_scaler"].inverse_transform(actions.detach().cpu().numpy())
                pred_actions_raw = self.scalers["action_scaler"].inverse_transform(
                    actor_extra_info["pi_actions"].detach().cpu().numpy()
                )
                raw_action_diff = target_actions_raw - pred_actions_raw

                # Aggregate heavy scalar logs
                log_dict = {
                    "train/target_actions_raw/mean": float(target_actions_raw.mean()),
                    "train/target_actions_raw/std": float(target_actions_raw.std()),
                    "train/pred_actions_raw/mean": float(pred_actions_raw.mean()),
                    "train/pred_actions_raw/std": float(pred_actions_raw.std()),
                    "train/raw_action_diff/mean": float(raw_action_diff.mean()),
                    "train/Vs/mean": float(value_extra_info["Vs"].mean()),
                    "train/Vs/std": float(value_extra_info["Vs"].std()),
                    "train/Advs/mean": float(value_extra_info["Advs"].mean()),
                    "train/Advs/std": float(value_extra_info["Advs"].std()),
                    "train/Qs/mean": float(critic_extra_info["Qs"].mean()),
                    "train/Qs/std": float(critic_extra_info["Qs"].std()),
                    "train/td_targets/mean": float(critic_extra_info["td_targets"].mean()),
                    "train/td_targets/std": float(critic_extra_info["td_targets"].std()),
                    # Fractions of TD targets that hit HL-Gauss support edges (diagnose clipping)
                    "train/td_edge_hi": float(
                        (critic_extra_info["td_targets"] >= self.qf.hlg_loss.vmax)
                        .float()
                        .mean()
                    ),
                    "train/td_edge_lo": float(
                        (critic_extra_info["td_targets"] <= self.qf.hlg_loss.vmin)
                        .float()
                        .mean()
                    ),
                    "train/pi_logprobs/mean": float(actor_extra_info["pi_logprobs"].mean()),
                    "train/bc_logprob/mean": float(actor_extra_info["bc_logprob"].mean()),
                }
                # Add optional metrics if they exist
                for key in ["g_Q_norm", "g_E_norm", "alpha_ratio"]:
                    if key in actor_extra_info:
                        log_dict[f"train/{key}"] = float(actor_extra_info[key])
                
                self.log_dict(log_dict, **log_args)

                # Regression report for raw action predictions
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
                    ("value", "Vs", value_extra_info["Vs"].detach().cpu().numpy()),
                    ("value", "Advs", value_extra_info["Advs"].detach().cpu().numpy()),
                    ("critic", "Qs", critic_extra_info["Qs"].detach().cpu().numpy()),
                    ("critic", "td_targets", critic_extra_info["td_targets"].detach().cpu().numpy()),
                    ("actor", "pi_actions", actor_extra_info["pi_actions"].detach().cpu().numpy()),
                    ("actor", "pi_logprobs", actor_extra_info["pi_logprobs"].detach().cpu().numpy()),
                    ("actor", "bc_logprob", actor_extra_info["bc_logprob"].detach().cpu().numpy()),
                    ("actor", "mu", actor_extra_info["mu"].detach().cpu().numpy()),
                    ("actor", "std", actor_extra_info["std"].detach().cpu().numpy()),
                    ("data", "data_action", actions.detach().cpu().numpy()),
                    ("actor", "pred_actions_raw", pred_actions_raw),
                    ("actor", "raw_action_diff", raw_action_diff),
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

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        # No-op: ensure validation loop runs so on_validation_epoch_end is called
        return None

    def on_validation_epoch_end(self) -> None:
        # Evaluate via OnlineCampaignEvaluator on rank 0 and broadcast results
        mean_returns = 0.0
        mean_score = 0.0
        mean_budget_spent_ratio = 0.0

        if self.trainer.is_global_zero:
            agent_config = (
                IQLBiddingAgent,
                "IQL",
                {
                    "model": self,
                    "state_dim": self.hparams.state_dim,
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
                    .filter(pl.col("agent_name") == "IQL")
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
    def predict(self, state: np.ndarray) -> np.ndarray:
        if self.training:
            raise RuntimeError("Cannot call predict in training mode. Use model.eval() before inference.")
    
        if state.ndim != 2:
            raise ValueError(f"Expected state to be 2D, got {state.ndim}D.")

        state = self.scalers["state_scaler"].transform(state)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        mean, std = self.actor(state)
        action = (
            BiasedSoftplusNormal(mean, std).deterministic_sample
            .detach().cpu().numpy()
        )
        action = self.scalers["action_scaler"].inverse_transform(action)
        return action.squeeze(1)

    def on_fit_start(self) -> None:
        self.scalers = self.trainer.datamodule.scalers

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.scalers is None:
            raise ValueError("Scalers missing: fit or assign scalers before training to enable checkpoint saving.")
        checkpoint["scalers"] = self.scalers

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.scalers = checkpoint["scalers"]


class IQLBiddingAgent(BaseBiddingAgent):
    """
    IQL Agent
    """
    def __init__(
        self,
        adv_indicies: list[int] | np.ndarray,
        budget: list[float] | np.ndarray,
        cpa: list[float] | np.ndarray,
        category: list[int] | np.ndarray,
        name: str = "IQL",
        model_dir: str | Path | None = None,
        checkpoint_file: str = "best.ckpt",
        state_dim: int | None = None,
        model: IQLModel | None = None,
    ) -> None:
        super().__init__(adv_indicies, budget, cpa, category, name)

        if not ((model_dir is None) ^ (model is None)):
            raise ValueError("Provide exactly one of model_dir or model.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is not None:
            self.model = model
        else:
            self.model = IQLModel.load_from_checkpoint(
                Path(model_dir) / checkpoint_file,
                map_location=device,
                state_dim=state_dim,
                action_dim=1,
            )
        self.model.eval()

    def reset(
        self,
        budget: list[float] | np.ndarray | None = None,
        cpa: list[float] | np.ndarray | None = None,
        budget_ratio: float | None = None,
        cpa_ratio: float | None = None,
    ) -> None:
        super().reset(
            budget=budget, 
            cpa=cpa, 
            budget_ratio=budget_ratio, 
            cpa_ratio=cpa_ratio,
        )

    def bidding(
        self,
        timeStepIndex: int,
        pValues: np.ndarray,
        pValueSigmas: np.ndarray,
        auction_history: object,
    ) -> np.ndarray:
        states = extract_state_features(
            adv_indices=self.adv_indicies,
            timeStepIndex=timeStepIndex,
            budget=self.budget,
            remaining_budget=self.remaining_budget,
            auction_history=auction_history,
            pValues=pValues,
            total_ticks=NUM_TICKS,
        )

        # Use model's predict for normalization and inference
        alpha = self.model.predict(states)
        alpha = alpha[None, :].clip(min=0.0)
        return alpha * pValues
