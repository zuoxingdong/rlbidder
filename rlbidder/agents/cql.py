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
    polyak_update,
)
from rlbidder.models.optim import (
    LinearWarmupConstantCosineAnnealingLR,
    clip_grad_norm_fast,
    fix_global_step_for_multi_optimizers,
)
from rlbidder.models.utils import extract_state_features, grad_total_norm
from rlbidder.utils import log_distribution, regression_report


class CQLModel(L.LightningModule):
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
        num_models: int = 5,
        hlg_q_num_atoms: int = 101,
        hlg_q_vmin: float = 0.0,
        hlg_q_vmax: float = 20.0,
        hlg_q_sigma_to_bin_width_ratio: float = 0.75,
        hlg_q_prior_scale: float = 40.9,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        use_actor_scheduler: bool = True,
        temperature: float = 1.0,
        cql_alpha: float = 0.01,
        n_actions: int = 30,
        cql_target_action_gap: float = 10.0,
        bc_alpha: float = 0.01,
        alpha_init_value: float = 0.1,
        target_entropy: float = -0.5,
        enable_q_normalization: bool = True,
        mean_l2_coeff: float = 0.00001,  # Coefficient on E[mean^2] regularizer
        std_l2_coeff: float = 0.0001,   # Coefficient on E[std^2] regularizer
        ood_uniform_min: float = 0.0,  # OOD uniform min (BiasedSoftplusNormal -> positive actions)
        ood_uniform_max: float = 100.0, # OOD uniform max
        actor_grad_clip_norm: float = 1.0,
        critic_grad_clip_norm: float = 1.0,
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
        # Distributional critic with HLGauss
        self.qf = EnsembledQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            num_models=num_models,
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
        
        self.alpha = LearnableScalar(
            init_value=math.log(alpha_init_value),  # Initial value in log space
            log_space=True,
            min_log_value=math.log(1e-6),  # Minimum log alpha value
            max_log_value=math.log(1e9),   # Maximum log alpha value
        )
        self.alpha_prime = LearnableScalar(
            init_value=math.log(1.0),  # Initial value in log space (log(1.0) = 0.0)
            log_space=True,
            min_log_value=None,  # No minimum for alpha_prime
            max_log_value=math.log(1e6),  # Maximum log value for alpha_prime
        )

        # Constants and derived values (not hyperparameters)
        self.cql_clip_diff_min = -np.inf  # Q-function lower loss clipping
        self.cql_clip_diff_max = np.inf   # Q-function upper loss clipping

        # Placeholder for scalers
        # These will be set during training or loaded from checkpoint
        self.scalers = None

        # Enable manual optimization for custom update order
        self.automatic_optimization = False

    def configure_optimizers(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
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
            eta_min=1e-6,  # Minimum learning rate  # Align with IQL to avoid too-low LR
            last_epoch=-1,
        )
        opt_qf = optim.AdamW(
            self.qf.parameters(),
            lr=self.hparams.lr_critic,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            fused=True,  # Use fused AdamW for more efficient training
        )
        opt_alpha_prime = optim.AdamW(
            self.alpha_prime.parameters(),
            lr=self.hparams.lr_critic,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            fused=True,
        )
        
        return (
            {"optimizer": opt_actor, "lr_scheduler": {"scheduler": sch_actor}},
            {"optimizer": opt_alpha},
            {"optimizer": opt_qf},
            {"optimizer": opt_alpha_prime},
        )

    def update_actor(
        self,
        opt_actor: optim.Optimizer,
        opt_alpha: optim.Optimizer,
        sch_actor: optim.lr_scheduler._LRScheduler,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        extra_info = {}
        alpha, log_alpha = self.alpha(return_log=True)
        
        # Actor distribution (BiasedSoftplusNormal)
        mean, std = self.actor(states)
        dist = BiasedSoftplusNormal(mean, std)
        pi_actions, pi_logprobs = dist.rsample_and_log_prob()
        pi_logprobs = pi_logprobs.squeeze(-1)
        # # NOTE: analytic LogNormal entropy depends on μ -> pushes μ upward (bigger actions)
        # entropy = dist.entropy()
        # MC estimate of entropy
        entropy = -pi_logprobs
        assert pi_actions.ndim == 2 and entropy.ndim == 1 and pi_logprobs.ndim == 1, "Actions and log probabilities must be 2D and 1D tensors"

        # Q estimate on policy actions
        q = (
            self.qf(states, pi_actions)[1].squeeze(-1)  # (ensemble, batch)
            .topk(k=2, dim=0, largest=False).values.mean(dim=0)  # bottom-k conservative aggregator -> (batch,)
        )
        assert q.ndim == 1 and pi_logprobs.ndim == 1, "Q-values and log probabilities must be 1D tensors"
        
        # Normalize Q values by the absolute mean to make the loss scale invariant.
        if self.hparams.enable_q_normalization:
            q_normalizer = q.abs().mean().detach()
            q = q / (q_normalizer + 1e-6)
        
        # Actor loss with entropy regularization (avoid gradient flow to alpha)
        actor_loss = -(q + alpha.detach() * entropy).mean()
        
        # Temperature loss (update alpha) in log-space for better scaling
        # Increase alpha when entropy < target, decrease when > target
        alpha_loss = log_alpha * (entropy - self.hparams.target_entropy).detach().mean()
        
        # BC loss on dataset actions (tanh-bounded in [-1, 1])
        bc_logprob = dist.log_prob(actions).squeeze(-1)
        bc_loss = -(self.hparams.bc_alpha * bc_logprob).mean()

        # Separate mean/std magnitude regularization
        mean_l2_loss = self.hparams.mean_l2_coeff * (mean ** 2).mean()
        std_l2_loss = self.hparams.std_l2_coeff * (std ** 2).mean()

        # Total actor loss (no alpha loss)
        actor_total_loss = actor_loss + bc_loss + mean_l2_loss + std_l2_loss

        # Gradient norm diagnostics for collapse detection
        if ((self.global_step + 1) % self.trainer.log_every_n_steps == 0):
            g_Q_norm = grad_total_norm(-q.mean(), self.actor.parameters())
            g_E_norm = grad_total_norm(pi_logprobs.mean(), self.actor.parameters())
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
            foreach=True,
        )
        opt_actor.step()
        
        # Update alpha parameters separately
        self.manual_backward(alpha_loss)
        opt_alpha.step()
        
        # Step actor LR scheduler if enabled
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
            "mean_l2_loss": mean_l2_loss.detach(),
            "std_l2_loss": std_l2_loss.detach(),
            "normed_q": q.detach(),
            "pi_actions": pi_actions.detach(),
            "pi_logprobs": pi_logprobs.detach(),
            "bc_logprob": bc_logprob.detach(),
            "temp_error": (entropy - self.hparams.target_entropy).detach(),
            "mu": mean.detach(),
            "std": std.detach(),
            "actor_grad_norm": actor_grad_norm.item(),
        })

        return total_loss, extra_info

    def update_q_critic(
        self,
        opt_qf: optim.Optimizer,
        opt_alpha_prime: optim.Optimizer,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        # Compute Q distribution logits and values for dataset
        q_logits, qs = self.qf(states, actions)  # logits: [ensemble, batch, atoms], qs: [ensemble, batch, 1]
        qs = qs.squeeze(-1)  # [ensemble, batch]
        assert qs.ndim == 2, "Q-values must be 2D tensor (ensemble, batch)"
        
        # Sample Actions and Compute Log Probabilities
        # [batch_size, n_actions, action_dim]
        with torch.no_grad():
            batch_size, action_dim = actions.shape
            # 1) sample random actions (uniform in [u_min, u_max])
            u_min = float(self.hparams.ood_uniform_min)
            u_max = float(self.hparams.ood_uniform_max)
            random_actions = (
                actions.new_empty((batch_size, self.hparams.n_actions, action_dim), requires_grad=False)
                .uniform_(u_min, u_max)
            )
            random_actions_logprob = - action_dim * math.log(u_max - u_min)  # -log(b - a) for uniform(a, b)
            # 2) sample policy actions on current states and next states (BiasedSoftplusNormal policy)
            mean_all, std_all = self.actor(torch.cat([states, next_states], dim=0))
            all_actions, all_logprobs = (
                BiasedSoftplusNormal(mean_all, std_all)
                .rsample_and_log_prob((self.hparams.n_actions,))
            )
            pi_actions, next_actions = all_actions.chunk(2, dim=0)  # (B, n_actions, A), (B, n_actions, A)
            pi_logprobs, next_logprobs = all_logprobs.chunk(2, dim=0)  # (B, n_actions), (B, n_actions)

            target_q_values = (
                self.target_qf(next_states, next_actions[:, 0, :])[1].squeeze(-1)
                .topk(k=2, dim=0, largest=False).values.mean(dim=0)  # bottom-k conservative aggregator
            )
            assert target_q_values.ndim == 1
            
            td_target = rewards + (1.0 - dones) * self.hparams.gamma * target_q_values.detach()
            assert rewards.ndim == 1 and dones.ndim == 1 and td_target.ndim == 1

        # Compute Q-function losses with HLGauss
        qf_loss = self.qf.hlg_loss(
            q_logits,
            td_target.expand_as(qs)[..., None].detach(),
            reduction="mean",
        )

        # CQL Regularization

        # Compute Q-values for CQL regularization
        # HACK: batched forward pass for all Q-functions
        all_states = torch.cat([states, next_states], dim=0)  # [2*batch_size, state_dim]
        ood_actions = (
            torch.cat(
                [
                    random_actions,
                    pi_actions,
                    next_actions,
                ],
                dim=1,
            )
            .repeat(2, 1, 1)  # [2*batch_size, 3*n_actions, action_dim]
        )
        # avoid gradient flow to actor
        ood_actions = ood_actions.detach()
        # Single forward pass for each Q-function
        qs_ood_all = self.qf(all_states, ood_actions)[1].squeeze(-1)  # [ensemble, 2*batch, 3*n_actions]
        qs_ood, qs_ood_next = qs_ood_all.chunk(2, dim=1)  # Each: [ensemble, batch, 3*n_actions]
        qs_rand, qs_pi, _ = qs_ood.chunk(3, dim=2)  # Each: [ensemble, batch, n_actions]
        _, _, qs_next = qs_ood_next.chunk(3, dim=2)  # Each: [ensemble, batch, n_actions]
        assert (
            qs_ood.ndim == 3 and qs_ood_next.ndim == 3 and
            qs_rand.ndim == 3 and qs_pi.ndim == 3 and qs_next.ndim == 3 and
            pi_logprobs.ndim == 2 and next_logprobs.ndim == 2
        )
        
        # Importance sampling
        # avoid gradient flow to actor (logprobs)
        cql_cat_qs = torch.cat(
            [
                qs_rand - random_actions_logprob,
                qs_pi - pi_logprobs[None, ...].detach(),  # [ensemble_size, batch_size, n_actions]
                qs_next - next_logprobs[None, ...].detach(),  # [ensemble_size, batch_size, n_actions]
            ],
            dim=2,
        )
        assert cql_cat_qs.ndim == 3, "CQL concatenated Q-values must be 3D tensor (ensemble_size, batch_size, 3*n_actions)"
        cql_qs_ood = (
            (cql_cat_qs / self.hparams.temperature)
            .logsumexp(dim=2)  # [ensemble_size, batch_size]
            .mul(self.hparams.temperature)
        )
        assert cql_qs_ood.ndim == 2, "CQL Q-values for OOD actions must be 2D tensor (ensemble_size, batch_size)"
        
        cql_qs_diff = (
            (cql_qs_ood - qs)
            .clip(self.cql_clip_diff_min, self.cql_clip_diff_max)
            .mean(dim=-1)  # [ensemble_size,]
        )
        
        # Lagrange multiplier for CQL
        alpha_prime = self.alpha_prime()
        alpha_prime_loss = -(
            alpha_prime
            * self.hparams.cql_alpha
            * (cql_qs_diff - self.hparams.cql_target_action_gap).detach()  # avoid gradient flow to Q-values
        ).mean()
        
        for opt in self.optimizers():
            opt.zero_grad(set_to_none=True)
        # Update alpha_prime parameters separately
        self.manual_backward(alpha_prime_loss)
        opt_alpha_prime.step()
        
        # use newly updated alpha_prime
        alpha_prime = self.alpha_prime()
        # HACK: note that when using Lagrange multiplier, the CQL loss is also using alpha_prime!
        cql_loss = (
            alpha_prime.detach()  # avoid gradient flow to alpha_prime
            * self.hparams.cql_alpha
            * (cql_qs_diff - self.hparams.cql_target_action_gap)
        ).mean()
        
        # Only Q-function and CQL loss for Q-function optimizer (no alpha_prime loss)
        qf_total_loss = qf_loss + cql_loss
        
        # HACK: avoid incorrect gradient accumulation with multiple optimizers
        for opt in self.optimizers():
            opt.zero_grad(set_to_none=True)
        
        # Update Q-function parameters
        self.manual_backward(qf_total_loss)
        critic_grad_norm = clip_grad_norm_fast(
            self.qf.parameters(),
            max_norm=self.hparams.critic_grad_clip_norm,
            foreach=True,
        )
        opt_qf.step()

        # Total loss for logging (combination of both losses)
        total_loss = qf_total_loss + alpha_prime_loss

        extra_info = {
            "qf_values": qs.detach().flatten(),
            "target_q": target_q_values.detach().flatten(),
            "td_targets": td_target.detach().flatten(),
            "cql_qs_ood": cql_qs_ood.detach().flatten(),
            "cql_cat_qs": cql_cat_qs.detach().flatten(),
            "alpha_prime": alpha_prime.detach(),
            "qf_loss": qf_loss.detach(),
            "cql_loss": cql_loss.detach(),
            "alpha_prime_loss": alpha_prime_loss.detach(),
            "critic_grad_norm": critic_grad_norm.item(),
        }

        return total_loss, extra_info


    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        if not self.training:
            raise RuntimeError("Cannot call training_step in evaluation mode. Use model.train() before training.")

        opt_actor, opt_alpha, opt_qf, opt_alpha_prime = self.optimizers()
        sch_actor = self.lr_schedulers()

        # Fix global step tracking for multiple optimizers
        fix_global_step_for_multi_optimizers(self.trainer, opt_alpha, opt_qf, opt_alpha_prime)

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

        # === Q-networks ===
        critic_loss, critic_extra_info = self.update_q_critic(
            opt_qf=opt_qf,
            opt_alpha_prime=opt_alpha_prime,
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=rewards,
            dones=dones,
        )

        # === Actor ===
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

        # lightweight logs (aggregate)
        log_args = dict(on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log_dict(
            {
                "train/actor_loss": actor_loss,
                "train/critic_loss": critic_loss,
                "train/bc_loss": actor_extra_info["bc_loss"],
                "train/alpha_loss": actor_extra_info["alpha_loss"],
                "train/mean_l2_loss": actor_extra_info["mean_l2_loss"],
                "train/std_l2_loss": actor_extra_info["std_l2_loss"],
                "train/alpha": actor_extra_info["alpha"],
                "train/alpha_prime": critic_extra_info["alpha_prime"],
                "train/alpha_prime_loss": critic_extra_info["alpha_prime_loss"],
                "train/qf_loss": critic_extra_info["qf_loss"],
                "train/cql_loss": critic_extra_info["cql_loss"],
                "train/entropy": actor_extra_info["entropy"].mean(),
                "train/temp_error": actor_extra_info["temp_error"].mean(),
                "train/actor_lr": float(opt_actor.param_groups[0]["lr"]),
            },
            **log_args,
        )

        # heavy logs only every log_every_n_steps on rank 0 and when scalers are available
        if (
            self.trainer.is_global_zero
            and (self.global_step % self.trainer.log_every_n_steps == 0)
            and (self.scalers is not None)
        ):
            with torch.no_grad():
                target_actions_raw = self.scalers["action_scaler"].inverse_transform(
                    actions.detach().cpu().numpy()
                )
                pred_actions_raw = self.scalers["action_scaler"].inverse_transform(
                    actor_extra_info["pi_actions"].detach().cpu().numpy()
                )
                raw_action_diff = target_actions_raw - pred_actions_raw

                # Regression report for raw action predictions
                self.log_dict(
                    regression_report(
                        y_true=target_actions_raw,
                        y_pred=pred_actions_raw,
                        prefix="train/raw_action/",
                    ),
                    **log_args,
                )

                # Aggregate heavy scalar logs
                log_dict = {
                    "train/target_actions_raw/mean": float(target_actions_raw.mean()),
                    "train/target_actions_raw/std": float(target_actions_raw.std()),
                    "train/pred_actions_raw/mean": float(pred_actions_raw.mean()),
                    "train/pred_actions_raw/std": float(pred_actions_raw.std()),
                    "train/raw_action_diff/mean": float(raw_action_diff.mean()),
                    "train/td_targets/mean": float(critic_extra_info["td_targets"].mean()),
                    "train/td_targets/std": float(critic_extra_info["td_targets"].std()),
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
                    "train/target_q/mean": float(critic_extra_info["target_q"].mean()),
                    "train/target_q/std": float(critic_extra_info["target_q"].std()),
                    "train/qf_values/mean": float(critic_extra_info["qf_values"].mean()),
                    "train/qf_values/std": float(critic_extra_info["qf_values"].std()),
                    "train/cql_qs_ood/mean": float(critic_extra_info["cql_qs_ood"].mean()),
                    "train/cql_qs_ood/std": float(critic_extra_info["cql_qs_ood"].std()),
                    "train/cql_cat_qs/mean": float(critic_extra_info["cql_cat_qs"].mean()),
                    "train/cql_cat_qs/std": float(critic_extra_info["cql_cat_qs"].std()),
                }
                for key in ["g_Q_norm", "g_E_norm", "alpha_ratio"]:
                    if key in actor_extra_info:
                        log_dict[f"train/{key}"] = float(actor_extra_info[key])
                self.log_dict(log_dict, **log_args)

                # Distribution logging with contexts
                distributions = [
                    ("critic", "qf_values", critic_extra_info["qf_values"].detach().cpu().numpy()),
                    ("critic", "target_q", critic_extra_info["target_q"].detach().cpu().numpy()),
                    ("critic", "cql_qs_ood", critic_extra_info["cql_qs_ood"].detach().cpu().numpy()),
                    ("critic", "cql_cat_qs", critic_extra_info["cql_cat_qs"].detach().cpu().numpy()),
                    ("actor", "pi_actions", actor_extra_info["pi_actions"].detach().cpu().numpy()),
                    ("actor", "pi_logprobs", actor_extra_info["pi_logprobs"].detach().cpu().numpy()),
                    ("actor", "bc_logprob", actor_extra_info["bc_logprob"].detach().cpu().numpy()),
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
                CQLBiddingAgent,
                "CQL",
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
                    .filter(pl.col("agent_name") == "CQL")
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


class CQLBiddingAgent(BaseBiddingAgent):
    """
    CQL Agent
    """
    def __init__(
        self,
        adv_indicies: list[int] | np.ndarray,
        budget: list[float] | np.ndarray,
        cpa: list[float] | np.ndarray,
        category: list[int] | np.ndarray,
        name: str = "CQL",
        model_dir: str | Path | None = None,
        checkpoint_file: str = "best.ckpt",
        state_dim: int | None = None,
        model: CQLModel | None = None,
    ) -> None:
        super().__init__(adv_indicies, budget, cpa, category, name)

        if not ((model_dir is None) ^ (model is None)):
            raise ValueError("Provide exactly one of model_dir or model.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is not None:
            self.model = model
        else:
            self.model = CQLModel.load_from_checkpoint(
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
