import copy
from pathlib import Path
from typing import Any, Iterable

import lightning as L
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim

from rlbidder.agents.base import BaseBiddingAgent
from rlbidder.agents.dt import DTModel
from rlbidder.agents.get_baseline_agent_configs import get_baseline_agent_configs
from rlbidder.constants import DEFAULT_SEED, NUM_TICKS, STATE_DIM
from rlbidder.evaluation import summarize_agent_scores
from rlbidder.models.networks import (
    EnsembledQNetwork,
    ValueNetwork,
    init_trunc_normal,
    polyak_update,
)
from rlbidder.models.optim import (
    LinearWarmupConstantCosineAnnealingLR,
    clip_grad_norm_fast,
    get_decay_and_no_decay_params,
)
from rlbidder.models.transformers import DecisionTransformer, DTInferenceBuffer
from rlbidder.models.utils import extract_state_features, masked_mean
from rlbidder.utils import log_distribution


class QTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int,
        episode_len: int,
        embedding_dim: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        embedding_dropout: float,
        num_q_models: int,
        q_bottom_k: int,
        hlg_q_num_atoms: int,
        hlg_q_vmin: float,
        hlg_q_vmax: float,
        hlg_q_sigma_to_bin_width_ratio: float,
        hlg_q_prior_scale: float,
    ) -> None:
        super().__init__()

        self.backbone = DecisionTransformer(
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
        # Initialize transformer backbone like DT/GAVE
        self.backbone.apply(lambda m: init_trunc_normal(m, std=0.02))

        # Q ensemble (critic) and its Polyak target
        self.qf = EnsembledQNetwork(
            state_dim=embedding_dim,
            action_dim=embedding_dim,
            num_models=num_q_models,
            hidden_sizes=[embedding_dim, embedding_dim, embedding_dim],
            output_dim=hlg_q_num_atoms,
            layer_norm=True,
            last_layer_gain=np.sqrt(2),  # HACK: avoid tiny std for Q-ensemble
            vmin=hlg_q_vmin,
            vmax=hlg_q_vmax,
            sigma_to_bin_width_ratio=hlg_q_sigma_to_bin_width_ratio,
            prior_scale=hlg_q_prior_scale,
        )
        self.target_qf = copy.deepcopy(self.qf)
        self.target_qf.eval()
        for p in self.target_qf.parameters():
            p.requires_grad_(False)

        # Aggregator hyperparameter for conservative Q-targets
        self.q_bottom_k = q_bottom_k

        # State value network
        self.vf = ValueNetwork(
            state_dim=embedding_dim,
            hidden_sizes=[embedding_dim, embedding_dim, embedding_dim],
            layer_norm=True,
        )

    def forward(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        time_steps: torch.Tensor,
        mask: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        expectile: float,
        gamma: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        r_out, s_out, a_out = self.backbone(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go * 0.0,  # NOTE: not used rtgs for GAS
            time_steps=time_steps,
        )

        # === Value Network === (using flattened slice with next-state availability)
        batch_size, seq_len, _ = s_out.shape
        with torch.no_grad():
            tgt_q_logits, tgt_q_values = self.target_qf(
                s_out.flatten(end_dim=1).detach(),
                a_out.flatten(end_dim=1).detach(),
            )
            # values: (num_models, B*T, 1) -> (num_models, B, T)
            tgt_q_values = tgt_q_values.reshape(tgt_q_values.shape[0], batch_size, seq_len, -1).squeeze(-1)
            # Conservative aggregator across Q-ensemble models: bottom-k mean with configurable k
            k = int(max(1, min(self.q_bottom_k, tgt_q_values.shape[0])))
            q_target = (
                tgt_q_values
                .topk(k=k, dim=0, largest=False).values.mean(dim=0)  # -> (B, T)
                .detach()
            )
        v_values = self.vf(s_out)  # (B, T, 1)
        # Tau-expectile regression toward conservative Q target
        diff = q_target - v_values.squeeze(-1)  # (B, T)
        tau = float(expectile)
        weights = torch.where(diff < 0, 1.0 - tau, tau)
        value_loss = masked_mean(mask, weights * diff.square())

        # Critic loss: TD with bootstrap from V(next)
        with torch.no_grad():
            v_tp1 = self.vf(s_out[:, 1:, :].detach())
            v_tp1 = v_tp1.squeeze(-1)
            td_target = rewards[:, :-1] + (1.0 - dones[:, :-1]) * gamma * v_tp1  # (B, T-1)

        q_logits, q_values = self.qf(
            s_out[:, :-1, :].flatten(end_dim=1), 
            a_out[:, :-1, :].flatten(end_dim=1),
        )
        # reshape to (num_models, B, T, ...)
        num_models = q_values.shape[0]
        q_loss = (
            self.qf.hlg_loss(
                q_logits,
                td_target[None, ...].expand(num_models, -1, -1)[..., None].detach(),
                reduction="none",
            )
            .reshape(num_models, batch_size, seq_len - 1)
            .mean(dim=0)
        )
        # Reduce atoms/extra dims if present, then average across models and apply mask
        critic_loss = masked_mean(mask[:, :-1], q_loss)

        # For logging, return per-timestep Q-values and V-values
        with torch.no_grad():
            q_reshaped = q_values.reshape(num_models, batch_size, seq_len - 1)
            advs = q_reshaped - v_values[None, :, :-1]
            expanded_mask = mask[None, :, :-1].expand_as(q_reshaped)
            extra_info = {
                "td_targets": td_target[mask[:, :-1]].detach(),
                "Qs": q_reshaped[expanded_mask].detach(),
                "Vs": v_values.squeeze(-1)[mask].detach(),
                "Advs": advs[expanded_mask].detach(),
            }
            
        return value_loss, critic_loss, extra_info

    def soft_update_target_qf(self, tau: float) -> None:
        polyak_update(
            params=self.qf.parameters(),
            target_params=self.target_qf.parameters(),
            tau=tau,
        )

    def parameters_for_optimizer(self) -> dict[str, Iterable[torch.Tensor]]:
        return {
            "backbone": self.backbone.parameters(),
            "qf": self.qf.parameters(),
            "vf": self.vf.parameters(),
        }


class GASModel(L.LightningModule):
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
        use_lr_scheduler: bool = True,
        lr_warmup_steps: int = 50_000,  # Number of warmup steps
        lr_constant_steps: int = 300_000,  # Number of constant steps
        lr_min: float = 1e-8,  # Minimum learning rate
        val_metric: str = "mean_score",
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        gamma: float = 0.99,
        tau: float = 0.01,
        expectile: float = 0.7,
        critic_grad_clip_norm: float = 10.0,
        num_q_models: int = 5,
        q_bottom_k: int = 2,
        num_networks: int = 3,
        num_q_eval_actions: int = 20,
        action_sampling_ratio: float = 0.01,
        backbone_model_dir: str = None,
        backbone_checkpoint_file: str = "best.ckpt",
        hlg_q_num_atoms: int = 151,
        hlg_q_vmin: float = 0.0,
        hlg_q_vmax: float = 10.0,
        hlg_q_sigma_to_bin_width_ratio: float = 0.75,
        hlg_q_prior_scale: float = 40.9,
        loss_coeff_value: float = 100.0,
        loss_coeff_critic: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.np_rng = np.random.Generator(np.random.PCG64())
        
        # Build ensemble of subnetworks
        self.networks = nn.ModuleList(
            [
                QTransformer(
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
                    num_q_models=self.hparams.num_q_models,
                    q_bottom_k=self.hparams.q_bottom_k,
                    hlg_q_num_atoms=self.hparams.hlg_q_num_atoms,
                    hlg_q_vmin=self.hparams.hlg_q_vmin,
                    hlg_q_vmax=self.hparams.hlg_q_vmax,
                    hlg_q_sigma_to_bin_width_ratio=self.hparams.hlg_q_sigma_to_bin_width_ratio,
                    hlg_q_prior_scale=self.hparams.hlg_q_prior_scale,
                )
                for _ in range(self.hparams.num_networks)
            ]
        )

        # Load backbone model
        self.backbone_model = DTModel.load_from_checkpoint(
            Path(self.hparams.backbone_model_dir) / self.hparams.backbone_checkpoint_file,
            map_location=self.device,
            state_dim=self.hparams.state_dim,
            action_dim=1,
        )
        self.backbone_model.eval()
        self.backbone_model.freeze()

        # Placeholder for scalers
        # These will be set during training or loaded from checkpoint
        self.scalers = None

        # Enable manual optimization for custom update order
        self.automatic_optimization = False
        
    def configure_optimizers(self) -> dict[str, Any] | dict:
        # Single optimizer with parameter groups for each subnetwork (DT, Q, V)
        param_groups = []
        for net in self.networks:
            params = net.parameters_for_optimizer()
            # Apply DT-style decay/no-decay to backbone only (keep qf/vf no decay)
            decay_params, no_decay_params = get_decay_and_no_decay_params(net.backbone)
            param_groups.extend([
                {
                    "params": decay_params,
                    "lr": self.hparams.lr,
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay_params,
                    "lr": self.hparams.lr,
                    "weight_decay": 0.0,
                },
                {
                    "params": list(params["qf"]),
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                },
                {
                    "params": list(params["vf"]),
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                },
            ])
        opt = optim.AdamW(
            param_groups, 
            betas=(0.9, 0.95),
            fused=True,
        )

        if not self.hparams.use_lr_scheduler:
            return {"optimizer": opt}

        sch = LinearWarmupConstantCosineAnnealingLR(
            optimizer=opt,
            warmup_epochs=self.hparams.lr_warmup_steps,
            max_epochs=self.trainer.max_steps,
            constant_epochs=self.hparams.lr_constant_steps,
            warmup_start_lr=self.hparams.lr_min,
            eta_min=self.hparams.lr_min,
            last_epoch=-1,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch}}
    
    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        if not self.training:
            raise RuntimeError("Cannot call training_step in evaluation mode. Use model.train() before training.")

        opt = self.optimizers()
        sch = self.lr_schedulers() if self.hparams.use_lr_scheduler else None

        states, actions, rewards, penalties, dones, rtgs, time_steps, mask = batch
        time_steps = time_steps.masked_fill(~mask.to(torch.bool), 0)  # mask out padding timesteps
        scaled_rtgs = rtgs / self.hparams.rtg_scale

        # CPA-penalty reward shaping
        rewards = rewards * penalties

        # Scale rewards
        rewards = rewards * self.hparams.reward_scale + self.hparams.reward_bias
        
        # Forward through each subnetwork and compute respective losses
        extra_infos = []
        value_losses = []
        critic_losses = []

        for net in self.networks:
            v_loss, c_loss, extra_info = net(
                states=states,
                actions=actions,
                returns_to_go=scaled_rtgs[..., :-1],
                time_steps=time_steps,
                mask=mask,
                rewards=rewards,
                dones=dones,
                expectile=self.hparams.expectile,
                gamma=self.hparams.gamma,
            )

            extra_infos.append(extra_info)
            value_losses.append(v_loss)
            critic_losses.append(c_loss)

        value_loss = self.hparams.loss_coeff_value * torch.stack(value_losses).mean()
        critic_loss = self.hparams.loss_coeff_critic * torch.stack(critic_losses).mean()
        total_loss = value_loss + critic_loss

        # optimizer step
        opt.zero_grad(set_to_none=True)
        self.manual_backward(total_loss)
        # Only clip gradients for the trainable networks, not the frozen backbone model
        critic_grad_norms = []
        for net in self.networks:
            grad_norm = clip_grad_norm_fast(
                net.parameters(),
                max_norm=self.hparams.critic_grad_clip_norm,
                foreach=True,
            )
            critic_grad_norms.append(grad_norm)
        critic_grad_norm = torch.stack(critic_grad_norms).mean()
        opt.step()
        # Step LR scheduler if enabled
        if sch is not None:
            sch.step()
        
        # Soft update target networks
        for net in self.networks:
            net.soft_update_target_qf(self.hparams.tau)

        # Logging
        with torch.no_grad():
            log_args = dict(on_step=True, on_epoch=False, prog_bar=True, logger=True)
            log_dict = {
                "train/total_loss": total_loss.item(),
                "train/value_loss": value_loss.item(),
                "train/critic_loss": critic_loss.item(),
                "train/critic_grad_norm": critic_grad_norm.item(),
                "train/lr": float(opt.param_groups[0]["lr"]),
            }
            log_dict.update({"global_step": self.global_step})
            self.log_dict(log_dict, **log_args)
            
            # Heavy logs every log_every_n_steps on rank 0, using sequential mask like DT
            if (
                self.logger is not None
                and self.trainer.is_global_zero
                and (self.global_step % self.trainer.log_every_n_steps == 0)
            ):
                # Aggregate Q over Q-ensemble dimension to align with training loss averaging
                extra_info = {k: torch.stack([info[k] for info in extra_infos]).flatten() for k in extra_infos[0]}
                valid_vs = extra_info["Vs"]
                valid_qs = extra_info["Qs"]
                valid_td = extra_info["td_targets"]
                valid_advs = extra_info["Advs"]

                self.log_dict(
                    {
                        "train/Vs/mean": float(valid_vs.mean().item()),
                        "train/Vs/std": float(valid_vs.std(unbiased=False).item()),
                        "train/Qs/mean": float(valid_qs.mean().item()),
                        "train/Qs/std": float(valid_qs.std(unbiased=False).item()),
                        "train/td_targets/mean": float(valid_td.mean().item()),
                        "train/td_targets/std": float(valid_td.std(unbiased=False).item()),
                        "train/Advs/mean": float(valid_advs.mean().item()),
                        "train/Advs/std": float(valid_advs.std(unbiased=False).item()),
                    },
                    **log_args,
                )

                # Distribution logging with contexts via a compact loop
                distributions = [
                    ("value", "Vs", valid_vs.detach().cpu().numpy()),
                    ("critic", "Qs", valid_qs.detach().cpu().numpy()),
                    ("critic", "td_targets", valid_td.detach().cpu().numpy()),
                    ("value", "Advs", valid_advs.detach().cpu().numpy()),
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
        # Run expensive evaluation on rank 0, broadcast results to all ranks, then log everywhere.
        mean_returns = 0.0
        mean_score = 0.0
        mean_budget_spent_ratio = 0.0

        if self.trainer.is_global_zero:
            agent_config = (
                GASBiddingAgent,
                "GAS",
                {
                    "model": self,
                    "state_dim": self.hparams.state_dim,
                    "target_rtg": self.hparams.target_rtg,
                    "backbone_model": self.backbone_model,
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
                    .filter(pl.col("agent_name") == "GAS")
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
        backbone_actions: np.ndarray,
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

        # Backbone actions are provided in RAW space
        backbone_actions_raw = backbone_actions.reshape(batch_size, action_dim)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rtgs = torch.as_tensor(rtgs, dtype=torch.float32, device=self.device)
        time_steps = torch.as_tensor(time_steps, dtype=torch.int64, device=self.device)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        
        
        num_agents = actions.shape[0]
        # Sample candidate actions in RAW space around backbone action, then scale for Q-eval
        ensemble_size = self.hparams.num_q_eval_actions
        eps_range = self.hparams.action_sampling_ratio
        eps = self.np_rng.uniform((1.0 - eps_range), (1.0 + eps_range), size=(num_agents, ensemble_size, action_dim))
        action_ensembles_raw = (backbone_actions_raw[:, None, :] * eps).squeeze(-1)
        # Include original backbone raw actions in the ensemble
        action_ensembles_raw = np.concatenate([
            backbone_actions_raw[:, None, :].squeeze(-1),
            action_ensembles_raw
        ], axis=1)
        ensemble_size += 1
        # Scale raw candidates for model evaluation
        action_ensembles = (
            self.scalers["action_scaler"]
            .transform(action_ensembles_raw.reshape(-1, action_dim))
            .reshape(num_agents, ensemble_size, action_dim)
            .squeeze(-1)
        )
        action_ensembles = torch.as_tensor(action_ensembles, dtype=torch.float32, device=self.device)

        # Expand actions to include ensemble dimension
        batch_size, seq_len, action_dim = actions.shape
        expanded_actions = actions.unsqueeze(1).expand(num_agents, ensemble_size, seq_len, action_dim).clone()
        # Expand other tensors similarly
        expanded_states = (
            states.unsqueeze(1).expand(num_agents, ensemble_size, seq_len, state_dim).clone())
        expanded_rtgs = rtgs.unsqueeze(1).expand(num_agents, ensemble_size, seq_len).clone()
        expanded_time_steps = time_steps.unsqueeze(1).expand(num_agents, ensemble_size, seq_len).clone()
        # expanded_mask = mask.unsqueeze(1).expand(num_agents, ensemble_size, seq_len).clone()

        # Fill in the sampled action based on current pointer
        if curr_pointer < self.hparams.seq_len:
            expanded_actions[:, :, curr_pointer, :] = action_ensembles[..., None]
        else:
            # If pointer exceeds seq_len, fill the last position
            expanded_actions[:, :, -1, :] = action_ensembles[..., None]
        
        # Aggregate Q predictions across subnetworks
        q_preds = []
        adv_preds = []
        for net in self.networks:
            r_out, s_out, a_out = net.backbone(
                states=expanded_states.flatten(end_dim=1),
                actions=expanded_actions.flatten(end_dim=1),
                returns_to_go=expanded_rtgs.flatten(end_dim=1) * 0.0,  # NOTE: not used rtgs for GAS
                time_steps=expanded_time_steps.flatten(end_dim=1),
            )
            
            # take out s_out, a_out based on the current pointer
            if curr_pointer < self.hparams.seq_len:
                s_sel = s_out[:, curr_pointer, :]
                a_sel = a_out[:, curr_pointer, :]
            else:
                # If pointer exceeds seq_len, take the last valid s_out, a_out
                s_sel = s_out[:, -1, :]
                a_sel = a_out[:, -1, :]


            # Q(s, a) for each candidate action
            _, q_values = net.qf(s_sel, a_sel)
            k = int(max(1, min(self.hparams.q_bottom_k, q_values.shape[0])))
            q_pred = (
                q_values.squeeze(-1)
                .topk(k=k, dim=0, largest=False).values  # bottom-k conservative aggregator
                .mean(dim=0)
                .reshape(num_agents, ensemble_size)
            )
            q_preds.append(q_pred)

            # V(s), it should be the same for all candidate actions
            s_sel_ = s_sel.reshape(num_agents, ensemble_size, -1)[:, 0, :]
            v_pred = net.vf(s_sel_).squeeze(-1)  # [B]
            adv_preds.append(q_pred - v_pred[:, None])

        q_preds = torch.stack(q_preds, dim=1)
        adv_preds = torch.stack(adv_preds, dim=1)
        
        # Choose either Q-voting or advantage voting
        vote_type = "adv"
        if vote_type == "q":
            vote_values = q_preds
        elif vote_type == "adv":
            vote_values = adv_preds
        else:
            raise ValueError(f"Invalid vote type: {vote_type}")

        # Voting-based action selection
        # Min-max normalization over the action ensemble (num_agents, DT ensemble, action ensemble)
        val_min = vote_values.min(dim=2, keepdim=True).values
        val_max = vote_values.max(dim=2, keepdim=True).values
        vote_values = (vote_values - val_min) / (val_max - val_min).clamp_min(1e-6)
        # Average over the DT ensemble
        vote_values = vote_values.mean(dim=1)

        # Take out the best action from the ensemble based on the argmax index of vote-value (num_agents, ensemble_size)
        best_action_idx = vote_values.argmax(dim=1)
        # Select best action in RAW space using chosen indices
        best_action_raw = action_ensembles_raw[range(num_agents), best_action_idx.detach().cpu().numpy()]

        # If advantage voting, apply a fallback mechanism
        if vote_type == "adv":
            adv_mean = adv_preds.mean(dim=1)
            best_adv = adv_mean.gather(1, best_action_idx[:, None]).squeeze(1)
            backbone_adv = adv_mean[:, 0]
            margin = 0.001
            fallback = (best_adv < margin) | (best_adv < backbone_adv)
            best_action_raw = np.where(
                fallback.detach().cpu().numpy(),
                backbone_actions_raw.squeeze(-1),
                best_action_raw,
            )

        # Return RAW action
        assert best_action_raw.shape == (num_agents,)
        return best_action_raw
    
    def on_fit_start(self) -> None:
        self.scalers = self.trainer.datamodule.scalers
    
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.scalers is None:
            raise ValueError("Scalers missing: fit or assign scalers before training to enable checkpoint saving.")
        checkpoint["scalers"] = self.scalers
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.scalers = checkpoint["scalers"]
    
    
class GASBiddingAgent(BaseBiddingAgent):
    def __init__(
        self,
        adv_indicies: list[int] | np.ndarray,
        budget: list[float] | np.ndarray,
        cpa: list[float] | np.ndarray,
        category: list[int] | np.ndarray,
        name: str = "GAS",
        model_dir: str | Path | None = None,
        checkpoint_file: str = "best.ckpt",
        state_dim: int | None = None,
        model: GASModel | None = None,
        target_rtg: float = 4.0,
        backbone_model_dir: str | None = None,
        backbone_checkpoint_file: str = "best.ckpt",
        backbone_model: L.LightningModule | None = None,
    ) -> None:
        super().__init__(adv_indicies, budget, cpa, category, name)

        if not ((model_dir is None) ^ (model is None)):
            raise ValueError("Provide exactly one of model_dir or model.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is not None:
            self.model = model
        else:
            self.model = GASModel.load_from_checkpoint(
                Path(model_dir) / checkpoint_file,
                backbone_model_dir=backbone_model_dir,
                backbone_checkpoint_file=backbone_checkpoint_file,
                map_location=device,
                state_dim=state_dim,
                action_dim=1,
            )
        self.model.eval()
        
        self.target_rtg = target_rtg
        
        if not ((backbone_model_dir is None) ^ (backbone_model is None)):
            raise ValueError("Provide exactly one of backbone_model_dir or backbone_model.")
        
        if backbone_model is not None:
            self.backbone_model = backbone_model
        else:
            self.backbone_model = DTModel.load_from_checkpoint(
                Path(backbone_model_dir) / backbone_checkpoint_file,
                map_location=device,
                state_dim=state_dim,
                action_dim=1,
            )
        self.backbone_model.eval()
        self.backbone_model.freeze()
        
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

        # RTG update consistent with DT/GAVE: apply CPA penalty to conversions
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

        # Base actions from backbone DT
        backbone_actions = self.backbone_model.predict(
            states=seq_states,
            actions=seq_actions,
            rtgs=seq_rtgs,
            time_steps=seq_timesteps,
            mask=seq_mask,
            curr_pointer=curr_pointer,
        )

        # GAS action using Q-guided selection
        alpha = self.model.predict(
            states=seq_states,
            actions=seq_actions,
            rtgs=seq_rtgs * 0.0,  # NOTE: not used rtgs for GAS
            time_steps=seq_timesteps,
            mask=seq_mask,
            curr_pointer=curr_pointer,
            backbone_actions=backbone_actions,
        )

        assert alpha.ndim == 1, f"Expected alpha to be 1D, got {alpha.ndim}D."
        self.last_action[:] = alpha[:, None]

        alpha = alpha[None, :].clip(min=0.0)
        return alpha * pValues
