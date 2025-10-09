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
from rlbidder.constants import DEFAULT_SEED, NUM_TICKS
from rlbidder.evaluation import summarize_agent_scores
from rlbidder.models.networks import init_trunc_normal
from rlbidder.models.utils import extract_state_features
from rlbidder.utils import log_distribution, regression_report


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_sizes: list[int] = [512, 512, 512],
    ) -> None:
        super(Actor, self).__init__()

        all_sizes = [state_dim] + hidden_sizes
        layers = []
        # Build the network layers
        for in_features, out_features in zip(all_sizes[:-1], all_sizes[1:]):
            layers += [
                nn.Linear(in_features, out_features),
                nn.ReLU()
            ]
        # Add final output layer
        layers += [
            nn.Linear(all_sizes[-1], 1),
            # nn.Tanh(),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class BCModel(L.LightningModule):
    def __init__(
        self,
        state_dim: int,
        lr: float = 1e-4,
        hidden_sizes: list[int] = [512, 512, 512],
        val_metric: str = "mean_score",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.actor = Actor(state_dim, hidden_sizes=hidden_sizes)
        self.actor.apply(lambda m: init_trunc_normal(m, std=1e-2))
        
        # Placeholder for scalers
        # These will be set during training or loaded from checkpoint
        self.scalers = None

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.actor(states)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=0,
            fused=True,
        )

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        states, actions, rewards, next_states, dones, cpa_compliance_ratios = batch

        preds = self(states)
        loss = F.mse_loss(preds, actions, reduction="mean")
        
        # lightweight logs
        log_args = dict(on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(
            {
                "train/loss": loss,
                "train/target_actions/mean": actions.mean(),
                "train/pred_actions/mean": preds.mean(),
            },
            **log_args,
        )
        # heavy logs only every log_every_n_steps
        if (
            self.logger is not None
            and self.trainer.is_global_zero
            and (self.global_step % self.trainer.log_every_n_steps == 0)
            and (self.scalers is not None)
        ):
            with torch.no_grad():
                target_actions_raw = self.scalers["action_scaler"].inverse_transform(actions.detach().cpu().numpy())
                pred_actions_raw = self.scalers["action_scaler"].inverse_transform(preds.detach().cpu().numpy())

                self.log_dict(
                    regression_report(
                        y_true=target_actions_raw,
                        y_pred=pred_actions_raw,
                        prefix="train/raw_action/",
                    ),
                    **log_args,
                )

                log_distribution(
                    logger=self.logger,
                    data=pred_actions_raw,
                    name="pred_actions_raw",
                    step=self.global_step,
                    context={"subset": "train"},
                )

                # Log some statistics
                self.log_dict(
                    {
                        "train/target_actions_raw/mean": target_actions_raw.mean(),
                        "train/target_actions_raw/std": target_actions_raw.std(),
                        "train/pred_actions_raw/mean": pred_actions_raw.mean(),
                        "train/pred_actions_raw/std": pred_actions_raw.std(),
                    },
                    **log_args,
                )

        return loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        # No-op: existence of this hook ensures Lightning runs the validation loop
        # so that on_validation_epoch_end gets called at scheduled steps.
        return None

    def on_validation_epoch_end(self) -> None:
        # Run expensive evaluation on rank 0, broadcast results to all ranks, then log everywhere.
        mean_returns = 0.0
        mean_score = 0.0
        mean_budget_spent_ratio = 0.0

        if self.trainer.is_global_zero:
            agent_config = (
                BCBiddingAgent,
                "BC",
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
                    .filter(pl.col("agent_name") == "BC")
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
        action = self.actor(state).detach().cpu().numpy()
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


class BCBiddingAgent(BaseBiddingAgent):
    """
    Behavioral Cloning (BC) Agent
    """
    def __init__(
        self,
        adv_indicies: list[int] | np.ndarray,
        budget: list[float] | np.ndarray,
        cpa: list[float] | np.ndarray,
        category: list[int] | np.ndarray,
        name: str = "BC",
        model_dir: str | Path | None = None,
        checkpoint_file: str = "best.ckpt",
        state_dim: int | None = None,
        model: BCModel | None = None,
    ) -> None:
        super().__init__(adv_indicies, budget, cpa, category, name)

        if not ((model_dir is None) ^ (model is None)):
            raise ValueError("Provide exactly one of model_dir or model.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is not None:
            self.model = model
        else:
            self.model = BCModel.load_from_checkpoint(
                Path(model_dir) / checkpoint_file,
                map_location=device,
                state_dim=state_dim,
            )
        self.model.eval()

    def reset(self, budget=None, cpa=None, budget_ratio=None, cpa_ratio=None) -> None:
        super().reset(
            budget=budget, 
            cpa=cpa, 
            budget_ratio=budget_ratio, 
            cpa_ratio=cpa_ratio,
        )

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
        
        alpha = self.model.predict(states)
        alpha = alpha[None, :].clip(min=0.0)
        return alpha * pValues
