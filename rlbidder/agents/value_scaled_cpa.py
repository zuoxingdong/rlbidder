import numpy as np

from rlbidder.agents.base import BaseBiddingAgent
from rlbidder.evaluation.history import StepwiseAuctionHistory


class ValueScaledCPABiddingAgent(BaseBiddingAgent):
    """
    Bidding agent using a value-scaled CPA strategy.
    CPA is scaled by mean-normalized pValues.
    """
    def __init__(
        self,
        adv_indicies: list[int] | np.ndarray,
        budget: list[float] | np.ndarray,
        cpa: list[float] | np.ndarray,
        category: list[int] | np.ndarray,
        name: str = "ValueScaledCPA",
        clip_weights: tuple[float, float] | None = (0.51, 1.23),
    ) -> None:
        super().__init__(adv_indicies, budget, cpa, category, name)
        self.clip_weights = clip_weights

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

    def _validate_shapes(self, pValues: np.ndarray) -> None:
        # shape validation
        if pValues.ndim != 2:
            raise ValueError(f"pValues must be 2D (batch, advertisers); got {pValues.shape=}")
        if pValues.shape[1] != self.cpa.shape[0]:
            raise ValueError(f"{pValues.shape[1]=} must match number of advertisers ({self.cpa.shape[0]=})")

    def bidding(
        self,
        timeStepIndex: int,
        pValues: np.ndarray,
        pValueSigmas: np.ndarray,
        auction_history: StepwiseAuctionHistory,
    ) -> np.ndarray:
        self._validate_shapes(pValues)

        denom = pValues.mean(axis=0, keepdims=True).clip(min=1e-8)
        weights = pValues / denom
        if self.clip_weights is not None:
            lo, hi = self.clip_weights
            weights = weights.clip(lo, hi)

        # Alpha: CPA weighted by value distribution
        alpha = self.cpa[None, :] * weights
        return alpha * pValues
