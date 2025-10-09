import numpy as np

from rlbidder.agents.base import BaseBiddingAgent
from rlbidder.evaluation.history import StepwiseAuctionHistory


class FixedCPABiddingAgent(BaseBiddingAgent):
    """
    Bidding agent that uses a fixed CPA multiplier for bids.
    """
    def __init__(
        self,
        adv_indicies: list[int] | np.ndarray,
        budget: list[float] | np.ndarray,
        cpa: list[float] | np.ndarray,
        category: list[int] | np.ndarray,
        name: str = "FixedCPA",
    ) -> None:
        super().__init__(adv_indicies, budget, cpa, category, name)

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
        auction_history: StepwiseAuctionHistory,
    ) -> np.ndarray:
        alpha = self.cpa[None, :]
        return alpha * pValues
