import numpy as np

from rlbidder.agents.base import BaseBiddingAgent


class ConstantBidAgent(BaseBiddingAgent):
    """
    Bidding agent that bids a constant amount for all auctions.
    """
    def __init__(
        self,
        adv_indicies: list[int] | np.ndarray,
        budget: list[float] | np.ndarray,
        cpa: list[float] | np.ndarray,
        category: list[int] | np.ndarray,
        name: str = "ConstantBid",
        bid_fraction: float = 0.003,  # Fraction of CPA to bid
    ) -> None:
        super().__init__(adv_indicies, budget, cpa, category, name)
        self.bid_fraction = bid_fraction

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
        # Bid a constant fraction of CPA for all auctions
        constant_bid = self.cpa * self.bid_fraction
        return np.full_like(pValues, constant_bid[None, :])
