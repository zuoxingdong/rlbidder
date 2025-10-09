import numpy as np

from rlbidder.agents.base import BaseBiddingAgent
from rlbidder.evaluation.history import StepwiseAuctionHistory


class StochasticCPABiddingAgent(BaseBiddingAgent):
    """
    Bidding agent that samples CPA from a normal distribution for each bid.
    The mean is the fixed CPA, and std is a configurable ratio of CPA.
    Uses numpy 2.0 random Generator with a configurable seed.
    """
    def __init__(
        self,
        adv_indicies: list[int] | np.ndarray,
        budget: list[float] | np.ndarray,
        cpa: list[float] | np.ndarray,
        category: list[int] | np.ndarray,
        cpa_std_ratio: float = 0.01,
        seed: int | None = None,
        name: str = "StochasticCPA",
    ) -> None:
        super().__init__(adv_indicies, budget, cpa, category, name)
        self.cpa_std_ratio = cpa_std_ratio
        self.seed = seed
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def reset(
        self,
        budget: list[float] | np.ndarray | None = None,
        cpa: list[float] | np.ndarray | None = None,
        budget_ratio: float | None = None,
        cpa_ratio: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().reset(
            budget=budget, 
            cpa=cpa, 
            budget_ratio=budget_ratio, 
            cpa_ratio=cpa_ratio,
        )
        # Optionally reseed the generator on reset
        if seed is not None:
            self.seed = seed
            self.rng = np.random.Generator(np.random.PCG64(seed))

    def bidding(
        self,
        timeStepIndex: int,
        pValues: np.ndarray,
        pValueSigmas: np.ndarray,
        auction_history: StepwiseAuctionHistory,
    ) -> np.ndarray:
        # Sample CPA for each advertiser from N(mean=CPA, std=CPA * ratio)
        std = self.cpa * self.cpa_std_ratio
        sampled_cpa = self.rng.normal(loc=self.cpa, scale=std)
        sampled_cpa = np.clip(sampled_cpa, 1e-8, None)  # avoid negative or zero CPA
        alpha = sampled_cpa[None, :]
        return alpha * pValues
