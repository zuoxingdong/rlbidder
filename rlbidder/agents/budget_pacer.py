import numpy as np

from rlbidder.agents.base import BaseBiddingAgent
from rlbidder.constants import NUM_TICKS
from rlbidder.evaluation.history import StepwiseAuctionHistory


class BudgetPacerBiddingAgent(BaseBiddingAgent):
    """
    Budget pacing agent for batched advertisers.
    Uses CPA as the base action for each advertiser.
    """
    def __init__(
        self,
        adv_indicies: list[int] | np.ndarray,
        budget: list[float] | np.ndarray,
        cpa: list[float] | np.ndarray,
        category: list[int] | np.ndarray,
        expected_spend_ratio: np.ndarray | list[float] | None = None,
        name: str = "BudgetPacer",
        low_spend_threshold: float = 0.7,
        high_spend_threshold: float = 1.1,
        increase_factor: float = 1.2,
        decrease_factor: float = 0.7,
    ) -> None:
        super().__init__(adv_indicies, budget, cpa, category, name)
        
        # Store configurable parameters
        self.low_spend_threshold = low_spend_threshold
        self.high_spend_threshold = high_spend_threshold
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        
        # expected_spend_ratio shape: (num_timesteps, num_batched_agents)
        num_batched_agents = len(adv_indicies)
        if expected_spend_ratio is None:
            self.expected_spend_ratio = np.ones((NUM_TICKS, num_batched_agents), dtype=float)
        else:
            arr = np.array(expected_spend_ratio, dtype=float)
            if arr.ndim == 1:
                # Broadcast to (num_timesteps, num_batched_agents)
                self.expected_spend_ratio = np.tile(arr[:, None], (1, num_batched_agents))
            else:
                self.expected_spend_ratio = arr
                
        # track alpha for adjustment
        self.alpha = None

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
        self.alpha = None

    def bidding(
        self,
        timeStepIndex: int,
        pValues: np.ndarray,
        pValueSigmas: np.ndarray,
        auction_history: StepwiseAuctionHistory,
    ) -> np.ndarray:
        if timeStepIndex == 0:  # first step with CPA
            self.alpha = self.cpa.astype(float).copy()
        else:  # Adjust alpha based on previous spend
            prev_tick_cost = auction_history.sum_costs[timeStepIndex - 1, self.adv_indicies]

            ratio_to_go = self.expected_spend_ratio[timeStepIndex:, :].sum(axis=0)
            prev_ratio = self.expected_spend_ratio[timeStepIndex - 1, :]
            scaled_prev_tick_cost = prev_tick_cost / prev_ratio
            expected_spend = scaled_prev_tick_cost * ratio_to_go
            normalized_spend = expected_spend / self.remaining_budget

            remaining_steps = self.expected_spend_ratio.shape[0] - timeStepIndex
            projected_spend = (prev_tick_cost * remaining_steps) / self.remaining_budget
            
            # Adjust alpha based on spend ratios
            # - Increase alpha if normalized spend is low
            # - Decrease alpha if projected spend is high
            # Use numpy.select for vectorized alpha adjustment
            conds = [normalized_spend < self.low_spend_threshold, projected_spend > self.high_spend_threshold]
            choices = [self.alpha * self.increase_factor, self.alpha * self.decrease_factor]
            self.alpha = np.select(conds, choices, default=self.alpha)
        
        alpha = self.alpha[None, :]
        return alpha * pValues
