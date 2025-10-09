from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from rlbidder.evaluation.history import StepwiseAuctionHistory


class BaseBiddingAgent(ABC):
    """Base interface for all bidding agents."""

    def __init__(
        self,
        adv_indicies: Sequence[int],
        budget: Sequence[float],
        cpa: Sequence[float],
        category: Sequence[int],
        name: str = "Base",
    ) -> None:
        self.adv_indicies = np.array(adv_indicies, dtype=int)
        self.budget = np.array(budget, dtype=float)
        self.remaining_budget = np.array(budget, dtype=float)
        self.cpa = np.array(cpa, dtype=float)
        self.category = np.array(category, dtype=int)
        self.name = name
        self.num_advertisers = len(adv_indicies)

        # Store original budget and cpa to support reset with ratios
        self._original_budget = self.budget.copy()
        self._original_cpa = self.cpa.copy()

    @abstractmethod
    def reset(
        self,
        budget: Sequence[float] | None = None,
        cpa: Sequence[float] | None = None,
        budget_ratio: float | None = None,
        cpa_ratio: float | None = None,
    ) -> None:
        """
        Reset the agent's state for a new episode or simulation.
        Optionally update budget/cpa or apply ratios.
        """
        # Check for conflicting arguments
        if budget is not None and budget_ratio is not None:
            raise ValueError("Provide either 'budget' or 'budget_ratio', not both.")
        if cpa is not None and cpa_ratio is not None:
            raise ValueError("Provide either 'cpa' or 'cpa_ratio', not both.")

        # Update budget and cpa if new values are provided
        if budget is not None:
            self.budget = np.array(budget, dtype=float)
        if cpa is not None:
            self.cpa = np.array(cpa, dtype=float)

        # Apply ratios if provided
        if budget_ratio is not None:
            self.budget = self._original_budget.copy() * budget_ratio
        if cpa_ratio is not None:
            self.cpa = self._original_cpa.copy() * cpa_ratio
        
        # If nothing is provided, reset to original values
        if budget is None and cpa is None and budget_ratio is None and cpa_ratio is None:
            self.budget = self._original_budget.copy()
            self.cpa = self._original_cpa.copy()
        
        self.remaining_budget = self.budget.copy()

    @abstractmethod
    def bidding(
        self,
        timeStepIndex: int,
        pValues: np.ndarray,
        pValueSigmas: np.ndarray,
        auction_history: StepwiseAuctionHistory,
    ) -> np.ndarray:
        """
        Decide on a bid amount for the current auction.
        """
        raise NotImplementedError

    def spend(self, amount: Sequence[float]) -> None:
        """
        Deduct amount from remaining budget, ensuring it does not go negative.
        """
        self.remaining_budget = (self.remaining_budget - np.array(amount, dtype=float)).clip(min=0.0)

    def __repr__(self) -> str:
        # Limit floating point precision for display
        def fmt(arr: np.ndarray) -> list[float]:
            return np.round(arr.astype(float), 4).tolist()
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"budget={fmt(self.budget)}, "
            f"remaining_budget={fmt(self.remaining_budget)}, "
            f"cpa={fmt(self.cpa)}, "
            f"category={self.category.tolist()}, "
            f"adv_indicies={self.adv_indicies.tolist()}"
            f")"
        )
