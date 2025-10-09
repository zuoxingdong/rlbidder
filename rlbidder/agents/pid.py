import numpy as np

from rlbidder.agents.base import BaseBiddingAgent
from rlbidder.constants import NUM_TICKS
from rlbidder.evaluation.history import StepwiseAuctionHistory


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, shape: tuple[int, ...]):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.shape = shape
        self.integral = np.zeros(shape, dtype=float)
        self.last_error = np.zeros(shape, dtype=float)

    def reset(self) -> None:
        self.integral = np.zeros(self.shape, dtype=float)
        self.last_error = np.zeros(self.shape, dtype=float)

    def update(self, error: np.ndarray) -> np.ndarray:
        error = np.asarray(error)
        if error.shape != self.shape:
            raise ValueError(f"Error shape {error.shape} does not match expected {self.shape}")
        
        self.integral += error
        derivative = error - self.last_error
        self.last_error = error
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        return control


class PIDBudgetPacerBiddingAgent(BaseBiddingAgent):
    """
    Budget pacing agent using a PID controller for alpha adjustment.
    """
    def __init__(
        self,
        adv_indicies: list[int] | np.ndarray,
        budget: list[float] | np.ndarray,
        cpa: list[float] | np.ndarray,
        category: list[int] | np.ndarray,
        expected_spend_ratio: np.ndarray | list[float] | None = None,
        pid_params: tuple[float, float, float] = (0.01, 2.568e-06, 0.000128),
        name: str = "PIDBudgetPacer",
    ) -> None:
        super().__init__(adv_indicies, budget, cpa, category, name)
        
        num_batched_agents = len(adv_indicies)
        
        # expected_spend_ratio shape: (num_timesteps, num_batched_agents)
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
        
        self.pid = PIDController(*pid_params, shape=(num_batched_agents,))

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
        self.pid.reset()

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

            # Use numpy.select for vectorized error assignment
            conds = [normalized_spend < 0.7, projected_spend > 1.1]
            # Positive error to increase alpha
            # Negative error to decrease alpha
            # Default to 0.0 if neither condition is met
            choices = [1.0, -1.0]
            error = np.select(conds, choices, default=0.0)
            
            # Update PID controller
            delta = self.pid.update(error)
            delta = np.clip(delta, -1.5, 1.5)
            self.alpha = np.maximum(0.01, self.alpha * np.exp(delta))

        alpha = self.alpha[None, :]
        return alpha * pValues


class PIDCPABiddingAgent(BaseBiddingAgent):
    """
    PID controller agent to keep realized CPA close to target CPA.
    """
    def __init__(
        self,
        adv_indicies: list[int] | np.ndarray,
        budget: list[float] | np.ndarray,
        cpa: list[float] | np.ndarray,
        category: list[int] | np.ndarray,
        pid_params: tuple[float, float, float] = (0.004, 8.556e-06, 0.00373),
        name: str = "PIDCPA",
    ) -> None:
        super().__init__(adv_indicies, budget, cpa, category, name)
        
        num_batched_agents = len(adv_indicies)
        
        self.cumulative_cost = np.zeros(num_batched_agents, dtype=float)
        self.cumulative_conversions = np.zeros(num_batched_agents, dtype=float)
        
        # track alpha for adjustment
        self.alpha = None

        self.pid = PIDController(*pid_params, shape=(num_batched_agents,))

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
        num_batched_agents = len(self.adv_indicies)
        self.cumulative_cost = np.zeros(num_batched_agents, dtype=float)
        self.cumulative_conversions = np.zeros(num_batched_agents, dtype=float)
        self.alpha = None
        self.pid.reset()

    def bidding(
        self,
        timeStepIndex: int,
        pValues: np.ndarray,
        pValueSigmas: np.ndarray,
        auction_history: StepwiseAuctionHistory,
    ) -> np.ndarray:
        if timeStepIndex == 0:  # first step with CPA
            self.alpha = self.cpa.astype(float).copy()
        else:  # # Adjust alpha based on estimated CPA
            prev_tick_cost = auction_history.sum_costs[timeStepIndex - 1, self.adv_indicies]
            prev_tick_conversions = auction_history.sum_conversions[timeStepIndex - 1, self.adv_indicies]
            self.cumulative_cost += prev_tick_cost
            self.cumulative_conversions += prev_tick_conversions
            
            
            with np.errstate(divide='ignore', invalid='ignore'):
                realized_cpa = np.where(
                    self.cumulative_conversions > 0, 
                    self.cumulative_cost / self.cumulative_conversions, 
                    np.nan,
                )
            
            
            # Use a reference CPA for error calculation (could be original or running average)
            error = np.nan_to_num(
                ((self.cpa - realized_cpa) / self.cpa),
                nan=0.0,
            )
            error *= prev_tick_conversions
            
            # # Compare realized_cpa and self.cpa to assign error term
            # conds = [realized_cpa > self.cpa * 1.1, realized_cpa < self.cpa * 0.9]
            # choices = [-1.0, 1.0]  # Decrease alpha if CPA too high, increase if too low
            # error = np.select(conds, choices, default=0.0)
            
            # Update PID controller
            delta = self.pid.update(error)
            delta = np.clip(delta, -2, 2)
            # Only adjust where cumulative_conversions > 0, else set delta to zero (no adjustment)
            # Avoid division by zero and suppress warnings by using np.divide with where argument
            delta = np.divide(
                delta, 
                self.cumulative_conversions, 
                out=np.zeros_like(delta), 
                where=self.cumulative_conversions > 0
            )
            
            self.alpha = np.maximum(0.01, self.alpha * np.exp(delta))

        alpha = self.alpha[None, :]
        return alpha * pValues
