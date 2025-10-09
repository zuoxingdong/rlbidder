from typing import Self

import numpy as np
import polars as pl
from tqdm import tqdm


class StepwiseAuctionHistory(object):
    """
    Tracks and aggregates auction metrics for each time step (tick) in a stepwise auction process.
    Stores both average and sum statistics for each advertiser at each step.
    """

    def __init__(self, num_steps: int, num_advertisers: int) -> None:
        """
        Initialize the StepwiseAuctionHistory.

        Args:
            num_steps (int): Number of time steps (ticks) in the auction.
            num_advertisers (int): Number of advertisers participating.
        """
        self.num_steps = num_steps
        self.num_advertisers = num_advertisers

        # Initialize history arrays
        self.volumes = np.zeros((num_steps,))
        self.pValues = np.zeros((num_steps, num_advertisers))

        # Average metrics per tick
        self.avg_bids = np.zeros((num_steps, num_advertisers))
        self.avg_alphas = np.zeros((num_steps, num_advertisers))
        self.avg_leastWinningCosts = np.zeros((num_steps, num_advertisers))
        self.avg_win_status = np.zeros((num_steps, num_advertisers))
        self.avg_costs = np.zeros((num_steps, num_advertisers))
        self.avg_conversions = np.zeros((num_steps, num_advertisers))

        # Standard deviation metrics per tick
        self.std_bids = np.zeros((num_steps, num_advertisers))
        self.std_alphas = np.zeros((num_steps, num_advertisers))
        self.std_leastWinningCosts = np.zeros((num_steps, num_advertisers))

        # Sum metrics per tick
        self.sum_bids = np.zeros((num_steps, num_advertisers))
        self.sum_leastWinningCosts = np.zeros((num_steps, num_advertisers))
        self.sum_win_status = np.zeros((num_steps, num_advertisers))
        self.sum_costs = np.zeros((num_steps, num_advertisers))
        self.sum_conversions = np.zeros((num_steps, num_advertisers))
        
        # Optional meta information
        self.agent_names = np.empty(num_advertisers, dtype=np.dtypes.StringDType)
        self.agent_budgets = np.zeros(num_advertisers)
        self.agent_cpas = np.zeros(num_advertisers)

        # Track how many steps have been added
        self.steps_added = 0

    @classmethod
    def from_lazy_frame(cls, lf: pl.LazyFrame) -> Self:
        """
        Create a StepwiseAuctionHistory from a polars lazy frame.

        Args:
            lf: Polars lazy frame containing auction data.
    
        Returns:
            StepwiseAuctionHistory: Populated history object.
        """

        # Process the lazy frame
        lf = lf.collect()

        # Determine dimensions
        delivery_period_index = lf.select("deliveryPeriodIndex").unique().item()
        num_steps = lf.select("timeStepIndex").n_unique()
        num_advertisers = lf.select("advertiserNumber").n_unique()

        # Create history instance
        history = cls(num_steps=num_steps, num_advertisers=num_advertisers)

        # Load data for each time step
        for time_step in tqdm(range(num_steps), desc="Loading auction history"):
            step_data = (
                lf
                .filter(
                    (pl.col("deliveryPeriodIndex") == delivery_period_index)
                    & (pl.col("timeStepIndex") == time_step)
                )
                .select(
                    pl.col("pValue"),
                    pl.col("bid"),
                    pl.col("leastWinningCost"),
                    pl.col("win_status"),
                    pl.col("cost"),
                    pl.col("conversion"),
                )
            )
            
            (
                pValues,
                bids,
                least_winning_costs,
                win_status,
                costs,
                conversions,
            ) = step_data.to_numpy().T

            # Stack columns to matrices shaped (num_impressions, num_advertisers)
            pValues_mat = np.stack(pValues, axis=1)
            bids_mat = np.stack(bids, axis=1)
            least_winning_costs_mat = np.stack(least_winning_costs, axis=1)
            win_status_mat = np.stack(win_status, axis=1)
            costs_mat = np.stack(costs, axis=1)
            conversions_mat = np.stack(conversions, axis=1)

            # Derive alphas safely: alpha = bid / pValue
            with np.errstate(divide='ignore', invalid='ignore'):
                alphas_mat = np.divide(
                    bids_mat,
                    pValues_mat,
                    out=np.zeros_like(bids_mat, dtype=float),
                    where=(pValues_mat != 0),
                )

            history.push(
                pValues=pValues_mat,
                bids=bids_mat,
                least_winning_costs=least_winning_costs_mat,
                win_status=win_status_mat,
                costs=costs_mat,
                conversions=conversions_mat,
                alphas=alphas_mat,
            )

        return history

    def set_agent_meta_info(self, agents: list[object]) -> None:
        """
        Set the meta information for the agents.
        """
        for agent in agents:
            self.agent_names[agent.adv_indicies] = agent.name
            self.agent_budgets[agent.adv_indicies] = agent.budget
            self.agent_cpas[agent.adv_indicies] = agent.cpa

    def push(
        self,
        pValues: np.ndarray,
        bids: np.ndarray,
        least_winning_costs: np.ndarray,
        win_status: np.ndarray,
        costs: np.ndarray,
        conversions: np.ndarray,
        alphas: np.ndarray | None = None,
    ) -> None:
        """
        Push a new step's data into the history.

        Args:
            pValues (np.ndarray): Array of p-values for this step (shape: [num_impressions, num_advertisers]).
            bids (np.ndarray): Array of bids for this step (shape: [num_impressions, num_advertisers]).
            least_winning_costs (np.ndarray): Array of least winning costs (shape: [num_impressions, num_advertisers]).
            win_status (np.ndarray): Array of win statuses (shape: [num_impressions, num_advertisers]).
            costs (np.ndarray): Array of costs (shape: [num_impressions, num_advertisers]).
            conversions (np.ndarray): Array of conversions (shape: [num_impressions, num_advertisers]).
            alphas (np.ndarray | None): Array of raw actions (shape: [num_impressions, num_advertisers]). If None,
                it will be derived as bids / pValues with safe division.
        Raises:
            IndexError: If all steps have already been added.
        """
        if self.steps_added >= self.num_steps:
            raise IndexError("All steps have already been added.")

        step_index = self.steps_added

        # Derive alphas if not provided
        if alphas is None:
            with np.errstate(divide='ignore', invalid='ignore'):
                alphas = np.divide(
                    bids,
                    pValues,
                    out=np.zeros_like(bids, dtype=float),
                    where=(pValues != 0),
                )

        # Update history arrays
        self.volumes[step_index] = pValues.shape[0]
        self.pValues[step_index] = np.mean(pValues, axis=0)

        # Update average metrics
        self.avg_bids[step_index] = np.mean(bids, axis=0)
        self.avg_alphas[step_index] = np.mean(alphas, axis=0)
        self.avg_leastWinningCosts[step_index] = np.mean(least_winning_costs, axis=0)
        self.avg_win_status[step_index] = np.mean(win_status, axis=0)
        self.avg_costs[step_index] = np.mean(costs, axis=0)
        self.avg_conversions[step_index] = np.mean(conversions, axis=0)
        
        # Update standard deviation metrics
        self.std_bids[step_index] = np.std(bids, axis=0)
        self.std_alphas[step_index] = np.std(alphas, axis=0)
        self.std_leastWinningCosts[step_index] = np.std(least_winning_costs, axis=0)
        
        # Update sum metrics
        self.sum_bids[step_index] = np.sum(bids, axis=0)
        self.sum_leastWinningCosts[step_index] = np.sum(least_winning_costs, axis=0)
        self.sum_win_status[step_index] = np.sum(win_status, axis=0)
        self.sum_costs[step_index] = np.sum(costs, axis=0)
        self.sum_conversions[step_index] = np.sum(conversions, axis=0)

        # Increment steps added
        self.steps_added += 1

    def get_state_features(self) -> np.ndarray:
        """
        Compute and return state features for all advertisers.

        Returns:
            np.ndarray: State features of shape (num_advertisers, 13).

        Raises:
            ValueError: If no steps have been added to the history.
        """
        state = np.zeros((self.num_advertisers, 13))

        if self.steps_added > 0:
            # Create a slice object for all proceeding steps
            all_steps = slice(0, self.steps_added)
            # Create a slice object for the last 3 steps
            last3 = slice(max(0, self.steps_added - 3), self.steps_added)
            
            # avg_bid_all
            state[:, 0] = np.mean(self.avg_bids[all_steps], axis=0)
            # avg_bid_last_3
            state[:, 1] = np.mean(self.avg_bids[last3], axis=0)
            # avg_leastWinningCost_all
            state[:, 2] = np.mean(self.avg_leastWinningCosts[all_steps], axis=0)
            # avg_pValue_all
            state[:, 3] = np.mean(self.pValues[all_steps], axis=0)
            # avg_conversionAction_all
            state[:, 4] = np.mean(self.avg_conversions[all_steps], axis=0)
            # avg_xi_all
            state[:, 5] = np.mean(self.avg_win_status[all_steps], axis=0)
            # avg_leastWinningCost_last_3
            state[:, 6] = np.mean(self.avg_leastWinningCosts[last3], axis=0)
            # avg_pValue_last_3
            state[:, 7] = np.mean(self.pValues[last3], axis=0)
            # avg_conversionAction_last_3
            state[:, 8] = np.mean(self.avg_conversions[last3], axis=0)
            # avg_xi_last_3
            state[:, 9] = np.mean(self.avg_win_status[last3], axis=0)
            # last_3_timeStepIndexs_volume
            state[:, 10] = np.sum(self.volumes[last3], axis=0)
            # historical_volume
            state[:, 11] = np.sum(self.volumes[all_steps], axis=0)
            # CPA compliance ratio = CPAConstraint / ecpa, with safe division
            ecpa = np.nan_to_num(self.compute_realized_cpa(), nan=0.0, posinf=0.0, neginf=0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                state[:, 12] = np.divide(
                    self.agent_cpas, 
                    ecpa, 
                    out=np.zeros_like(ecpa, dtype=float), 
                    where=(ecpa > 0)
                )
        return state

    def get_total_conversions(self) -> np.ndarray:
        """
        Compute and return the total conversions for each advertiser.

        Returns:
            np.ndarray: Total conversions of shape (num_advertisers,).
        """
        all_steps = slice(0, self.steps_added)
        return np.sum(self.sum_conversions[all_steps], axis=0)

    def get_total_costs(self) -> np.ndarray:
        """
        Compute and return the total costs for each advertiser.

        Returns:
            np.ndarray: Total costs of shape (num_advertisers,).
        """
        all_steps = slice(0, self.steps_added)
        return np.sum(self.sum_costs[all_steps], axis=0)

    def compute_realized_cpa(self) -> np.ndarray:
        """
        Compute and return the realized CPA (total cost / total conversions) for each advertiser.

        Returns:
            np.ndarray: Realized CPA of shape (num_advertisers,).
            If total conversions is 0 for an advertiser, returns np.nan for that advertiser.
        """
        total_costs = self.get_total_costs()
        total_conversions = self.get_total_conversions()
        with np.errstate(divide='ignore', invalid='ignore'):
            realized_cpa = np.where(total_conversions > 0, total_costs.astype(float) / total_conversions, np.nan)
        return realized_cpa

    def compute_cpa_penalty_series(
        self,
        adv_indices: np.ndarray | list[int] | slice,
        penalty_power: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute per-step CPA compliance penalties and per-step conversions for the specified advertisers.

        Semantics:
        - conversions_step = self.sum_conversions per step
        - eCPA(t) = costs_cum(t) / conversions_cum(t)
        - compliance_ratio(t) = agent_cpa / eCPA(t), defaulting to 1 where conversions_cum == 0
        - penalty(t) = clip(compliance_ratio(t), 0, 1) ** penalty_power

        Args:
            adv_indices: Indices selecting advertisers (list/ndarray/slice).
            penalty_power: Exponent applied to the clipped compliance ratio.

        Returns:
            penalties: shape [T, A]
            conversions_step: shape [T, A]

        Raises:
            ValueError: If no steps have been added to the history.
        """
        T = len(self)
        if T == 0:
            raise ValueError("No steps have been added; cannot compute CPA penalties.")

        conversions_step = self.sum_conversions[:T, adv_indices]
        costs_cum = self.sum_costs[:T, adv_indices].cumsum(axis=0)
        conversions_cum = conversions_step.cumsum(axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            cpa_compliance_ratios = np.where(
                conversions_cum > 0,
                self.agent_cpas[adv_indices] / (costs_cum / conversions_cum),
                1.0,
            )

        penalties = np.clip(cpa_compliance_ratios, 0.0, 1.0) ** penalty_power
        return penalties, conversions_step

    def __len__(self) -> int:
        """
        Returns:
            int: Number of steps added to the history.
        """
        return self.steps_added
    
    def __repr__(self) -> str:
        """
        Returns:
            str: String representation of the StepwiseAuctionHistory object.
        """
        return (
            f"StepwiseAuctionHistory("
            f"steps_added={self.steps_added}, "
            f"num_advertisers={self.num_advertisers}, "
            f")"
        )
