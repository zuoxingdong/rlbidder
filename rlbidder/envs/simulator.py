import numpy as np

from rlbidder.constants import (
    DEFAULT_SEED,
    MIN_REMAINING_BUDGET,
    NUM_SLOTS_DEFAULT,
    RESERVE_PV_PRICE,
    SLOT_EXPOSURE_COEFFS,
)
from rlbidder.envs.sampler import ValueSampler


class AuctionSimulator:
    def __init__(
        self,
        num_advertisers: int,
        min_remaining_budget: float = MIN_REMAINING_BUDGET,
        seed: int = DEFAULT_SEED,
        num_slots: int | None = None,
        slot_exposure_coefficients: np.ndarray | None = None,
        reserve_pv_price: float = RESERVE_PV_PRICE,
    ) -> None:
        self.num_advertisers = num_advertisers
        self.min_remaining_budget = min_remaining_budget

        self.rng = np.random.Generator(np.random.PCG64(seed))

        if num_slots is None or slot_exposure_coefficients is None:
            num_slots = NUM_SLOTS_DEFAULT
            slot_exposure_coefficients = np.array(SLOT_EXPOSURE_COEFFS)

        self.num_slots = num_slots
        self.slot_exposure_coefficients = slot_exposure_coefficients
        self.reserve_pv_price = reserve_pv_price

        self.value_sampler = ValueSampler(
            num_advertisers=num_advertisers, 
            seed=seed, 
            lower_bound_std=-2, 
            upper_bound_std=2,
        )

    def zero_bids_for_low_budget(self, bids: np.ndarray, remaining_budgets: np.ndarray) -> np.ndarray:
        """
        Zero out bids for advertisers whose remaining budget is below self.min_remaining_budget.
        Args:
            bids: np.ndarray of shape (num_pv, num_advertisers)
            remaining_budgets: np.ndarray of shape (num_advertisers,)
        Returns:
            np.ndarray: masked bids of the same shape as input
        """
        mask = remaining_budgets >= self.min_remaining_budget
        masked_bids = bids * mask[None, :]
        return masked_bids

    def simulate_ad_bidding_multi_slots(
        self,
        p_values: np.ndarray,
        p_value_sigmas: np.ndarray,
        bids: np.ndarray,
        remaining_budgets: np.ndarray,
        sampled_conversions: np.ndarray | None = None,
        sampled_slot_exposures: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_pv, num_adv = bids.shape
        k = min(num_adv, self.num_slots + 1)  # top-k

        if sampled_slot_exposures is None:
            # Broadcast sampling (no tile) + exposure continuity
            sampled_slot_exposures = (
                (self.rng.random((num_pv, self.num_slots)) < self.slot_exposure_coefficients)
                .cumprod(axis=1)
            )

        # Pre-allocations
        win_status = np.zeros((num_pv, num_adv), dtype=bool)
        cost = np.zeros((num_pv, num_adv), dtype=float)
        least_winning_price = np.zeros((num_pv,), dtype=float)
        winner_exposure = np.zeros((num_pv, num_adv), dtype=float)

        # In-place masking for budgets (monotonic growth)
        bids_masked = bids.copy()
        converged = False
        while not converged:
            # Max allowed winners per PV via non-zero bids
            # (need s+1 bidders to price slot s)
            # If no allocations possible anywhere, short-circuit
            num_nonzero_bids = np.count_nonzero(bids_masked, axis=1)
            allowed_winners = np.clip(num_nonzero_bids - 1, 0, self.num_slots)
            if np.all(allowed_winners == 0):
                win_status.fill(False)
                cost.fill(0.0)
                least_winning_price.fill(0.0)
                break

            # Top-(num_slots+1) per pv using argpartition + small-k sort
            part_idx = np.argpartition(-bids_masked, kth=k - 1, axis=1)[:, :k]
            part_vals = np.take_along_axis(bids_masked, part_idx, axis=1)
            order = np.argsort(-part_vals, axis=1)
            top_idx_sorted = np.take_along_axis(part_idx, order, axis=1)

            # Winners = first num_slots indices; market price = next value per slot
            winner_indices = top_idx_sorted[:, :self.num_slots]

            # Market prices per slot
            m = max(0, min(self.num_slots, k - 1))
            market_prices_per_slot = np.zeros((num_pv, self.num_slots), dtype=float)
            if m > 0:
                nxt = np.take_along_axis(bids_masked, top_idx_sorted[:, 1 : 1 + m], axis=1)
                market_prices_per_slot[:, :m] = nxt

            # Gate by available competition: only first allowed_winners slots count
            slot_mask = (np.arange(self.num_slots)[None, :] < allowed_winners[:, None])

            # Reserve-price gating: mark slots with market price below reserve as unsold
            reserve_mask = (market_prices_per_slot >= self.reserve_pv_price)
            slot_mask = slot_mask & reserve_mask

            # Apply final mask to market prices
            market_prices_per_slot *= slot_mask

            # Build per-ad arrays using put_along_axis
            win_status.fill(False)
            np.put_along_axis(win_status, winner_indices, slot_mask, axis=1)
            winner_market_prices = np.zeros((num_pv, num_adv), dtype=float)
            np.put_along_axis(winner_market_prices, winner_indices, market_prices_per_slot, axis=1)
            winner_exposure.fill(0.0)
            np.put_along_axis(winner_exposure, winner_indices, sampled_slot_exposures * slot_mask, axis=1)

            cost = winner_market_prices * winner_exposure

            # Least winning price (pre-exposure): price of the last allocated slot per pv
            least_winning_price.fill(0.0)
            effective_winners = slot_mask.sum(axis=1)
            has_alloc = effective_winners > 0
            if np.any(has_alloc):
                last_idx = effective_winners[has_alloc] - 1
                least_winning_price[has_alloc] = market_prices_per_slot[has_alloc, last_idx]

            # Budget enforcement
            cumulative_cost = cost.cumsum(axis=0)
            over_budget = cumulative_cost > remaining_budgets[None, :]

            new_bids_masked = np.where(over_budget, 0.0, bids_masked)
            converged = np.array_equal(new_bids_masked, bids_masked)
            bids_masked = new_bids_masked

        if sampled_conversions is None:  # NOTE: much slower!
            _, sampled_conversions = self.value_sampler.sample(
                p_values, p_value_sigmas, sample_conversions=True
            )
        conversions = win_status * winner_exposure * sampled_conversions

        return win_status, cost, conversions, least_winning_price
