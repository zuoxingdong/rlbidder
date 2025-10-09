from typing import Any, Dict, Sequence

import numpy as np
import polars as pl
from rliable import library as rly
from rliable import metrics as rly_metrics
from tqdm import tqdm

from rlbidder.evaluation.history import StepwiseAuctionHistory


def participation_rate_per_tick(history: StepwiseAuctionHistory, bid_threshold: float = 0.01) -> np.ndarray:
    """
    Participation rate: Active bidders / eligible bidders.
    
    Bids per auction / Active bidders: Direct signal of demand intensity; low counts flag weak competition and pressure on prices.
    Unique bidders/diversity: Reduces concentration risk; more distinct buyers means healthier, more resilient demand.
    """
    active_bidders = np.sum(history.avg_bids > bid_threshold, axis=1)
    total_eligible = history.avg_bids.shape[1]  # Total number of advertisers
    return active_bidders / total_eligible


def bid_gap_ratio_per_tick(history: StepwiseAuctionHistory, bid_threshold: float = 0.0, eps: float = 1e-12) -> np.ndarray:
    """
    First-second bid gap ratio per tick: (top1/top2) - 1, ignoring bids <= bid_threshold.
    Returns np.nan where fewer than 2 active bidders exist at a tick.

    It measures the depth right at the top of the book; shrinking gap can warn of colliding floors or thin demand.
    """
    bids = history.avg_bids.copy()
    active = bids > bid_threshold
    bids[~active] = -np.inf  # exclude inactive from top-2
    top2 = -np.partition(-bids, kth=1, axis=1)[:, :2]
    first, second = top2[:, 0], top2[:, 1]
    ratio = np.full(bids.shape[0], np.nan, dtype=float)
    valid = np.isfinite(second) & (second > eps)
    ratio[valid] = first[valid] / np.maximum(second[valid], eps) - 1.0
    return ratio


def hhi_cumulative_by_wins(history: StepwiseAuctionHistory, return_enc: bool = False) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Market concentration (HHI by wins/spend): Quantifies dependency on a few buyers; high HHI raises pricing and stability risk.
    
    Calculates the Herfindahl-Hirschman Index (HHI) based on cumulative win distribution
    from the start of the auction period to each timestep. HHI values range from 1/n 
    (perfectly competitive) to 1 (monopolistic), where n is the number of agents.
    
    Args:
        history: StepwiseAuctionHistory containing auction results
        return_enc: Whether to return the effective number of competitors (ENC)
        
    Returns:
        np.ndarray: HHI values for each timestep, shape (timesteps,)
    """
    # cumulative wins from start to each timestep
    wins = history.sum_win_status.astype(float)
    cumulative_wins = np.cumsum(wins, axis=0)
    
    total = cumulative_wins.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.divide(cumulative_wins, total, out=np.zeros_like(cumulative_wins), where=total > 0)
    hhi = np.nansum(shares**2, axis=1)
    if return_enc:
        with np.errstate(divide="ignore", invalid="ignore"):
            enc = np.divide(1.0, hhi, out=np.full_like(hhi, np.nan, dtype=float), where=hhi > 0)
        return hhi, enc
    return hhi


def gini_cumulative_by_wins(history: StepwiseAuctionHistory) -> np.ndarray:
    """
    Gini coefficient of the cumulative distribution of wins from start to each timestep.
    It measures the inequality in cumulative win distribution across agents; higher values indicate concentration.
    Returns 0 when a timestep has zero total cumulative wins.
    """
    # cumulative wins from start to each timestep
    wins = history.sum_win_status.astype(float)
    x = np.cumsum(wins, axis=0)  # (T, N)
    T, N = x.shape
    x_sorted = np.sort(x, axis=1)
    i = np.arange(1, N + 1, dtype=float)[None, :]  # (1, N)
    cum_term = (i * x_sorted).sum(axis=1)          # (T,)
    sums = x.sum(axis=1)                            # (T,)
    with np.errstate(divide='ignore', invalid='ignore'):
        g = (2.0 * cum_term) / (N * np.maximum(sums, 1e-12)) - (N + 1) / N
    g = np.where(sums > 0, g, 0.0)
    return g


def top_of_book_pressure_per_tick(history: StepwiseAuctionHistory, pct: float = 0.05, bid_threshold: float = 0.0) -> np.ndarray:
    """
    Number of bids within ±pct of the (proxy) clearing price per tick.
    Clearing price proxy per tick is the mean of avg_leastWinningCosts across advertisers.

    It measures the pressure on the top of the book; high pressure can indicate colliding floors or thin demand.
    Higher values

    Only counts bids > bid_threshold.
    """
    # price per tick (scalar): proxy from average clearing prices across advertisers
    price_t = np.nanmean(history.avg_leastWinningCosts, axis=1)  # (T,)
    lower = (1.0 - pct) * price_t[:, None]
    upper = (1.0 + pct) * price_t[:, None]
    bids = history.avg_bids
    mask = (bids > bid_threshold) & (bids >= lower) & (bids <= upper)
    return mask.sum(axis=1).astype(int)


def realized_volatility(history: StepwiseAuctionHistory, window: int = 4, use_log: bool = True, eps: float = 1e-12) -> np.ndarray:
    """
    Rolling realized volatility of clearing price returns over time.
    Returns temporal pattern of volatility: sqrt(sum r_t^2) over rolling windows.
    
    Args:
        history: Auction history containing clearing price data
        window: Rolling window size for volatility calculation
        use_log: Whether to use log returns (True) or simple returns (False)
        eps: Small value to avoid log(0) issues
        
    Returns:
        np.ndarray: Rolling volatility per timestep (NaN where insufficient history)
    """
    # Proxy clearing price per tick
    p = np.nanmean(history.avg_leastWinningCosts, axis=1)
    p = np.where(p > eps, p, np.nan)
    
    # Calculate returns
    if use_log:
        r = np.diff(np.log(p))
    else:
        r = np.diff(p) / p[:-1]
    
    # Rolling realized volatility
    out = np.full(p.shape, np.nan, dtype=float)
    for t in range(window, len(p)):
        out[t] = np.sqrt(np.nansum(np.square(r[t - window:t])))
    
    return out


def bid_aggressiveness_mean_per_tick(history: StepwiseAuctionHistory, bid_threshold: float = 0.01, eps: float = 1e-12) -> np.ndarray:
    """
    Aggressiveness per tick/advertiser: (clearing_price - b) / clearing_price, using clearing price proxy clearing_price = clearing_price from avg_leastWinningCosts.
    Returns matrix (T, N). Only defined where clearing_price > 0 and bid > bid_threshold; else NaN.
    """
    clearing_price = np.nanmean(history.avg_leastWinningCosts, keepdims=True, axis=1)  # (T, 1)
    b = history.avg_bids
    out = np.where(
        (clearing_price > eps) & (b > bid_threshold),
        (clearing_price - b) / clearing_price,
        np.nan,
    )
    out = np.nanmean(out, axis=1)
    return out


def winner_set_jaccard_per_tick(history: StepwiseAuctionHistory, win_threshold: float = 0.0) -> np.ndarray:
    """
    Jaccard similarity of winner sets between adjacent ticks based on `avg_win_status > win_threshold`.
    Returns (T,) with NaN at t=0.
    """
    W = history.avg_win_status > win_threshold  # (T, N)
    T, N = W.shape
    out = np.full(T, np.nan, dtype=float)
    for t in range(1, T):
        inter = np.sum(W[t] & W[t - 1])
        union = np.sum(W[t] | W[t - 1])
        out[t] = inter / union if union > 0 else np.nan
    return out


def _nan_rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return x
    x = np.asarray(x, dtype=float)
    w = int(window)
    sums = np.convolve(np.nan_to_num(x, nan=0.0), np.ones(w), mode="same")
    cnts = np.convolve(~np.isnan(x), np.ones(w), mode="same")
    out = np.divide(sums, cnts, out=np.full_like(x, np.nan, dtype=float), where=cnts > 0)
    return out


def compute_budget_pacing_line(traffic: np.ndarray, budget: float, timesteps: np.ndarray) -> np.ndarray:
    """
    Compute the budget pacing line based on traffic distribution.
    """
    if traffic.sum() > 0:
        pacing_weights = traffic / traffic.sum()
        pacing_line = budget * np.cumsum(pacing_weights)
    else:
        pacing_line = budget * (timesteps / (len(timesteps) - 1 if len(timesteps) > 1 else 1))
    return pacing_line


def compute_campaign_metrics(auction_history: StepwiseAuctionHistory, agent_idx: int) -> Dict[str, Any]:
    """
    Compute all derived metrics needed for visualization for a specific agent.
    
    Args:
        auction_history: StepwiseAuctionHistory object
        agent_idx: Index of the agent to compute metrics for
    """
    timesteps = np.arange(auction_history.num_steps)
    avg_bids = auction_history.avg_bids[:, agent_idx]
    std_bids = auction_history.std_bids[:, agent_idx]
    avg_alphas = auction_history.avg_alphas[:, agent_idx]
    std_alphas = auction_history.std_alphas[:, agent_idx]
    costs = auction_history.sum_costs[:, agent_idx]
    conversions = auction_history.sum_conversions[:, agent_idx]
    sum_wins = auction_history.sum_win_status[:, agent_idx]
    traffic = auction_history.volumes
    values = auction_history.pValues[:, agent_idx]
    win_rates = auction_history.avg_win_status[:, agent_idx]
    avg_auction_price = auction_history.avg_leastWinningCosts[:, agent_idx]
    std_auction_price = auction_history.std_leastWinningCosts[:, agent_idx]
    expected_conversions = np.cumsum(auction_history.pValues[:, agent_idx])

    cumulative_spend = np.cumsum(costs)
    cumulative_conversions = np.cumsum(conversions)
    per_step_cpa = np.where(
        conversions == 0,
        np.nan,
        costs / (conversions + 1e-10)
    )
    per_step_cpa_rolling = np.convolve(np.nan_to_num(per_step_cpa, nan=0), np.ones(3)/3, mode='same')
    cumulative_cpa = np.where(
        cumulative_conversions == 0,
        np.nan,
        cumulative_spend / (cumulative_conversions + 1e-10)
    )
    conversion_rate_per_win = np.divide(conversions, sum_wins, out=np.full_like(conversions, np.nan, dtype=np.float64), where=sum_wins > 0)
    avg_conversion_rate_per_win = np.nanmean(conversion_rate_per_win)

    # Bid premium vs clearing price proxy
    with np.errstate(divide="ignore", invalid="ignore"):
        bid_premium_pct = 100.0 * np.divide(
            (avg_bids - avg_auction_price),
            avg_auction_price,
            out=np.full_like(avg_bids, np.nan, dtype=np.float64),
            where=avg_auction_price > 0,
        )
    
    metrics = dict(
        timesteps=timesteps,
        avg_bids=avg_bids,
        std_bids=std_bids,
        avg_alphas=avg_alphas,
        std_alphas=std_alphas,
        costs=costs,
        conversions=conversions,
        cumulative_spend=cumulative_spend,
        cumulative_conversions=cumulative_conversions,
        per_step_cpa=per_step_cpa,
        per_step_cpa_rolling=per_step_cpa_rolling,
        cumulative_cpa=cumulative_cpa,
        conversion_rate_per_win=conversion_rate_per_win,
        avg_conversion_rate_per_win=avg_conversion_rate_per_win,
        traffic=traffic,
        values=values,
        win_rates=win_rates,
        avg_auction_price=avg_auction_price,
        std_auction_price=std_auction_price,
        expected_conversions=expected_conversions,
        sum_wins=sum_wins,
        bid_premium_pct=bid_premium_pct,
    )
    return metrics


def compute_agent_interval_estimates(
    df: pl.DataFrame,
    score_col: str = "mean_conversion",
    run_setting_cols: list = ["period", "rotate_index"],
    normalize_scores: bool = True,
    reps: int = 20000,
    task_bootstrap: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Compute interval estimates for agent performance using rliable library.
    
    Args:
        df: Polars DataFrame containing agent performance data
        score_col: Column name to use as the score metric
        run_setting_cols: List of columns that define experimental settings/tasks
        normalize_scores: Whether to normalize scores by the overall mean
        reps: Number of bootstrap replications
        task_bootstrap: Whether to perform bootstrapping over tasks
        
    Returns:
        tuple: (aggregate_scores, aggregate_score_cis) dictionaries
    """
    # Compute score dictionary for each agent: score shape (num_runs x num_settings)
    base_score = df.select(score_col).mean().item() if normalize_scores else 1.0
    
    score_dict = (
        df
        .with_columns(
            normalized_score=pl.col(score_col) / base_score
        )
        .group_by(["agent_name", "seed"])
        .agg(
            pl.col("normalized_score").sort_by(run_setting_cols)
        )
        .group_by("agent_name")
        .agg(
            pl.col("normalized_score").sort_by(["seed"])
        )
        .sort("agent_name")
        .rows_by_key("agent_name", unique=True)
    )
    score_dict = {k: np.array(v).squeeze() for k, v in score_dict.items()}

    aggregate_func = lambda x: np.array([
        rly_metrics.aggregate_mean(x),
        rly_metrics.aggregate_median(x),
        rly_metrics.aggregate_iqm(x),
        rly_metrics.aggregate_optimality_gap(x)
    ])

    aggregate_scores = {}
    aggregate_score_cis = {}
    
    for k, v in tqdm(score_dict.items()):
        # Compute Mean, Median, IQM, and Optimality Gap with 95% confidence intervals
        # Return scores and CIs = [lower bounds, upper bounds]
        scores, cis = rly.get_interval_estimates(
            {k: v},
            aggregate_func,  # e.g. rly.aggregate_iqm or custom
            reps=reps,
            task_bootstrap=task_bootstrap,
        )
        aggregate_scores[k] = scores[k]
        aggregate_score_cis[k] = cis[k]
    
    return aggregate_scores, aggregate_score_cis


def compute_auction_budget_pacing(
    auction_histories: Sequence[StepwiseAuctionHistory],
    agent_name: str = "CQL",
    sample_size: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts budget and budget left ratio for a given agent from a list of auction histories,
    and returns a random sample of the results.

    Args:
        auction_histories (list): List of auction history objects.
        agent_name (str): Name of the agent to filter.
        sample_size (int): Number of samples to return.
        seed (int): Random seed for reproducibility.

    Returns:
        budget (np.ndarray): Sampled budgets, shape (sample_size,).
        budget_left_ratio (np.ndarray): Sampled budget left ratios, shape (sample_size, num_steps).
    """
    budget = []
    budget_left_ratio = []
    for history in auction_histories:
        b = history.agent_budgets
        pacing = (b[None, :] - history.sum_costs.cumsum(axis=0)) / b[None, :]
        pacing = pacing.T  # (num_advertisers, num_steps)

        filter_index = (history.agent_names == agent_name)

        budget.append(b[filter_index])
        budget_left_ratio.append(pacing[filter_index])
    budget = np.concatenate(budget, axis=0)
    budget_left_ratio = np.concatenate(budget_left_ratio, axis=0)

    if len(budget) < sample_size:
        raise ValueError(f"Requested sample_size={sample_size} but only {len(budget)} samples available.")

    rng = np.random.Generator(np.random.PCG64(seed))
    sample_index = rng.choice(len(budget), size=sample_size, replace=False)
    budget = budget[sample_index]
    budget_left_ratio = budget_left_ratio[sample_index]

    return budget, budget_left_ratio
