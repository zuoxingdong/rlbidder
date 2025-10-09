import numpy as np
import polars as pl

from rlbidder.agents.budget_pacer import BudgetPacerBiddingAgent
from rlbidder.agents.fixed_cpa import FixedCPABiddingAgent
from rlbidder.agents.pid import PIDBudgetPacerBiddingAgent, PIDCPABiddingAgent
from rlbidder.agents.stochastic_cpa import StochasticCPABiddingAgent
from rlbidder.agents.value_scaled_cpa import ValueScaledCPABiddingAgent

# Predefined budgets and CPA constraints that mirror the rotation used during benchmarking.
# 48 advertisers total (6 categories × 8 agents each),
# allowing us to expose heterogeneous budget/CPA regimes for each agent.
EVAL_BUDGETS = np.array([
    [2900, 4350, 3000, 2400, 4800, 2000, 2050, 3500],
    [4600, 2000, 2800, 2350, 2050, 2900, 4750, 3450],
    [2000, 3500, 2200, 2700, 3100, 2100, 4850, 4100],
    [2000, 4800, 3050, 4250, 2850, 2250, 2000, 3900],
    [2000, 3250, 4450, 3550, 2700, 2100, 4650, 2000],
    [3400, 2650, 2300, 4100, 4800, 4450, 2000, 2050],
])
EVAL_CPAS = np.array([
    [100, 70, 90, 110, 60, 130, 120, 80],
    [70, 130, 100, 110, 120, 90, 60, 80],
    [130, 80, 110, 100, 90, 120, 60, 70],
    [120, 60, 90, 70, 100, 110, 130, 80],
    [120, 90, 70, 80, 100, 110, 60, 130],
    [90, 100, 110, 80, 60, 70, 130, 120],
])
# NOTE: this indicies only determined pValues, pValueSigmas, not the budget and cpa
ADVERTISER_INDICES = np.array([
    [0, 8, 16, 24, 32, 40, 6, 14],
    [1, 9, 17, 25, 33, 41, 7, 15],
    [2, 10, 18, 26, 34, 42, 22, 30],
    [3, 11, 19, 27, 35, 43, 23, 31],
    [4, 12, 20, 28, 36, 44, 38, 46],
    [5, 13, 21, 29, 37, 45, 39, 47],]
)


def create_shuffled_agent_indices(lf: pl.LazyFrame, seed: int = 42) -> np.ndarray:
    """
    Create shuffled agent indices grouped by advertiser category.
    
    Args:
        lf: Polars LazyFrame containing the campaign data
        seed: Random seed for reproducible shuffling
        
    Returns:
        numpy.ndarray: Array of shuffled advertiser indices with shape (num_categories, num_advertisers_per_category)
    """
    campaign_summary = (
        lf
        .group_by("advertiserCategoryIndex")
        .agg(
            pl.col("advertiserNumber").n_unique().alias("num_advertisers"),
            pl.col("budget").unique(),
            pl.col("CPAConstraint").unique(),
            pl.col("advertiserNumber").unique().alias("advertiser_index"),
        )
        .sort("advertiserCategoryIndex")
        .collect()
    )

    rng = np.random.Generator(np.random.PCG64(seed))

    agent_indices = np.vstack(campaign_summary["advertiser_index"].to_list())
    np.apply_along_axis(rng.shuffle, axis=0, arr=agent_indices)
    
    return agent_indices


BASELINE_AGENT_CATALOG: list[tuple[type, str, dict, dict, dict]] = [
    (
        FixedCPABiddingAgent,
        "FixedCPA",
        {
            "summary": "Bid using a constant CPA target; deterministic baseline",
            "ideal_for": "Simple baseline",
        },
        {},
        {},
    ),
    (
        StochasticCPABiddingAgent,
        "StochasticCPA",
        {
            "summary": "Samples CPA multipliers around target to encourage exploration",
            "ideal_for": "Simple baseline",
        },
        {"seed": 42, "cpa_std_ratio": 0.01},
        {},
    ),
    (
        ValueScaledCPABiddingAgent,
        "ValueScaledCPA",
        {
            "summary": "Scales bids based on p-value ratios while keeping CPA on target",
            "ideal_for": "Simple baseline",
        },
        {"clip_weights": (0.51, 1.23)},
        {},
    ),
    (
        BudgetPacerBiddingAgent,
        "BudgetPacer",
        {
            "summary": "Heuristic budget pacing with multiplicative adjustments",
            "ideal_for": "Simple budget pacing",
        },
        {
            "low_spend_threshold": 0.66,
            "high_spend_threshold": 1.45,
            "increase_factor": 1.13,
            "decrease_factor": 0.56,
        },
        {},
    ),
    (
        PIDBudgetPacerBiddingAgent,
        "PIDBudgetPacer",
        {
            "summary": "PID controller on spend trajectory",
            "ideal_for": "Simple budget pacing",
        },
        {"pid_params": (0.01, 2.568e-06, 0.000128)},
        {},
    ),
    (
        PIDCPABiddingAgent,
        "PIDCPA",
        {
            "summary": "PID controller on CPA violation",
            "ideal_for": "Simple CPA control",
        },
        {"pid_params": (0.004, 8.556e-06, 0.00373)},
        {},
    ),
]


def get_baseline_agent_configs(seed: int = 42) -> list[tuple[type, str, dict, dict]]:
    """Return canonical baseline agent configs for evaluation loops.

    Each entry is a tuple containing the agent class, a human-readable name,
    keyword arguments passed to the constructor, and a campaign configuration
    dictionary containing budgets/CPAs/indices. The catalog uses deterministic
    budgets so evaluation comparisons remain reproducible.

    Args:
        seed: Random seed for the stochastic CPA agent; forwarded automatically.

    Example:
        >>> configs = get_baseline_agent_configs()
        >>> for cls, name, agent_kwargs, campaign_cfg in configs:
        ...     print(name, campaign_cfg["adv_indicies"][:3])

    Returns:
        list[tuple[type, str, dict, dict]]: Ready-to-instantiate agent configs.
    """

    configs: list[tuple[type, str, dict, dict]] = []

    for i, (agent_class, agent_name, meta, agent_kws, campaign_cfg) in enumerate(BASELINE_AGENT_CATALOG):
        merged_agent_kwargs = {**agent_kws}
        if agent_class is StochasticCPABiddingAgent:
            merged_agent_kwargs.setdefault("seed", seed)

        campaign_config = {
            **campaign_cfg,
            "budget": EVAL_BUDGETS[i],
            "cpa": EVAL_CPAS[i],
            "category": np.arange(len(EVAL_BUDGETS)),
            "adv_indicies": ADVERTISER_INDICES[i],
        }
        configs.append((agent_class, f"{agent_name}-env", merged_agent_kwargs, campaign_config))

    return configs


def describe_baseline_agents() -> pl.DataFrame:
    """Summarize baseline agent catalog with metadata for documentation tables."""

    rows = []
    for (_, agent_name, meta, agent_kwargs, _) in BASELINE_AGENT_CATALOG:
        rows.append(
            {
                "agent": agent_name,
                "summary": meta.get("summary", ""),
                "ideal_for": meta.get("ideal_for", ""),
                "default_kwargs": agent_kwargs,
            }
        )
    return pl.DataFrame(rows)
