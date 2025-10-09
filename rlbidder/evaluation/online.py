import logging
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import lightning as L
import numpy as np
import polars as pl
from tqdm import tqdm

from rlbidder.constants import (
    DEFAULT_SEED,
    MIN_REMAINING_BUDGET,
    NUM_SLOTS_DEFAULT,
    RESERVE_PV_PRICE,
    SLOT_EXPOSURE_COEFFS,
)
from rlbidder.envs.sampler import sample_pValues_and_conversions_scipy
from rlbidder.envs.simulator import AuctionSimulator
from rlbidder.evaluation.history import StepwiseAuctionHistory
from rlbidder.evaluation.utils import evaluate_score_with_constraint_penalty
from rlbidder.utils import generate_seeds

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def load_delivery_period_auction_data(
    data_dir: Path,
    period: int,
    verbose: bool = True,
) -> dict[str, Any]:
    logger.info("Loading data for period=%s from %s...", period, data_dir)
    lf = (
        pl.scan_parquet(data_dir / f"eval-period-{period}.parquet")
        .filter(pl.col("deliveryPeriodIndex") == period)
    )
    logger.debug("LazyFrame created and filtered by deliveryPeriodIndex.")

    logger.debug("Calculating number of advertisers and timesteps...")
    num_advertisers, num_timesteps = (
        lf.select(
            pl.col("advertiserNumber").approx_n_unique(),
            pl.col("timeStepIndex").approx_n_unique(),
        )
        .collect()
        .to_numpy()
        .squeeze()
    )
    logger.info("Found %s advertisers and %s timesteps.", num_advertisers, num_timesteps)

    logger.debug("Aggregating budget, CPA, and category for each advertiser...")
    budget, cpa, category = (
        lf
        .group_by("advertiserNumber")
        .agg(
            pl.col("budget").first(),
            pl.col("CPAConstraint").first(),
            pl.col("advertiserCategoryIndex").first(),
        )
        .sort("advertiserNumber")
        .collect()
        .select(["budget", "CPAConstraint", "advertiserCategoryIndex"])
        .to_numpy()
        .T
    )
    logger.debug("Advertiser info loaded.")

    logger.debug("Aggregating pValues and pValueSigmas for each timestep...")
    df_values = (
        lf
        .group_by("timeStepIndex")
        .agg(
            pl.col("pValue").sort_by("advertiserNumber"),
            pl.col("pValueSigma").sort_by("advertiserNumber"),
        )
        .sort("timeStepIndex")
        .collect()
    )
    logger.debug("pValues and pValueSigmas aggregated.")

    logger.debug("Converting pValues and pValueSigmas to numpy arrays...")
    pValues = [np.asarray(list(p), dtype=float).T for p in df_values.get_column("pValue")]
    pValueSigmas = [np.asarray(list(p), dtype=float).T for p in df_values.get_column("pValueSigma")]
    logger.debug("Conversion complete.")

    logger.info("All data loaded and processed.")
    return {
        "lf": lf,
        "num_advertisers": num_advertisers,
        "num_timesteps": num_timesteps,
        "budget": budget,
        "cpa": cpa,
        "category": category,
        "df_values": df_values,
        "pValues": pValues,
        "pValueSigmas": pValueSigmas,
    }


def simulate_multi_agent_campaign(
    agents: list,
    period: int,
    rotate_index: int,
    num_advertisers: int,
    num_timesteps: int,
    pValues: list[np.ndarray],
    pValueSigmas: list[np.ndarray],
    sampled_conversions: list[np.ndarray] | None = None,
    budget_ratio: float = 1.0,
    cpa_ratio: float = 1.0,
    min_remaining_budget: float = MIN_REMAINING_BUDGET,
    verbose: bool = True,
    seed: int = DEFAULT_SEED,
) -> tuple[pl.DataFrame, pl.DataFrame, StepwiseAuctionHistory]:
    L.seed_everything(seed, verbose=verbose)

    # Create AuctionSimulator
    simulator = AuctionSimulator(
        num_advertisers=num_advertisers, 
        min_remaining_budget=min_remaining_budget,
        seed=seed,
        num_slots=NUM_SLOTS_DEFAULT,
        slot_exposure_coefficients=np.array(SLOT_EXPOSURE_COEFFS),
        reserve_pv_price=RESERVE_PV_PRICE,
    )
    
    auction_history = StepwiseAuctionHistory(
        num_steps=num_timesteps, 
        num_advertisers=num_advertisers,
    )
    auction_history.set_agent_meta_info(agents)

    # Reset all agents with the provided budget and cpa ratios
    for agent in agents:
        agent.reset(
            budget_ratio=budget_ratio,
            cpa_ratio=cpa_ratio,
        )

    pbar = tqdm(range(num_timesteps), desc=f"Period {period} Rotate {rotate_index}", disable=not verbose)
    for timeStep_index in pbar:
        # Sample pValues and pValueSigmas for the current timestep
        p_values = pValues[timeStep_index]
        p_value_sigmas = pValueSigmas[timeStep_index]

        # Assume all agents start with full budgets
        remaining_budget = np.zeros(num_advertisers)
        for agent in agents:
            remaining_budget[agent.adv_indicies] = agent.remaining_budget

        # Create a bids array with the shape (num_pv, num_advertisers)
        bids = np.zeros_like(p_values)
        for agent in agents:
            bids[:, agent.adv_indicies] = agent.bidding(
                timeStepIndex=timeStep_index,
                pValues=p_values[:, agent.adv_indicies],
                pValueSigmas=p_value_sigmas[:, agent.adv_indicies],
                auction_history=auction_history,
            )

        # Mask bids for agents with low remaining budget
        bids = simulator.zero_bids_for_low_budget(bids, remaining_budget)

        # Simulate the auction
        win_status, cost, conversions, least_winning_price = simulator.simulate_ad_bidding_multi_slots(
            p_values,
            p_value_sigmas,
            bids,
            remaining_budget,
            sampled_conversions=sampled_conversions[timeStep_index] if sampled_conversions is not None else None,
        )

        # Update each agent's remaining budget
        for agent in agents:
            total_cost = cost[:, agent.adv_indicies].sum(axis=0)
            agent.spend(total_cost)

        # Update auction history with the sampled values
        auction_history.push(
            pValues=p_values,
            bids=bids,
            least_winning_costs=least_winning_price,
            win_status=win_status,
            costs=cost,
            conversions=conversions,
        )

    # Extract results from auction history
    total_conversions = auction_history.get_total_conversions()
    realized_cpa = auction_history.compute_realized_cpa()
    cpa_constraints = np.zeros(num_advertisers)
    for agent in agents:
        cpa_constraints[agent.adv_indicies] = agent.cpa
    scores = evaluate_score_with_constraint_penalty(
        value=total_conversions,
        estimated_constraint=realized_cpa,
        target_constraint=cpa_constraints,
        beta=2,
    )

    # Campaign performance summary
    df_campaign_report = pl.DataFrame({
        "period": [period],
        "rotate_index": [rotate_index],
        "num_agents": [len(agents)],
        "seed": [seed],
        "num_advertisers": [int(num_advertisers)],
        "num_timesteps": [int(num_timesteps)],
        "total_conversions_all": [float(np.nansum(total_conversions))],
        "mean_realized_cpa": [float(np.nanmean(realized_cpa))],
        "mean_cpa_constraint": [float(np.mean(cpa_constraints))],
        "total_conversions_per_advertiser": [total_conversions.tolist()],
        "realized_cpa_per_advertiser": [np.round(realized_cpa, 2).tolist()],
        "cpa_constraints_per_advertiser": [cpa_constraints.tolist()],
    })

    # Per-agent summary
    agent_summary = []
    for agent in agents:
        indices = agent.adv_indicies
        budget_spent_ratio = (agent.budget - agent.remaining_budget) / agent.budget
        cpa_exceed_ratio = (realized_cpa[indices] - cpa_constraints[indices]) / (cpa_constraints[indices] + 1e-10)
        agent_summary.append({
            "period": period,
            "rotate_index": rotate_index,
            "seed": seed,
            "agent_name": agent.name,
            "advertiser_indices": indices.tolist(),
            "score_list": scores[indices].tolist(),
            "conversion_list": total_conversions[indices].tolist(),
            "realized_cpa_list": realized_cpa[indices].tolist(),
            "cpa_constraint_list": cpa_constraints[indices].tolist(),
            "cpa_exceed_ratio_list": cpa_exceed_ratio.tolist(),
            "budget_spent_ratio_list": budget_spent_ratio.tolist(),
        })

    df_agent_summary = pl.DataFrame(agent_summary)
    return df_campaign_report, df_agent_summary, auction_history


def initialize_multi_agents(
    all_agent_configs: list[tuple],
    control_agent_configs: tuple,
    control_index: int,
    budget_ratio: float | None = None,
    cpa_ratio: float | None = None,
) -> list:
    all_agent_configs = deepcopy(all_agent_configs)
    control_agent_configs = deepcopy(control_agent_configs)

    # Replace control agent
    control_agent_configs = (
        *control_agent_configs[:3],  # (AgentClass, name, agent_kws)
        all_agent_configs[control_index][-1],  # keep the same campaign config
    )
    all_agent_configs[control_index] = control_agent_configs

    agents = []
    for i, (AgentClass, name, agent_kws, campaign_cfg) in enumerate(all_agent_configs):
        agent = AgentClass(
            name=name,
            **agent_kws,
            **campaign_cfg,
        )
        agent.reset(budget_ratio=budget_ratio, cpa_ratio=cpa_ratio)
        agents.append(agent)
    
    return agents


class OnlineCampaignEvaluator:
    def __init__(
        self,
        data_dir: Path,
        min_remaining_budget: float = MIN_REMAINING_BUDGET,
        delivery_period_indices: list[int] | None = None,
        seeds: list[int] | None = None,
        base_seed: int | None = None,
        num_seeds: int | None = None,
        cache_dir: Path | None = None,
        verbose: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.min_remaining_budget = min_remaining_budget
        self.verbose = verbose
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        
        if delivery_period_indices is None:
            # Determine unique deliveryPeriodIndex values from all campaign files
            delivery_period_indices = (
                pl.scan_parquet(self.data_dir / "eval-period-*.parquet")
                .select("deliveryPeriodIndex")
                .unique()
                .sort("deliveryPeriodIndex")
                .collect()
                .get_column("deliveryPeriodIndex")
                .to_list()
            )
            
        self.delivery_period_indices = [int(p) for p in delivery_period_indices]

        # Seed handling: explicit list wins; otherwise derive from base/num or fallback to default
        if seeds is not None:
            self.seeds = seeds
        elif base_seed is not None and num_seeds is not None:
            self.seeds = generate_seeds(base_seed, num_seeds)
        else:
            self.seeds = [DEFAULT_SEED]
        
        self.eval_data = {}
        for period in self.delivery_period_indices:
            logger.info("Loading auction data for period %s...", period)
            data = load_delivery_period_auction_data(self.data_dir, period, verbose=self.verbose)
            
            cache_root = self.cache_dir or self.data_dir
            logger.info(
                "Preparing presampled conversions for period %s with %s seeds (cache in %s)...",
                period,
                len(self.seeds),
                cache_root,
            )
            presampled_conversions = {}
            for seed in self.seeds:
                cache_path = cache_root / f"presampled_conversions_period-{period}_seed-{seed}.joblib"
                if cache_path.exists():
                    logger.info("Loading presampled conversions from cache: %s", cache_path.name)
                    try:
                        presampled_conversions[seed] = joblib.load(cache_path)
                        continue
                    except Exception as exc:
                        logger.warning("Failed to load cache %s, regenerating. Error: %s", cache_path.name, exc)
                # If no cache or load failed, generate and attempt to save
                _, sampled_conversions = sample_pValues_and_conversions_scipy(
                    pValues=data["pValues"], 
                    pValueSigmas=data["pValueSigmas"], 
                    seed=seed, 
                )
                presampled_conversions[seed] = sampled_conversions
                try:
                    joblib.dump(sampled_conversions, cache_path, compress=3)
                    logger.info("Saved presampled conversions to cache: %s", cache_path.name)
                except Exception as exc:
                    logger.warning("Failed to save cache to %s: %s", cache_path.name, exc)
            
            self.eval_data[period] = {
                "num_advertisers": data["num_advertisers"],
                "num_timesteps": data["num_timesteps"],
                "pValues": data["pValues"],
                "pValueSigmas": data["pValueSigmas"],
                "presampled_conversions": presampled_conversions,
            }
            
            logger.info(
                "Period %s data loaded: %s timesteps, %s advertisers",
                period,
                data['num_timesteps'],
                data['num_advertisers'],
            )

    def evaluate(
        self,
        control_agent_configs: tuple,
        all_agent_configs: list[tuple],
        budget_ratio: float | None = None,
        cpa_ratio: float | None = None,
        **kwargs: Any,
    ) -> tuple[pl.DataFrame, pl.DataFrame, list[StepwiseAuctionHistory], list]:
        campaign_reports = []
        agent_summaries = []
        auction_histories = []

        for period in self.delivery_period_indices:
            data = self.eval_data[period]
            
            rotate_indicies = range(len(all_agent_configs))
            pbar = tqdm(
                list(product(self.seeds, rotate_indicies)), 
                desc=f"Period {period}", 
                disable=not self.verbose
            )
            for seed, rotate_index in pbar:
                agents = initialize_multi_agents(
                    all_agent_configs=all_agent_configs,
                    control_agent_configs=control_agent_configs,
                    control_index=rotate_index,
                    budget_ratio=budget_ratio,
                    cpa_ratio=cpa_ratio,
                )
                df_campaign_report, df_agent_summary, auction_history = simulate_multi_agent_campaign(
                    agents=agents,
                    period=period,
                    rotate_index=rotate_index,
                    num_advertisers=data["num_advertisers"],
                    num_timesteps=data["num_timesteps"],
                    pValues=data["pValues"],
                    pValueSigmas=data["pValueSigmas"],
                    sampled_conversions=data["presampled_conversions"][seed],
                    budget_ratio=budget_ratio,
                    cpa_ratio=cpa_ratio,
                    min_remaining_budget=self.min_remaining_budget,
                    verbose=False,
                    seed=seed,
                )
                campaign_reports.append(df_campaign_report)
                agent_summaries.append(df_agent_summary)
                auction_histories.append(auction_history)

        df_campaign_reports = pl.concat(campaign_reports)
        df_agent_summaries = pl.concat(agent_summaries)

        return df_campaign_reports, df_agent_summaries, auction_histories, agents
