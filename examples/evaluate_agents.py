import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import draccus
import joblib
import torch

from rlbidder.agents import (  # noqa: E402
    BCBiddingAgent,
    BudgetPacerBiddingAgent,
    ConstantBidAgent,
    CQLBiddingAgent,
    DTBiddingAgent,
    FixedCPABiddingAgent,
    GASBiddingAgent,
    GAVEBiddingAgent,
    IQLBiddingAgent,
    PIDBudgetPacerBiddingAgent,
    PIDCPABiddingAgent,
    StochasticCPABiddingAgent,
    ValueScaledCPABiddingAgent,
)
from rlbidder.agents.get_baseline_agent_configs import get_baseline_agent_configs  # noqa: E402
from rlbidder.constants import (
    DEFAULT_NUM_EVAL_SEEDS,
    DEFAULT_SEED,
    MIN_REMAINING_BUDGET,
    STATE_DIM,
)
from rlbidder.evaluation import (  # noqa: E402
    OnlineCampaignEvaluator,
    ParallelOnlineCampaignEvaluator,
)
from rlbidder.utils import configure_logging

torch.set_float32_matmul_precision("high")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class AgentConfig:
    """Configuration for a single agent to evaluate"""
    agent_class: str = "IQLBiddingAgent"  # Class name of the agent
    name: Optional[str] = None  # Display name for the agent (if None, will use hardcoded name)
    model_dir: str = str(CURRENT_DIR / "checkpoints" / "iql")
    backbone_model_dir: str = str(CURRENT_DIR / "checkpoints" / "dt")
    checkpoint_file: str = "best.ckpt"
    state_dim: int = STATE_DIM
    seed: int = DEFAULT_SEED
    target_rtg: float = 1.0


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation process"""
    data_dir: str = str(PROJECT_ROOT / "data")
    evaluator_type: str = "ParallelOnlineCampaignEvaluator"  # Options: OnlineCampaignEvaluator, ParallelOnlineCampaignEvaluator
    min_remaining_budget: float = MIN_REMAINING_BUDGET
    verbose: bool = True
    delivery_period_indices: list[int] = field(default_factory=lambda: [7, 8])
    budget_ratio: Optional[float] = 1.0
    cpa_ratio: Optional[float] = 1.0
    num_seeds: int = DEFAULT_NUM_EVAL_SEEDS
    base_seed: int = DEFAULT_SEED
    num_workers: int | None = 8
    chunksize: int | None = 3
    output_dir: str = str(CURRENT_DIR / "eval")
    cache_dir: Optional[str] = None
    # Baseline agent configuration
    num_baseline_agents: Optional[int] = None  # Number of baseline agents to use (None = all)
    baseline_agents_seed: int = DEFAULT_SEED  # Seed for baseline agent sampling


@dataclass
class Config:
    """Main configuration class"""
    agent: AgentConfig = field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def generate_seeds(base_seed: int, num_seeds: int, max_value: int = 2**32 - 1) -> list[int]:
    """Generate deterministic seeds for reproducible evaluation"""
    seeds = []
    for i in range(num_seeds):
        key = f"{base_seed}-{i}".encode('utf-8')
        digest = hashlib.sha256(key).digest()
        # Use first 8 bytes for a 64-bit integer seed, then mod to fit numpy's uint32 range
        seed = int.from_bytes(digest[:8], "big") % (max_value + 1)
        seeds.append(seed)
    return seeds

def create_agent_config(cfg: AgentConfig) -> tuple[type, str, dict[str, Any], dict[str, Any]]:
    """Create agent configuration tuple from config"""
    model_dir = Path(cfg.model_dir)
    
    # Use provided name or fallback to hardcoded name based on agent class
    if cfg.agent_class == "CQLBiddingAgent":
        name = cfg.name if cfg.name is not None else "CQL"
        return (CQLBiddingAgent, name, {
            "model_dir": model_dir,
            "checkpoint_file": cfg.checkpoint_file,
            "state_dim": cfg.state_dim,
        }, {})
    elif cfg.agent_class == "IQLBiddingAgent":
        name = cfg.name if cfg.name is not None else "IQL"
        return (IQLBiddingAgent, name, {
            "model_dir": model_dir,
            "checkpoint_file": cfg.checkpoint_file,
            "state_dim": cfg.state_dim,
        }, {})
    elif cfg.agent_class == "BCBiddingAgent":
        name = cfg.name if cfg.name is not None else "BC"
        return (BCBiddingAgent, name, {
            "model_dir": model_dir,
            "checkpoint_file": cfg.checkpoint_file,
            "state_dim": cfg.state_dim,
        }, {})
    elif cfg.agent_class == "DTBiddingAgent":
        name = cfg.name if cfg.name is not None else "DT"
        return (DTBiddingAgent, name, {
            "model_dir": model_dir,
            "checkpoint_file": cfg.checkpoint_file,
            "state_dim": cfg.state_dim,
            "target_rtg": cfg.target_rtg,
        }, {})
    elif cfg.agent_class == "GAVEBiddingAgent":
        name = cfg.name if cfg.name is not None else "GAVE"
        return (GAVEBiddingAgent, name, {
            "model_dir": model_dir,
            "checkpoint_file": cfg.checkpoint_file,
            "state_dim": cfg.state_dim,
            "target_rtg": cfg.target_rtg,
        }, {})
    elif cfg.agent_class == "GASBiddingAgent":
        name = cfg.name if cfg.name is not None else "GAS"
        return (GASBiddingAgent, name, {
            "model_dir": model_dir,
            "backbone_model_dir": cfg.backbone_model_dir,
            "checkpoint_file": cfg.checkpoint_file,
            "state_dim": cfg.state_dim,
            "target_rtg": cfg.target_rtg,
        }, {})
    elif cfg.agent_class == "FixedCPABiddingAgent":
        name = cfg.name if cfg.name is not None else "FixedCPA"
        return (FixedCPABiddingAgent, name, {}, {})
    elif cfg.agent_class == "ConstantBidAgent":
        name = cfg.name if cfg.name is not None else "ConstantBid"
        return (ConstantBidAgent, name, {
            "bid_fraction": 0.003,  # this has been tuned!
        }, {})
    elif cfg.agent_class == "ValueScaledCPABiddingAgent":
        return (
            ValueScaledCPABiddingAgent,
            cfg.name if cfg.name is not None else "ValueScaledCPA",
            {
                "clip_weights": (0.51, 1.23),
            },
            {},
        )
    elif cfg.agent_class == "StochasticCPABiddingAgent":
        return (
            StochasticCPABiddingAgent,
            cfg.name if cfg.name is not None else "StochasticCPA",
            {
                "cpa_std_ratio": 0.01,
                "seed": cfg.seed,
            },
            {},
        )
    elif cfg.agent_class == "BudgetPacerBiddingAgent":
        return (
            BudgetPacerBiddingAgent,
            cfg.name if cfg.name is not None else "BudgetPacer",
            {
                "low_spend_threshold": 0.77,
                "high_spend_threshold": 1.62,
                "increase_factor": 1.01,
                "decrease_factor": 0.84,
            }, 
            {},
        )
    elif cfg.agent_class == "PIDBudgetPacerBiddingAgent":
        return (
            PIDBudgetPacerBiddingAgent,
            cfg.name if cfg.name is not None else "PIDBudgetPacer",
            {
                "pid_params": (0.01, 2.568e-06, 0.000128),
            },
            {},
        )
    elif cfg.agent_class == "PIDCPABiddingAgent":
        return (
            PIDCPABiddingAgent,
            cfg.name if cfg.name is not None else "PIDCPA",
            {
                "pid_params": (0.004, 8.556e-06, 0.00373),
            },
            {},
        )
    else:
        raise ValueError(f"Unknown agent_class: {cfg.agent_class}")


def create_evaluator(cfg: EvaluationConfig) -> OnlineCampaignEvaluator | ParallelOnlineCampaignEvaluator:
    """Create the appropriate evaluator based on configuration"""
    data_dir = Path(cfg.data_dir)
    
    if cfg.evaluator_type == "OnlineCampaignEvaluator":
        return OnlineCampaignEvaluator(
            data_dir=data_dir / "processed",
            min_remaining_budget=cfg.min_remaining_budget,
            delivery_period_indices=cfg.delivery_period_indices,
            base_seed=cfg.base_seed,
            num_seeds=cfg.num_seeds,
            cache_dir=(Path(cfg.cache_dir) if cfg.cache_dir is not None else None),
            verbose=cfg.verbose,
        )
    elif cfg.evaluator_type == "ParallelOnlineCampaignEvaluator":
        return ParallelOnlineCampaignEvaluator(
            data_dir=data_dir / "processed",
            min_remaining_budget=cfg.min_remaining_budget,
            verbose=cfg.verbose,
        )
    else:
        raise ValueError(f"Unknown evaluator_type: {cfg.evaluator_type}")


@draccus.wrap()
def main(cfg: Config = Config()) -> None:
    """Main evaluation function"""
    logger.info("Starting agent evaluation with config:")
    logger.info("  Agent: %s (%s)", cfg.agent.agent_class, cfg.agent.name)
    logger.info("  Evaluator: %s", cfg.evaluation.evaluator_type)
    logger.info("  Delivery periods: %s", cfg.evaluation.delivery_period_indices)
    logger.info("  Baseline agents: %s (seed: %s)", cfg.evaluation.num_baseline_agents or 'all', cfg.evaluation.baseline_agents_seed)
    
    # Create evaluator
    evaluator = create_evaluator(cfg.evaluation)
    
    # Create agent and competitor configurations
    agent_config = create_agent_config(cfg.agent)
    all_agent_configs = get_baseline_agent_configs(seed=cfg.evaluation.baseline_agents_seed)
    logger.info("Loaded %s baseline competitor agents", len(all_agent_configs))
    
    # Generate seeds for reproducible evaluation
    seeds = generate_seeds(cfg.evaluation.base_seed, cfg.evaluation.num_seeds)
    logger.info("Generated seeds: %s", seeds)
    
    # Ensure output directory exists
    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Evaluating agent: %s", agent_config[1])
    
    try:
        df_campaign_reports, df_agent_summaries, auction_histories, agents = evaluator.evaluate(
            control_agent_configs=agent_config,
            all_agent_configs=all_agent_configs,
            delivery_period_indices=cfg.evaluation.delivery_period_indices,
            budget_ratio=cfg.evaluation.budget_ratio,
            cpa_ratio=cfg.evaluation.cpa_ratio,
            seeds=seeds,
            num_workers=cfg.evaluation.num_workers,
            chunksize=cfg.evaluation.chunksize,
            cache_dir=(Path(cfg.evaluation.cache_dir) if cfg.evaluation.cache_dir is not None else None),
        )
        
        # Construct output filename with key configuration details
        agent_name = agent_config[1]
        checkpoint_file = agent_config[2].get("checkpoint_file", "nockpt").replace(".ckpt", "")
        
        filename = f"df_agent_summaries_{agent_name}_{checkpoint_file}.parquet"
        output_path = output_dir / filename
        
        df_agent_summaries.write_parquet(output_path)
        logger.info("Saved df_agent_summaries to %s", output_path)
        
        # Also save campaign reports for detailed analysis
        campaign_filename = f"df_campaign_reports_{agent_name}_{checkpoint_file}.parquet"
        campaign_output_path = output_dir / campaign_filename
        df_campaign_reports.write_parquet(campaign_output_path)
        logger.info("Saved df_campaign_reports to %s", campaign_output_path)
        
        # Save auction histories as well
        auction_filename = f"auction_histories_{agent_name}_{checkpoint_file}.joblib"
        auction_output_path = output_dir / auction_filename
        joblib.dump(auction_histories, auction_output_path)
        logger.info("Saved auction_histories to %s", auction_output_path)
        
    except Exception as e:
        logger.exception("Error during evaluation: %s", e)
        raise


if __name__ == "__main__":
    configure_logging(level=logging.DEBUG)
    main()
