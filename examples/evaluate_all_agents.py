#!/usr/bin/env python3
"""
Script to evaluate all supported agents in evaluate_agents.py
"""
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import draccus
import torch

from rlbidder.constants import DEFAULT_NUM_EVAL_SEEDS, DEFAULT_SEED
from rlbidder.utils import configure_logging

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# All supported agent classes from evaluate_agents.py
AGENT_CLASSES = [
    "FixedCPABiddingAgent",
    "ConstantBidAgent",
    "StochasticCPABiddingAgent",
    "ValueScaledCPABiddingAgent",
    "BudgetPacerBiddingAgent",
    "PIDBudgetPacerBiddingAgent",
    "PIDCPABiddingAgent",
    "BCBiddingAgent", 
    "CQLBiddingAgent",
    "IQLBiddingAgent",
    "DTBiddingAgent",
    "GAVEBiddingAgent",
    "GASBiddingAgent",
]
# Model-based agents that need checkpoints
MODEL_BASED_AGENTS = {
    "CQLBiddingAgent": {"model_dir": "checkpoints/cql", "checkpoint_file": "best.ckpt"},
    "IQLBiddingAgent": {"model_dir": "checkpoints/iql", "checkpoint_file": "best.ckpt"},
    "BCBiddingAgent": {"model_dir": "checkpoints/bc", "checkpoint_file": "best.ckpt"},
    "GAVEBiddingAgent": {"model_dir": "checkpoints/gave", "checkpoint_file": "best.ckpt"},
    "DTBiddingAgent": {"model_dir": "checkpoints/dt", "checkpoint_file": "best.ckpt"},
    "GASBiddingAgent": {"model_dir": "checkpoints/gas", "backbone_model_dir": "checkpoints/dt", "checkpoint_file": "best.ckpt"},
}


def _format_cli_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(v) for v in value) + "]"
    if isinstance(value, Path):
        return str(value)
    return str(value)


def run_evaluation(
    agent_class: str,
    *,
    extra_flags: Iterable[str] | None = None,
    **kwargs: object,
) -> bool:
    """Run evaluation for a specific agent class"""
    logger.info("%s", "="*60)
    logger.info("Evaluating %s", agent_class)
    logger.info("%s", "="*60)
    sys.stdout.flush()  # Ensure output is shown immediately
    
    # Build command arguments
    cmd = [sys.executable, "evaluate_agents.py"]
    cmd.extend(["--agent.agent_class", agent_class])
    
    # Add model-specific parameters
    if agent_class in MODEL_BASED_AGENTS:
        model_config = MODEL_BASED_AGENTS[agent_class]
        cmd.extend(["--agent.model_dir", model_config["model_dir"]])
        cmd.extend(["--agent.checkpoint_file", model_config["checkpoint_file"]])
    
    # Add any additional kwargs
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", _format_cli_value(value)])

    if extra_flags:
        cmd.extend(list(extra_flags))
    
    logger.info("Running: %s", ' '.join(cmd))
    sys.stdout.flush()  # Ensure output is shown immediately
    
    try:
        # Use subprocess.run without capture_output to show real-time progress
        subprocess.run(cmd, check=True, text=True)
        logger.info("✅ Successfully evaluated %s", agent_class)
        logger.info("%s", "="*60)
        sys.stdout.flush()
    except subprocess.CalledProcessError as e:
        logger.error("❌ Failed to evaluate %s", agent_class)
        logger.error("Error: Command failed with return code %s", e.returncode)
        logger.info("%s", "="*60)
        sys.stdout.flush()
        return False
    except FileNotFoundError as e:
        logger.error("❌ Checkpoint not found for %s: %s", agent_class, e)
        logger.info("%s", "="*60)
        sys.stdout.flush()
        return False
    
    return True

@dataclass
class EvaluateSweepConfig:
    evaluation_base_seed: int = DEFAULT_SEED
    evaluation_num_seeds: int = DEFAULT_NUM_EVAL_SEEDS
    extra_flags: list[str] = field(default_factory=list)
    working_dir: str = str(Path(__file__).parent)


@draccus.wrap()
def main(cfg: EvaluateSweepConfig = EvaluateSweepConfig()) -> None:
    """Main function to evaluate all agents"""
    logger.info("Starting batch evaluation of all supported agents...")
    
    current_dir = Path(cfg.working_dir).resolve()

    # Change to the examples directory
    import os
    os.chdir(current_dir)
    
    successful = []
    failed = []
    
    total_agents = len(AGENT_CLASSES)
    eval_kwargs = {
        "evaluation.base_seed": cfg.evaluation_base_seed,
        "evaluation.num_seeds": cfg.evaluation_num_seeds,
    }

    for i, agent_class in enumerate(AGENT_CLASSES, 1):
        logger.info("🔄 Progress: %s/%s agents", i, total_agents)
        success = run_evaluation(agent_class, extra_flags=cfg.extra_flags, **eval_kwargs)
        if success:
            successful.append(agent_class)
        else:
            failed.append(agent_class)
    
    # Summary
    logger.info("%s", "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("%s", "="*60)
    logger.info("✅ Successful (%s): %s", len(successful), ', '.join(successful))
    if failed:
        logger.info("❌ Failed (%s): %s", len(failed), ', '.join(failed))
    logger.info("Total: %s/%s agents evaluated successfully", len(successful), len(AGENT_CLASSES))

if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    main()
