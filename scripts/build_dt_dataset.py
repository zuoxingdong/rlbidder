import logging
from dataclasses import dataclass, field
from pathlib import Path

import draccus
from rlbidder.constants import CAMPAIGN_KEYS
from rlbidder.data.preprocess import build_dt_trajectory_dataset
from rlbidder.utils import configure_logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class BuildConfig:
    data_dir: str = (Path(__file__).parent.parent / "data").as_posix()
    reward_type: str = "reward_dense"  # {"reward_sparse", "reward_dense"}
    use_scaled_reward: bool = True
    beta: float = 2.0


@dataclass
class Config:
    build: BuildConfig = field(default_factory=BuildConfig)


@draccus.wrap()
def main(cfg: Config = Config()) -> None:
    """Main function to build Decision Transformer trajectory dataset."""
    logger.info("Decision Transformer trajectory building process started.")
    
    # Define input and output paths
    data_dir = Path(cfg.build.data_dir)
    input_path = data_dir / "scaled_transitions" / "scaled_transitions.parquet"
    output_dir = data_dir / "processed"
    output_path = output_dir / "trajectories.npz"
    
    # Ensure output directory exists
    if not output_dir.exists():
        logger.info("Creating directory: %s", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info("Directory exists: %s", output_dir)
    
    # Check if input file exists
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        logger.info("Please run scale_transitions.py first to generate scaled transitions.")
        return
    
    logger.info("Using input file: %s", input_path)
    logger.info("Using reward type: %s", cfg.build.reward_type)
    logger.info("Using scaled reward: %s", cfg.build.use_scaled_reward)
    
    build_dt_trajectory_dataset(
        parquet_path=input_path,
        output_path=output_path,
        campaign_keys=CAMPAIGN_KEYS,
        reward_type=cfg.build.reward_type,
        use_scaled_reward=cfg.build.use_scaled_reward,
        beta=cfg.build.beta,
    )
    
    if output_path.exists():
        file_size = output_path.stat().st_size / (1024 * 1024)  # Size in MB
        logger.info("Output file created successfully: %s", output_path)
        logger.info("File size: %.2f MB", file_size)
    else:
        logger.error("Failed to create output file: %s", output_path)
        return
    
    logger.info("Decision Transformer trajectory building process finished.")


if __name__ == "__main__":
    configure_logging(level=logging.DEBUG)
    main()
