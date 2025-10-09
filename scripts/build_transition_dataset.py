import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import draccus
from rlbidder.constants import CAMPAIGN_KEYS, NUM_TICKS, STATE_COLS
from rlbidder.data.io import display_file_summary
from rlbidder.data.preprocess import create_training_data_for_all_periods, create_training_data_for_all_trajectories
from rlbidder.utils import configure_logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class Config:
    data_dir: Path = Path(__file__).parent.parent / "data"
    mode: Literal["period", "trajectory"] = "trajectory"
    fix_post_done: bool = True


@draccus.wrap()
def main(cfg: Config = Config()) -> None:
    if cfg.mode == "period":
        create_training_data_for_all_periods(
            cfg.data_dir / "raw",
            cfg.data_dir / "processed",
            CAMPAIGN_KEYS,
            STATE_COLS,
            timeStepIndexNum=NUM_TICKS,
        )
    elif cfg.mode == "trajectory":
        create_training_data_for_all_trajectories(
            cfg.data_dir / "raw",
            cfg.data_dir / "processed",
            campaign_keys=CAMPAIGN_KEYS,
            state_dim=len(STATE_COLS),
            fix_post_done=cfg.fix_post_done,
        )
    else:
        logger.error("Unknown mode: %s. Please choose 'period' or 'trajectory'.", cfg.mode)
        return

    processed_dir = cfg.data_dir / "processed"
    display_file_summary(processed_dir, "*.parquet", max_display=10)


if __name__ == "__main__":
    configure_logging(level=logging.DEBUG)
    main()
