import logging
from dataclasses import dataclass
from pathlib import Path

import draccus
from rlbidder.constants import CAMPAIGN_KEYS
from rlbidder.data.io import display_file_summary
from rlbidder.data.preprocess import stepwise_aggregate_campaigns_per_period_and_sink
from rlbidder.utils import configure_logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class Config:
    data_dir: Path = Path(__file__).parent.parent / "data"


@draccus.wrap()
def main(cfg: Config = Config()) -> None:
    stepwise_aggregate_campaigns_per_period_and_sink(
        cfg.data_dir / "raw",
        cfg.data_dir / "processed",
        campaign_keys=CAMPAIGN_KEYS,
    )
    processed_dir = cfg.data_dir / "processed"
    display_file_summary(processed_dir, "eval-period-*.parquet", max_display=10)


if __name__ == "__main__":
    configure_logging(level=logging.DEBUG)
    main()

