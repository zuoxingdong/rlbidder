import logging
from dataclasses import dataclass
from pathlib import Path

import draccus
from rlbidder.data.io import display_file_summary
from rlbidder.data.preprocess import csv_to_parquet_lazy
from rlbidder.utils import configure_logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class Config:
    raw_data_dir: Path = Path(__file__).parent.parent / "data" / "raw"
    """Directory containing raw CSV files"""
    remove_csv: bool = False
    """Whether to remove CSV files after conversion"""


@draccus.wrap()
def main(cfg: Config = Config()) -> None:
    csv_to_parquet_lazy(cfg.raw_data_dir, cfg.remove_csv)
    display_file_summary(cfg.raw_data_dir, "*.parquet", max_display=10)


if __name__ == "__main__":
    configure_logging()
    main()
