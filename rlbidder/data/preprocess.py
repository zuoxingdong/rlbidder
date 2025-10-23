import logging
import re
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.base import TransformerMixin
from tqdm import tqdm

from rlbidder.constants import (
    CAMPAIGN_KEYS,
    ROLLING_WINDOW_LAST_K,
    STATE_ARRAY_COL,
    NEXT_STATE_ARRAY_COL,
    TIME_STEP_COL,
    PV_INDEX_COL,
    ACTION_COL,
    REWARD_SPARSE_COL,
    DONE_COL,
    CPA_COMPLIANCE_RATIO_COL,
    TRAIN_FILE_PREFIX,
    RAW_TRAJ_FILE_GLOB,
    PERIOD_FILE_PREFIX,
    EVAL_FILE_PREFIX,
    TRAIN_PERIOD_FILE_PREFIX,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def csv_to_parquet_lazy(data_dir: Path | str, remove_csv: bool = False) -> None:
    """
    Convert all CSV files in data_dir to Parquet format using Polars with lazy sink.
    Process files individually to avoid OOM issues.
    
    Args:
        data_dir (Path or str): Directory containing the CSV files.
        remove_csv (bool): Whether to remove CSV files after conversion. Defaults to False.
    """
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("*.csv"))
    
    for csv_path in csv_files:
        logger.info("Converting %s to %s", csv_path, csv_path.with_suffix('.parquet'))
        parquet_path = csv_path.with_suffix('.parquet')
        # Process each file individually to avoid memory issues
        (
            pl
            .scan_csv(csv_path)
            .sink_parquet(parquet_path)
        )
        if remove_csv:
            csv_path.unlink()
            logger.info("Removed %s", csv_path)


def stepwise_aggregate_campaigns_per_period_and_sink(
    data_dir: Path | str,
    output_dir: Path | str,
    campaign_keys: list[str] = CAMPAIGN_KEYS,
) -> None:
    """
    Scan data_dir for all period-*.parquet files, aggregate metrics, and write results
    to new parquet files named eval-{period}.parquet.
    Args:
        data_dir: directory containing period-*.parquet files
        output_dir: directory to save the aggregated parquet files
        campaign_keys: list of columns to group by
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    for parquet_file in data_dir.glob(f"{PERIOD_FILE_PREFIX}*.parquet"):
        output_path = Path(output_dir) / f"{EVAL_FILE_PREFIX}{parquet_file.name}"
        logger.info("Aggregating %s -> %s", parquet_file, output_path)
        (
            pl.scan_parquet(parquet_file)
            .sort(campaign_keys + [TIME_STEP_COL, PV_INDEX_COL])  # NOTE: sort pvIndex critical for correct agg order!
            .group_by(campaign_keys + [TIME_STEP_COL])
            .agg([
                pl.col("pValue"),
                pl.col("pValueSigma"),
                pl.col("leastWinningCost"),
                pl.col("bid"),
                pl.col("xi").alias("win_status"),
                (pl.col("cost") * pl.col("isExposed")).alias("cost"),
                (pl.col("conversionAction") * pl.col("isExposed")).alias("conversion"),
            ])
            .sort(campaign_keys + [TIME_STEP_COL])
            .sink_parquet(output_path)
        )
        
    logger.info("All files aggregated and sinked.")
       
        
def filter_campaign_data(
    lf: pl.LazyFrame,
    delivery_period_index: int,
    advertiser_numbers: list[int] | None = None,
) -> pl.LazyFrame:
    expr = pl.col("deliveryPeriodIndex") == delivery_period_index
    if advertiser_numbers is not None:
        expr = expr & pl.col("advertiserNumber").is_in(advertiser_numbers)
    return lf.filter(expr)


def aggregate_time_step_stats(
    lf: pl.LazyFrame,
    campaign_keys: list[str],
    timeStepIndexNum: int,
) -> pl.LazyFrame:
    lf_stats = (
        lf
        .with_columns(
            realAllCost=(
                (pl.col("isExposed") * pl.col("cost"))
                .sum()
                .over(campaign_keys)
            ),
            realAllConversion=(
                pl.col("conversionAction")
                .sum()
                .over(campaign_keys)
            ),
        )
        .group_by(campaign_keys + ["timeStepIndex"])
        .agg(
            realAllCost=pl.col("realAllCost").first(),
            realAllConversion=pl.col("realAllConversion").first(),
            bid_mean=pl.col("bid").mean(),
            leastWinningCost_mean=pl.col("leastWinningCost").mean(),
            conversionAction_mean=pl.col("conversionAction").mean(),
            xi_mean=pl.col("xi").mean(),
            pValue_mean=pl.col("pValue").mean(),
            timeStepIndex_volume=pl.len().cast(pl.Float64),
            remainingBudget=pl.col("remainingBudget").first(),
            action=(
                pl.when(pl.col("pValue").sum() > 0)
                .then(
                    pl.col("bid").sum() / pl.col("pValue").sum()
                )
                .otherwise(0)
            ),
            reward=(
                pl.col("conversionAction")
                .filter(pl.col("isExposed") == 1)
                .sum()
            ),
            reward_continuous=(
                pl.col("pValue")
                .filter(pl.col("isExposed") == 1)
                .sum()
            ),
            done=(
                pl.when(
                    (pl.col("timeStepIndex").first() == timeStepIndexNum - 1)
                    | (pl.col("isEnd").first() == 1)
                )
                .then(1)
                .otherwise(0)
            ),
            timeleft=((timeStepIndexNum - pl.col("timeStepIndex").first()) / timeStepIndexNum),
            bgtleft=(
                pl.when(pl.col("budget") > 0)
                .then(pl.col("remainingBudget") / pl.col("budget"))
                .otherwise(0)
                .first()
            ),
        )
    )
    return lf_stats


def add_rolling_features(lf: pl.LazyFrame, campaign_keys: list[str]) -> pl.LazyFrame:
    lf_rolling = (
        lf
        .sort(campaign_keys + ["timeStepIndex"])
        .with_columns(
            historical_volume=(
                pl.col("timeStepIndex_volume")
                .cum_sum()
                .shift(1, fill_value=0)
                .cast(pl.Float64)
                .over(campaign_keys)
            ),
            last_3_timeStepIndexs_volume=(
                pl.col("timeStepIndex_volume")
                .rolling_sum(window_size=3, min_samples=1)
                .shift(1, fill_value=0)
                .cast(pl.Float64)
                .over(campaign_keys)
            ),
            avg_bid_all=(
                pl.col("bid_mean")
                .cum_sum()
                .truediv(pl.arange(1, pl.len() + 1))
                .fill_nan(0).fill_null(0)  # Fill NaNs/Nulls after division, before shift
                .shift(1, fill_value=0)
                .over(campaign_keys)
            ),
            avg_leastWinningCost_all=(
                pl.col("leastWinningCost_mean")
                .cum_sum()
                .truediv(pl.arange(1, pl.len() + 1))
                .fill_nan(0).fill_null(0)  # Fill NaNs after division, before shift
                .shift(1, fill_value=0)
                .over(campaign_keys)
            ),
            avg_conversionAction_all=(
                pl.col("conversionAction_mean")
                .cum_sum()
                .truediv(pl.arange(1, pl.len() + 1))
                .fill_nan(0).fill_null(0)  # Fill NaNs after division, before shift
                .shift(1, fill_value=0)
                .over(campaign_keys)
            ),
            avg_xi_all=(
                pl.col("xi_mean")
                .cum_sum()
                .truediv(pl.arange(1, pl.len() + 1))
                .fill_nan(0).fill_null(0)  # Fill NaNs after division, before shift
                .shift(1, fill_value=0)
                .over(campaign_keys)
            ),
            avg_pValue_all=(
                pl.col("pValue_mean")
                .cum_sum()
                .truediv(pl.arange(1, pl.len() + 1))
                .fill_nan(0).fill_null(0)  # Fill NaNs after division, before shift
                .shift(1, fill_value=0)
                .over(campaign_keys)
            ),
            avg_bid_last_3=(
                pl.col("bid_mean")
                .rolling_mean(window_size=ROLLING_WINDOW_LAST_K, min_samples=1)
                .shift(1, fill_value=0)
                .over(campaign_keys)
            ),
            avg_leastWinningCost_last_3=(
                pl.col("leastWinningCost_mean")
                .rolling_mean(window_size=ROLLING_WINDOW_LAST_K, min_samples=1)
                .shift(1, fill_value=0)
                .over(campaign_keys)
            ),
            avg_conversionAction_last_3=(
                pl.col("conversionAction_mean")
                .rolling_mean(window_size=ROLLING_WINDOW_LAST_K, min_samples=1)
                .shift(1, fill_value=0)
                .over(campaign_keys)
            ),
            avg_xi_last_3=(
                pl.col("xi_mean")
                .rolling_mean(window_size=ROLLING_WINDOW_LAST_K, min_samples=1)
                .shift(1, fill_value=0)
                .over(campaign_keys)
            ),
            avg_pValue_last_3=(
                pl.col("pValue_mean")
                .rolling_mean(window_size=ROLLING_WINDOW_LAST_K, min_samples=1)
                .shift(1, fill_value=0)
                .over(campaign_keys)
            ),
        )
    )
    return lf_rolling


def build_state_column(lf: pl.LazyFrame, state_cols: list[str]) -> pl.LazyFrame:
    return lf.with_columns(
        **{STATE_ARRAY_COL: pl.concat_arr([pl.col(col) for col in state_cols])}
    )


def build_next_state_column(
    lf: pl.LazyFrame,
    state_cols: list[str],
    campaign_keys: list[str],
) -> pl.LazyFrame:
    return lf.with_columns(
        **{NEXT_STATE_ARRAY_COL: pl.concat_arr([
            pl.col(col).shift(-1, fill_value=0).over(campaign_keys)
            for col in state_cols
        ])}
    )
    
    
def get_delivery_period_indices(data_dir: Path | str, prefix: str = PERIOD_FILE_PREFIX) -> list[int]:
    data_dir = Path(data_dir)
    pattern = re.compile(rf"{prefix}(\d+)\.parquet")
    indices = []
    for file in data_dir.glob(f"{prefix}*.parquet"):
        match = pattern.match(file.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def create_training_data_for_all_periods(
    raw_dir: str | Path,
    processed_dir: str | Path,
    campaign_keys: list[str],
    state_cols: list[str],
    timeStepIndexNum: int,
) -> None:
    """
    Process all period parquet files in raw_dir and write processed training data to processed_dir.
    """
    # TODO: currently polars lazy sinks do not work, so we workaround with lazy=False
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    period_indices = get_delivery_period_indices(raw_dir, prefix=PERIOD_FILE_PREFIX)
    
    for period_idx in period_indices:
        parquet_file = raw_dir / f"{PERIOD_FILE_PREFIX}{period_idx}.parquet"
        out_path = processed_dir / f"{TRAIN_PERIOD_FILE_PREFIX}{period_idx}.parquet"

        logger.info("Processing file: %s -> Writing to: %s", parquet_file, out_path)
        
        lf = pl.scan_parquet(parquet_file)
        lf = filter_campaign_data(lf, period_idx)
        lf = aggregate_time_step_stats(lf, campaign_keys, timeStepIndexNum)
        lf = add_rolling_features(lf, campaign_keys)

        # Compute per-timestep cost and CPA compliance, then append ratio to state/next_state
        lf = (
            lf
            .with_columns(
                # next bgtleft within campaign for cost computation
                next_bgtleft=pl.col("bgtleft").shift(-1).over(campaign_keys)
            )
            .with_columns(
                cost=(
                    pl.when(pl.col("done") == 1)
                    .then(
                        # final step: reconcile with total cost
                        pl.col("realAllCost") - (1 - pl.col("bgtleft")) * pl.col("budget")
                    )
                    .otherwise(
                        # intermediate steps: delta of budget-left
                        (pl.col("bgtleft") - pl.col("next_bgtleft")) * pl.col("budget")
                    )
                ),
                # align naming with trajectory pipeline
                reward_sparse=pl.col("reward"),
                reward_dense=pl.col("reward_continuous"),
            )
            .drop("next_bgtleft")
        )

        lf = build_state_column(lf, state_cols)
        lf = build_next_state_column(lf, state_cols, campaign_keys)
        lf = lf.select(campaign_keys + ["timeStepIndex", "state", "reward_sparse", "reward_dense", "cost", "action", "next_state", "done"])

        lf = (
            calculate_cpa_compliance_metrics(lf)
            .with_columns(
                state=pl.concat_arr(
                    "state",
                    (
                        pl.col("cpa_compliance_ratio")
                        .shift(1, fill_value=0)
                        .over(["deliveryPeriodIndex", "advertiserNumber"], order_by="timeStepIndex", descending=False)
                    ),
                ),
                next_state=pl.concat_arr("next_state", "cpa_compliance_ratio"),
            )
        )
        
        lf_final = lf.sort(campaign_keys + [TIME_STEP_COL])
        lf_final.sink_parquet(out_path, lazy=False, mkdir=True)

    logger.info("All files processed.")


def parse_tuple_str_to_array(df: pl.DataFrame, col_name: str, array_len: int) -> pl.DataFrame:
    """
    Convert a string column of tuple-like values to a fixed-length float array column.

    Args:
        df (pl.DataFrame): Input DataFrame.
        col_name (str): Name of the column to convert.
        array_len (int): Expected length of the array.

    Returns:
        pl.DataFrame: DataFrame with the converted column.
    """
    return df.with_columns(
        pl.col(col_name)
        .str.strip_chars_start("(")
        .str.strip_chars_end(")")
        .str.split(",")
        .list.eval(pl.element().str.strip_chars().cast(pl.Float64))
        .cast(pl.Array(pl.Float64, array_len))
        .alias(col_name)
    )


def shift_truncate_post_done(
    lf: pl.LazyFrame, 
    campaign_keys: list[str]
) -> pl.LazyFrame:
    """
    For each campaign trajectory, shift the 'done' column to refer to the next state,
    then truncate all rows after the first occurrence of 'done'==1 (i.e., episode end).
    This is useful for RL training data preparation to ensure no transitions after terminal states.

    Args:
        lf (pl.LazyFrame): Input LazyFrame containing RL trajectory data.
        campaign_keys (list[str]): Columns to group by (e.g., campaign identifiers).

    Returns:
        pl.LazyFrame: Processed LazyFrame with post-done rows removed.
    """
    lf = (
        lf
        .with_columns(
            done_cumcount=(
                pl.col("done")
                .cum_sum()
                .over(
                    campaign_keys, 
                    order_by="timeStepIndex", 
                    descending=False,
                )
            )
        )
        .filter(pl.col("done_cumcount") <= 1)
        .drop("done_cumcount")
    )
    return lf


def calculate_cpa_compliance_metrics(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate CPA compliance metrics for trajectory data.
    
    Args:
        lf: LazyFrame with columns 'cost', 'reward_sparse', 'CPAConstraint',
            'deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex'
    
    Returns:
        LazyFrame with added columns: cum_cost, cum_conversions, ecpa, cpa_compliance_ratio
    """
    return (
        lf
        .with_columns(
            cum_cost=(
                pl.col("cost")
                .cum_sum()
                .over(["deliveryPeriodIndex", "advertiserNumber"], order_by="timeStepIndex", descending=False)
            ),
            cum_conversions=(
                pl.col("reward_sparse")
                .cum_sum()
                .over(["deliveryPeriodIndex", "advertiserNumber"], order_by="timeStepIndex", descending=False)
            )
        )
        .with_columns(
            ecpa=(
                pl.col("cum_cost") / pl.col("cum_conversions")
            )
        )
        .with_columns(
            # NOTE: this reflect what happens AFTER the action in current timestep, so it becomes a part of state from second timestep
            cpa_compliance_ratio=(
                (pl.col("CPAConstraint") / pl.col("ecpa")).fill_nan(0)
            )
        )
    )


def create_training_data_for_all_trajectories(
    raw_dir: str | Path,
    processed_dir: str | Path,
    campaign_keys: list[str],
    state_dim: int = 16,
    fix_post_done: bool = True,
) -> None:
    # TODO: currently polars lazy sinks do not work, so we workaround with lazy=False
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for idx, parquet_file in enumerate(raw_dir.glob(RAW_TRAJ_FILE_GLOB)):
        out_path = processed_dir / f"{TRAIN_FILE_PREFIX}{idx}.parquet"
        logger.info("Processing file: %s -> Writing to: %s", parquet_file, out_path)
        lf = pl.scan_parquet(parquet_file)
        # Rename reward columns before selecting
        lf = lf.rename({
            "reward": "reward_sparse",
            "reward_continuous": "reward_dense",
        })

        lf = parse_tuple_str_to_array(lf, "state", state_dim)
        lf = parse_tuple_str_to_array(lf, "next_state", state_dim)

        # Create cost column based on done status
        lf = lf.with_columns(
            cost=pl.when(pl.col("done") == 1)
            .then(
                # For done=1: cost_t = realAllCost - (1 - budget_left) * budget
                # Assuming state[1] is budget_left ratio, so (1 - state[1]) gives spent ratio
                pl.col("realAllCost") - (1 - pl.col("state")).arr.get(1) * pl.col("budget")
            )
            .otherwise(
                # For done=0: cost_t = (budget_left_current - budget_left_next) * budget
                (pl.col("state") - pl.col("next_state")).arr.get(1) * pl.col("budget")
            )
        )

        lf = lf.select(campaign_keys + ["timeStepIndex", "state", "reward_sparse", "reward_dense", "cost", "action", "next_state", "done"])
        
        # For done=1, fill next_state with state (original data is null)
        lf = lf.with_columns(
            next_state=(
                pl.when(pl.col("done") == 1)
                .then(pl.col("state"))
                .otherwise(pl.col("next_state"))
            )
        )
        if fix_post_done:  # original data is not standard RL format, avoid spurious transitions
            lf = shift_truncate_post_done(lf, campaign_keys)
        # else:  # keep original (WARN: ~12% spurious transitions)

        lf = (
            calculate_cpa_compliance_metrics(lf)
            .with_columns(
                # NOTE: this reflect what happens AFTER the action in current timestep, so it becomes a part of state from second timestep
                state=pl.concat_arr(
                    "state", 
                    (
                        pl.col("cpa_compliance_ratio")
                        .shift(1, fill_value=0)  # NOTE: initial cpa_compliance_ratio is always 0 as no action has been taken yet
                        .over(["deliveryPeriodIndex", "advertiserNumber"], order_by="timeStepIndex", descending=False)
                    )
                ),
                next_state=pl.concat_arr("next_state", "cpa_compliance_ratio"),
            )
        )
        
        lf_final = lf.sort(campaign_keys + ["timeStepIndex"])
        lf_final.sink_parquet(out_path, lazy=False, mkdir=True)
    
    logger.info("All files processed.")


def fit_and_scale_offline_rl_transitions(
    lf: pl.LazyFrame,
    campaign_keys: list[str],
    time_step_col: str,
    state_scaler: TransformerMixin,
    action_scaler: TransformerMixin,
    reward_dense_scaler: TransformerMixin,
    reward_sparse_scaler: TransformerMixin,
    save_dir: Path | str,
) -> pl.DataFrame:
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    
    df = (
        lf
        .select(
            campaign_keys + [time_step_col, "state", "action", "reward_dense", "reward_sparse", "next_state", "done", "cpa_compliance_ratio"]
        )
        .collect()
    )

    logger.debug("Fitting and transforming state...")
    scaled_state = state_scaler.fit_transform(df["state"].to_numpy())
    logger.debug("State scaling done. Shape: %s", scaled_state.shape)

    logger.debug("Transforming next_state...")
    scaled_next_state = state_scaler.transform(df["next_state"].to_numpy())
    logger.debug("Next state scaling done. Shape: %s", scaled_next_state.shape)

    logger.debug("Fitting and transforming action...")
    scaled_action = action_scaler.fit_transform(
        df[["action"]].to_numpy()  # Ensure action is 2D for scaler
    ).squeeze(1)
    logger.debug("Action scaling done. Shape: %s", scaled_action.shape)

    logger.debug("Fitting and transforming reward_dense...")
    scaled_reward_dense = reward_dense_scaler.fit_transform(
        df[["reward_dense"]]  # Ensure reward_dense is 2D for scaler
    ).squeeze(1)
    logger.debug("Reward dense scaling done. Shape: %s", scaled_reward_dense.shape)

    logger.debug("Fitting and transforming reward_sparse...")
    scaled_reward_sparse = reward_sparse_scaler.fit_transform(
        df[["reward_sparse"]]  # Ensure reward_sparse is 2D for scaler
    ).squeeze(1)
    logger.debug("Reward sparse scaling done. Shape: %s", scaled_reward_sparse.shape)

    # Merge campaign/time columns and scaled data horizontally
    df_campaign = df[campaign_keys + [time_step_col]]
    df_scaled = pl.concat(
        [
            df_campaign, 
            pl.from_dict({
                "state": scaled_state,
                "action": scaled_action,
                "reward_dense": df["reward_dense"],
                "reward_dense_scaled": scaled_reward_dense,
                "reward_sparse": df["reward_sparse"],
                "reward_sparse_scaled": scaled_reward_sparse,
                "next_state": scaled_next_state,
                "done": df["done"],
                "cpa_compliance_ratio": df["cpa_compliance_ratio"],
            })
        ],
        how="horizontal",
    )

    logger.info("Saving fitted scalers to: %s", save_dir)
    logger.debug("State scaler: %s", state_scaler)
    logger.debug("Action scaler: %s", action_scaler)
    logger.debug("Reward dense scaler: %s", reward_dense_scaler)
    logger.debug("Reward sparse scaler: %s", reward_sparse_scaler)
    joblib.dump(state_scaler, save_dir / "state_scaler.joblib")
    joblib.dump(action_scaler, save_dir / "action_scaler.joblib")
    joblib.dump(reward_dense_scaler, save_dir / "reward_dense_scaler.joblib")
    joblib.dump(reward_sparse_scaler, save_dir / "reward_sparse_scaler.joblib")

    df_scaled.write_parquet(save_dir / "scaled_transitions.parquet")
    logger.info("Scaled transitions saved to: %s", save_dir / "scaled_transitions.parquet")
    return df_scaled


def compute_rtgs(rewards: np.ndarray) -> np.ndarray:
    """Compute returns-to-go (RTGs) for a trajectory.
    
    Returns-to-go represents the cumulative reward from each timestep to the end
    of the trajectory. For a trajectory with rewards [r_0, r_1, ..., r_T], the
    RTG at timestep t is: RTG_t = sum(r_i for i in range(t, T+1)).
    
    Args:
        rewards: Array of reward values for each timestep in the trajectory.
    
    Returns:
        Array of returns-to-go, where each element represents the cumulative
        return from that timestep to the end of the trajectory.
    
    Example:
        >>> rewards = np.array([1.0, 2.0, 3.0])
        >>> compute_rtgs(rewards)
        array([6.0, 5.0, 3.0])
    """
    return np.cumsum(rewards[::-1])[::-1]


def compute_cpa_penalized_rtgs(rewards: np.ndarray, penalties: np.ndarray) -> np.ndarray:
    """
    Compute penalized returns-to-go (RTGs) by incorporating CPA compliance penalties.
    
    This function calculates cumulative returns from each timestep to the end of the trajectory,
    where rewards are scaled by penalty factors that reflect CPA compliance. The penalty
    factor typically ranges from 0 to 1, where values closer to 1 indicate better CPA
    compliance and result in less penalization of rewards.
    
    Args:
        rewards (np.ndarray): Array of reward values for each timestep in the trajectory.
        penalties (np.ndarray): Array of penalty factors (typically 0-1) for each timestep,
            where higher values indicate better CPA compliance and less penalization.
    
    Returns:
        np.ndarray: Array of penalized returns-to-go, where each element represents the
            cumulative penalized return from that timestep to the end of the trajectory.
    
    Note:
        This is equivalent to:
        G = (penalties * rewards).cumsum()
        return np.concatenate([[G[-1]], G[-1] - G[:-1]])
        
        The implementation uses reverse cumsum for computational efficiency.
    """
    return np.cumsum((penalties * rewards)[::-1])[::-1]


def build_dt_trajectory_dataset(
    parquet_path: Path | str,
    output_path: Path | str,
    campaign_keys: list[str],
    reward_type: str = "reward_sparse",
    use_scaled_reward: bool = False,
    beta: float = 2.0,
) -> None:
    """Build Decision Transformer trajectory dataset from parquet files.
    
    Args:
        parquet_path: Path to input parquet files
        output_path: Path to save output npz file
        campaign_keys: List of campaign key columns
        reward_type: Type of reward column to use (default: "reward_sparse")
        use_scaled_reward: Whether to use the scaled reward column (default: False).
    """
    
    logger.info("Scanning parquet files from: %s", parquet_path)
    lf = pl.scan_parquet(parquet_path)
    logger.info("Sorting and grouping data...")
    reward_column = f"{reward_type}_scaled" if use_scaled_reward else reward_type
    states, actions, rewards_sparse, rewards, dones, traj_lens, penalties, cpa_compliance_ratios = (
        lf
        .sort(campaign_keys + [TIME_STEP_COL])
        .group_by(campaign_keys)
        .agg([
            pl.col(STATE_ARRAY_COL),
            pl.col(ACTION_COL),
            pl.col(REWARD_SPARSE_COL),
            pl.col(reward_column).alias("reward"),
            pl.col(DONE_COL),
            pl.len().alias("traj_len"),
            (
                pl.when(pl.col(CPA_COMPLIANCE_RATIO_COL) > 0)
                .then((pl.col(CPA_COMPLIANCE_RATIO_COL).clip(0, 1).pow(beta)))
                .otherwise(1.0)
                .alias("penalty")
            ),
            pl.col(CPA_COMPLIANCE_RATIO_COL),
        ])
        .select([STATE_ARRAY_COL, ACTION_COL, REWARD_SPARSE_COL, "reward", DONE_COL, "traj_len", "penalty", CPA_COMPLIANCE_RATIO_COL])
        .collect()
        .to_numpy()
        .T
    )

    logger.info("Calculating state mean and std...")
    all_states = (
        lf
        .select(STATE_ARRAY_COL)
        .collect()
        .to_series()
        .to_numpy()
    )
    state_mean = all_states.mean(axis=0)
    state_std = all_states.std(axis=0)

    logger.info("Computing trajectory-level returns from %s for statistics...", reward_column)
    traj_returns = np.array([r.sum() for r in tqdm(rewards, total=len(rewards))])
    logger.info(
        "Traj return statistics:\n%s",
        pl.Series(traj_returns.flatten()).describe(percentiles=(0.05, 0.25, 0.5, 0.75, 0.95)),
    )

    logger.info("Computing CPA-penalized returns-to-go (rtgs) for each trajectory...")
    rtgs = [
        compute_cpa_penalized_rtgs(r, p)
        for r, p in tqdm(zip(rewards_sparse, penalties), total=len(rewards_sparse))
    ]
    logger.info(
        "CPA-penalized RTG statistics:\n%s",
        pl.Series([max(rtg) for rtg in rtgs]).describe(percentiles=(0.05, 0.25, 0.5, 0.75, 0.95)),
    )

    logger.info("Saving processed trajectories to: %s", output_path)
    np.savez(
        output_path,
        states=np.array(list(states), dtype=object),
        actions=np.array(list(actions), dtype=object),
        rewards=np.array(list(rewards), dtype=object),
        dones=np.array(list(dones), dtype=object),
        rtgs=np.array(list(rtgs), dtype=object),
        traj_lens=np.array(list(traj_lens), dtype=object),
        state_mean=state_mean,
        state_std=state_std,
        penalties=np.array(list(penalties), dtype=object),
        cpa_compliance_ratios=np.array(list(cpa_compliance_ratios), dtype=object),
    )
    logger.info("Successfully saved trajectories to: %s", output_path)
