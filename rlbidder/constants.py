# ============================================================================
# General Configuration
# ============================================================================
DEFAULT_SEED = 42
DEFAULT_NUM_EVAL_SEEDS = 3

# ============================================================================
# Feature Configuration
# ============================================================================
NUM_TICKS = 48
CAMPAIGN_KEYS = [
    "deliveryPeriodIndex",
    "advertiserNumber",
    "advertiserCategoryIndex",
    "budget",
    "CPAConstraint"
]
STATE_COLS = [
    "timeleft",
    "bgtleft",
    "avg_bid_all",
    "avg_bid_last_3",
    "avg_leastWinningCost_all",
    "avg_pValue_all",
    "avg_conversionAction_all",
    "avg_xi_all",
    "avg_leastWinningCost_last_3",
    "avg_pValue_last_3",
    "avg_conversionAction_last_3",
    "avg_xi_last_3",
    "pValue_mean",
    "timeStepIndex_volume",
    "last_3_timeStepIndexs_volume",
    "historical_volume",
]
STATE_COLS_WITH_CPA_COMPLIANCE_RATIO = STATE_COLS + ["cpa_compliance_ratio"]
STATE_DIM = len(STATE_COLS_WITH_CPA_COMPLIANCE_RATIO)  # 17
HISTORY_FEATURE_DIM = 13  # from StepwiseAuctionHistory.get_state_features
ROLLING_WINDOW_LAST_K = 3

# ============================================================================
# Auction Simulation Configuration
# ============================================================================
MIN_REMAINING_BUDGET = 0.1
NUM_SLOTS_DEFAULT = 3
SLOT_EXPOSURE_COEFFS = (1.0, 0.8, 0.6)  # convert to np.array where needed
RESERVE_PV_PRICE = 0.01
MIN_NONZERO_BIDDERS = 2  # require at least this many nonzero bids for a valid auction

# ============================================================================
# DataFrame Column Names
# ============================================================================
TIME_STEP_COL = "timeStepIndex"
PV_INDEX_COL = "pvIndex"
STATE_ARRAY_COL = "state"
NEXT_STATE_ARRAY_COL = "next_state"
ACTION_COL = "action"
DONE_COL = "done"
REWARD_DENSE_COL = "reward_dense"
REWARD_SPARSE_COL = "reward_sparse"
CPA_COMPLIANCE_RATIO_COL = "cpa_compliance_ratio"

# =========================================================================
# File naming conventions
# =========================================================================
TRAIN_FILE_PREFIX = "train-traj-"
RAW_TRAJ_FILE_GLOB = "autoBidding_aigb*.parquet"
PERIOD_FILE_PREFIX = "period-"
EVAL_FILE_PREFIX = "eval-"
TRAIN_PERIOD_FILE_PREFIX = "train-period-"
PROCESSED_SUBDIR = "processed"
SCALED_TRANSITIONS_SUBDIR = "scaled_transitions"
