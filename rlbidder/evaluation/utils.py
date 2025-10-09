import numpy as np
import polars as pl


def evaluate_score_with_constraint_penalty(
    value: np.ndarray,
    estimated_constraint: np.ndarray,
    target_constraint: np.ndarray,
    beta: float = 2,
) -> np.ndarray:
    """
    Calculates a penalized score by applying a constraint-based penalty to the input value.

    This function is typically used to adjust metrics such as conversions or clicks by penalizing
    the score when an estimated constraint (e.g., cost per acquisition) exceeds a specified target.
    The penalty is computed as a power function of the ratio between the target and estimated constraints,
    allowing for flexible control over penalty strength.

    Args:
        value (np.ndarray): The original score to be penalized (e.g., total conversions or clicks), shape (num_advertisers,).
        estimated_constraint (np.ndarray): The observed or estimated value of the constraint (e.g., actual CPA), shape (num_advertisers,).
        target_constraint (np.ndarray): The desired upper bound for the constraint (e.g., target CPA), shape (num_advertisers,).
        beta (float, optional): Exponent controlling the severity of the penalty when the constraint is exceeded. Defaults to 2.

    Returns:
        np.ndarray: The penalized score for each advertiser. If the estimated constraint exceeds the target, the score is reduced
        according to the penalty formula; otherwise, the original value is returned.
    """
    penalty = np.ones_like(value, dtype=float)
    mask = estimated_constraint > target_constraint
    coef = np.zeros_like(value, dtype=float)
    coef[mask] = target_constraint[mask] / (estimated_constraint[mask] + 1e-10)
    penalty[mask] = np.power(coef[mask], beta)
    return penalty * value


def calculate_agent_mean_score_above_quantile(df: pl.DataFrame, quantile: float = 0.5) -> pl.DataFrame:
    df_score = (
        df
        .explode("score_list")
        .select(
            agent=pl.col("agent_name"),
            score=pl.col("score_list"),
        )
        .group_by("agent")
        .agg(
            score=(
                pl.col("score")
                .filter(pl.col("score") > pl.col("score").quantile(quantile))
                .mean()
            ),
        )
    )
    return df_score
