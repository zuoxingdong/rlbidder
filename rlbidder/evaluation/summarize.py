import polars as pl


def summarize_agent_scores(
    df_agent_summaries: pl.DataFrame, 
    group_by_period: bool = False,
    return_intermediate: bool = False,
    use_best_advertiser: bool = False,
    filter_env_agents: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, pl.DataFrame]:
    """
    Summarize agent scores by finding the best performing advertiser per rotation
    and then averaging across all rotations for each agent.

    Args:
        df_agent_summaries: Polars DataFrame with agent summaries containing
            conversion_list, score_list, budget_spent_ratio_list, and cpa_exceed_ratio_list.
        group_by_period: Whether to include period in the final grouping.
        return_intermediate: Whether to return the intermediate DataFrame with 
            best advertiser metrics per rotation as well.
        use_best_advertiser: If True, use the best performing advertiser per rotation.
            If False, use the mean across all advertisers per rotation.
        filter_env_agents: If True, filter out agents with names containing "-env".

    Returns:
        Summary DataFrame with mean metrics per agent (and period if specified).
        If return_intermediate=True, returns tuple of (summary, intermediate_df).
    """
    if filter_env_agents:
        df_agent_summaries = df_agent_summaries.filter(~pl.col("agent_name").str.contains("-env"))
    
    # Find best performing advertiser per (period, seed, agent_name, rotate_index)
    df_per_rotation = (
        df_agent_summaries
        .with_columns(
            # Find index of advertiser with highest conversion for this rotation
            best_advertiser_idx=pl.col("conversion_list").list.arg_max(),
        )
        .select(
            pl.col("agent_name"),
            pl.col("period"), 
            pl.col("rotate_index"),
            pl.col("seed"),
            # Extract metrics for the best performing advertiser
            best_score=pl.col("score_list").list.get(pl.col("best_advertiser_idx")),
            best_conversion=pl.col("conversion_list").list.get(pl.col("best_advertiser_idx")),
            best_budget_spent_ratio=pl.col("budget_spent_ratio_list").list.get(pl.col("best_advertiser_idx")),
            best_cpa_exceed_ratio=pl.col("cpa_exceed_ratio_list").list.get(pl.col("best_advertiser_idx")),
            mean_score=pl.col("score_list").list.mean(),
            mean_conversion=pl.col("conversion_list").list.mean(),
            mean_budget_spent_ratio=pl.col("budget_spent_ratio_list").list.mean(),
            mean_cpa_exceed_ratio=pl.col("cpa_exceed_ratio_list").list.mean(),
        )
    )

    # Define grouping columns
    group_cols = ["agent_name"]
    if group_by_period:
        group_cols = ["period", "agent_name"]
        
    # Choose metrics based on use_best_advertiser flag
    prefix = "best" if use_best_advertiser else "mean"
    
    # Aggregate across all rotations
    df_summary = (
        df_per_rotation
        .group_by(group_cols)
        .agg(
            mean_returns=pl.col(f"{prefix}_conversion").mean(),
            mean_score=pl.col(f"{prefix}_score").mean(), 
            mean_budget_spent_ratio=pl.col(f"{prefix}_budget_spent_ratio").mean(),
            mean_cpa_exceed_ratio=pl.col(f"{prefix}_cpa_exceed_ratio").drop_nans().mean(),
        )
        .sort(group_cols)
    )

    if return_intermediate:
        return df_summary, df_per_rotation
    return df_summary
