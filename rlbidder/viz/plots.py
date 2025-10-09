import itertools
from typing import Any, Dict

import numpy as np
import plotly.colors
import plotly.express as px
import plotly.graph_objs as go
import polars as pl
from plotly.graph_objs import Table as _Table
from plotly.subplots import make_subplots

from rlbidder.evaluation.history import StepwiseAuctionHistory
from rlbidder.viz.market_metrics import (
    _nan_rolling_mean,
    bid_aggressiveness_mean_per_tick,
    bid_gap_ratio_per_tick,
    compute_budget_pacing_line,
    compute_campaign_metrics,
    gini_cumulative_by_wins,
    hhi_cumulative_by_wins,
    participation_rate_per_tick,
    realized_volatility,
    top_of_book_pressure_per_tick,
    winner_set_jaccard_per_tick,
)


def make_campaign_summary_table(metrics: Dict[str, Any], budget: float, cpa_target: float) -> _Table:
    """
    Build a compact summary table of key campaign outcomes.

    Args:
        metrics: Mapping of metric names produced by `compute_campaign_metrics`.
        budget: Total spend budget for the agent.
        cpa_target: Target cost per acquisition.

    Returns:
        Plotly Table trace to be added to a figure.
    """
    cumulative_spend = metrics["cumulative_spend"]
    cumulative_conversions = metrics["cumulative_conversions"]
    cumulative_cpa = metrics["cumulative_cpa"]
    win_rates = metrics["win_rates"]
    avg_conversion_rate_per_win = metrics["avg_conversion_rate_per_win"]
    final_cpa = cumulative_cpa[-1] if cumulative_conversions[-1] > 0 else np.nan
    return _Table(
        header=dict(values=["Metric", "Value"]),
        cells=dict(values=[
            [
                "Total Spend",
                "Total Conversions",
                "Final/Target CPA",
                "% Budget Spent",
                "Win Rate",
                "Conversion Rate (per win)"
            ],
            [
                f"{cumulative_spend[-1]:.2f}",
                f"{cumulative_conversions[-1]:.2f}",
                f"{final_cpa:.2f} / {cpa_target:.2f}" if not np.isnan(final_cpa) else f"N/A / {cpa_target:.2f}",
                f"{100 * cumulative_spend[-1] / budget:.1f}%",
                f"{np.mean(win_rates):.2f}",
                f"{avg_conversion_rate_per_win:.4f}" if not np.isnan(avg_conversion_rate_per_win) else "N/A"
            ]
        ])
    )


def add_campaign_dashboard_traces(
    fig: go.Figure,
    metrics: Dict[str, Any],
    budget: float,
    cpa_target: float,
    pacing_line: np.ndarray,
) -> None:
    """
    Add all traces used by the campaign dashboard figure in a single place.

    Args:
        fig: Target Plotly figure produced by `make_subplots`.
        metrics: Metrics computed by `compute_campaign_metrics`.
        budget: Total budget for the selected agent.
        cpa_target: Target cost per acquisition.
        pacing_line: Cumulative budget pacing line across timesteps.
    """
    timesteps = metrics["timesteps"]

    # Row 1: Budget and pacing
    fig.add_trace(
        go.Scatter(
            x=timesteps, 
            y=metrics["cumulative_spend"], 
            mode='lines', 
            name='Cumulative Spend',
            line=dict(color='steelblue'),
        ), 
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=timesteps, 
            y=[budget] * len(timesteps), 
            mode='lines', 
            name='Budget', 
            line=dict(dash='dash', color='red')
        ), 
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=timesteps, 
            y=pacing_line, 
            mode='lines', 
            name='Budget Pacing', 
            line=dict(dash='dot', color='orange')
        ), 
        row=1, col=1
    )
    
    # Row 1, col 2: Per-step CPA overlaid with Accumulated CPA
    fig.add_trace(
        go.Scatter(
            x=timesteps, 
            y=metrics["per_step_cpa"], 
            mode='markers', 
            name='Per-step CPA (scatter)', 
            marker=dict(size=6, opacity=0.45, color='royalblue')
        ), 
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=timesteps, 
            y=metrics["per_step_cpa_rolling"], 
            mode='lines', 
            name='Per-step CPA (rolling avg)', 
            line=dict(color='royalblue', dash='dot')
        ), 
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=timesteps, 
            y=[cpa_target] * len(timesteps), 
            mode='lines', 
            name='Target CPA', 
            line=dict(dash='dash', color='red')
        ), 
        row=1, col=2
    )
    
    # Accumulated CPA and single target/band on shared y-axis in Row 1, col 2
    upper_band = cpa_target * 1.1
    lower_band = cpa_target * 0.9
    fig.add_trace(
        go.Scatter(
            x=timesteps, 
            y=metrics["cumulative_cpa"], 
            mode='lines', 
            name='Accumulated CPA',
            line=dict(color='crimson'),
        ), 
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([timesteps, timesteps[::-1]]),
            y=np.concatenate([
                np.full_like(timesteps, upper_band), 
                np.full_like(timesteps, lower_band)[::-1]
            ]),
            fill='toself', 
            fillcolor='rgba(255, 0, 0, 0.1)', 
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", 
            showlegend=True, 
            name="+/-10% CPA Range"
        ), 
        row=1, col=2
    )
    
    # Row 2, col 1: Bid and auction price trends
    fig.add_trace(
        go.Scatter(
            x=timesteps, 
            y=metrics["avg_bids"], 
            mode='lines', 
            name='Average Bid', 
            line=dict(color='dodgerblue')
        ), 
        row=2, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=timesteps, 
            y=metrics["avg_bids"] + metrics["std_bids"], 
            mode='lines', 
            name='Bid +1std', 
            line=dict(color='dodgerblue', dash='dot'), 
            showlegend=False
        ), 
        row=2, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=timesteps, 
            y=metrics["avg_bids"] - metrics["std_bids"], 
            mode='lines', 
            name='Bid -1std', 
            line=dict(color='dodgerblue', dash='dot'), 
            fill='tonexty', 
            fillcolor='rgba(30,144,255,0.10)', 
            showlegend=False
        ), 
        row=2, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=timesteps, 
            y=metrics["avg_auction_price"], 
            mode='lines', 
            name='Avg Auction Price', 
            line=dict(color='darkorange')
        ), 
        row=2, col=1, secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=timesteps, 
            y=metrics["avg_auction_price"] + metrics["std_auction_price"], 
            mode='lines', 
            name='Price +1std', 
            line=dict(color='darkorange', dash='dot'), 
            showlegend=False
        ), 
        row=2, col=1, secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=timesteps, 
            y=metrics["avg_auction_price"] - metrics["std_auction_price"], 
            mode='lines', 
            name='Price -1std', 
            line=dict(color='darkorange', dash='dot'), 
            fill='tonexty', 
            fillcolor='rgba(255,140,0,0.10)', 
            showlegend=False
        ), 
        row=2, col=1, secondary_y=True
    )
    
    # Row 2, col 2: Win Rate (primary) overlaid with Value Distribution (secondary)
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=metrics["win_rates"],
            mode='lines',
            name='Win Rate',
            line=dict(color='seagreen')
        ),
        row=2, col=2, secondary_y=False
    )
    fig.add_trace(
        go.Bar(
            x=timesteps,
            y=metrics["values"],
            name='Value Distribution',
            marker_color='orange',
            opacity=0.55,
            width=0.8,
            marker_line_width=0
        ),
        row=2, col=2, secondary_y=True
    )

    # Row 3, col 1: Cumulative Conversions (primary) overlaid with Traffic Distribution (secondary)
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=metrics["cumulative_conversions"],
            mode='lines+markers',
            name='Cumulative Conversions',
            line=dict(color='teal'),
            marker=dict(symbol='square')
        ),
        row=3, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Bar(
            x=timesteps,
            y=metrics["traffic"],
            name='Traffic Volume',
            marker_color='mediumpurple',
            opacity=0.55,
            width=0.8,
            marker_line_width=0
        ),
        row=3, col=1, secondary_y=True
    )

    # Row 3, col 2: Alpha mean with ±1σ band (primary) overlaid with ENC, Pressure, and Bid Premium (secondary)
    alpha_mean = metrics["avg_alphas"]
    alpha_std = metrics["std_alphas"]
    upper = alpha_mean + alpha_std
    lower = alpha_mean - alpha_std

    fig.add_trace(
        go.Scatter(
            x=timesteps, y=upper, mode='lines', line=dict(width=0),
            name=None, hoverinfo='skip', showlegend=False
        ), row=3, col=2, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=timesteps, y=lower, mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(65,105,225,0.18)',
            name='Alpha ±1σ', hoverinfo='skip'
        ), row=3, col=2, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=timesteps, y=alpha_mean, mode='lines',
            name='Average Alpha', line=dict(color='royalblue', width=2.2),
        ), row=3, col=2, secondary_y=False
    )

    # Overlays on secondary axis
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=metrics["bid_premium_pct"],
            mode='lines',
            name='Bid Premium (%)',
            line=dict(color='gray')
        ), row=3, col=2, secondary_y=True
    )


def plot_pairwise_scatterplots(metrics: Dict[str, Any], metric_names: list[str]) -> go.Figure:
    """
    Plot pairwise scatterplots for selected metrics.

    Args:
        metrics: Mapping of metric arrays keyed by name.
        metric_names: Subset of metric names to plot pairwise.
    
    Returns:
        Plotly Figure with all pairwise scatterplots.
    """
    scatter_df = pl.DataFrame({k: metrics[k] for k in metric_names}).to_pandas()
    pairs = list(itertools.combinations(metric_names, 2))
    n_pairs = len(pairs)
    n_cols = 3
    n_rows = int(np.ceil(n_pairs / n_cols))
    fig_pairs = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f"{x} vs {y}" for x, y in pairs])
    for idx, (x, y) in enumerate(pairs):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        fig_pairs.add_trace(
            go.Scatter(
                x=scatter_df[x],
                y=scatter_df[y],
                mode='markers',
                marker=dict(size=5, opacity=0.6),
                name=f"{x} vs {y}",
                showlegend=False
            ),
            row=row, col=col
        )
        fig_pairs.update_xaxes(title_text=x, row=row, col=col)
        fig_pairs.update_yaxes(title_text=y, row=row, col=col)
    fig_pairs.update_layout(
        height=300 * n_rows,
        width=350 * n_cols,
        title_text="Pairwise Scatterplots of Key Auction Metrics (Unique Pairs)"
    )
    return fig_pairs


def plot_campaign_simulation_metrics(
    auction_history: StepwiseAuctionHistory,
    agent_idx: int,
    budget: float | None = None,
    cpa_target: float | None = None,
    model_name: str | None = None,
)-> tuple[go.Figure, go.Figure]:
    """
    Plot a multi-panel dashboard summarizing a single agent's campaign.

    If `budget`, `cpa_target`, or `model_name` are None, they are inferred from
    the `auction_history` metadata.

    Args:
        auction_history: Stepwise auction rollouts and per-tick aggregates.
        agent_idx: Index of the agent to visualize.
        budget: Total budget; inferred if None.
        cpa_target: Target CPA; inferred if None.
        model_name: Name shown in the figure title; inferred if None.
    """
    if budget is None:
        budget = auction_history.agent_budgets[agent_idx]
    if cpa_target is None:
        cpa_target = auction_history.agent_cpas[agent_idx]
    if model_name is None:
        model_name = str(auction_history.agent_names[agent_idx])
    
    metrics = compute_campaign_metrics(auction_history, agent_idx)
    pacing_line = compute_budget_pacing_line(metrics["traffic"], budget, metrics["timesteps"])
    summary_table = make_campaign_summary_table(metrics, budget, cpa_target)
    
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Accumulated Budget Spent",
            "Per-step & Accumulated CPA",
            "Bid & Auction Price Trends",
            "Win Rate & Value Distribution",
            "Cumulative Conversions & Traffic Distribution",
            "Alpha (mean ±1std) · Bid Premium",
            "Summary Table",
        ),
        shared_xaxes=True,
        vertical_spacing=0.07,
        horizontal_spacing=0.12,
        specs=[
            [{}, {}],
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"type": "table", "colspan": 2}, None],
        ]
    )

    add_campaign_dashboard_traces(fig, metrics, budget, cpa_target, pacing_line)
    # Row 4: Summary table
    fig.add_trace(summary_table, row=4, col=1)

    # Update axes and layout
    fig.update_layout(
        height=1450,
        width=1100,
        title_text=f"Campaign Simulation Metrics ({model_name})",
        hovermode='x unified',
        barmode='overlay',
        margin=dict(l=50, r=30, t=90, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.01,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            tracegroupgap=8,
            itemwidth=50
        )
    )
    fig.update_xaxes(title_text="Timestep", row=3, col=1, title_standoff=6)
    fig.update_xaxes(title_text="Timestep", row=3, col=2, title_standoff=6)
    fig.update_yaxes(title_text="Spend", row=1, col=1)
    fig.update_yaxes(title_text="CPA", row=1, col=2)
    fig.update_yaxes(title_text="Average Bid", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Avg Auction Price", row=2, col=1, secondary_y=True)
    fig.update_yaxes(
        title_text="Win Rate", 
        row=2, col=2, 
        rangemode='tozero',
        range=[0, float(np.nanmax(metrics["win_rates"])) * 1.1 if np.isfinite(np.nanmax(metrics["win_rates"])) else 1],
        showgrid=True,
    )
    fig.update_yaxes(
        title_text="Value",
        row=2, col=2, secondary_y=True,
        rangemode='tozero',
        range=[0, float(np.nanmax(metrics["values"])) * 1.1 if np.isfinite(np.nanmax(metrics["values"])) else 1],
        showgrid=False,
        side='right',
        title_standoff=8
    )
    fig.update_yaxes(
        title_text="Cumulative Conversions",
        row=3, col=1,
        rangemode='tozero',
        range=[0, float(np.nanmax(metrics["cumulative_conversions"])) * 1.1 if np.isfinite(np.nanmax(metrics["cumulative_conversions"])) else 1],
        showgrid=True,
        title_standoff=8
    )
    fig.update_yaxes(
        title_text="Traffic Volume",
        row=3, col=1, secondary_y=True,
        rangemode='tozero',
        range=[0, float(np.nanmax(metrics["traffic"])) * 1.1 if np.isfinite(np.nanmax(metrics["traffic"])) else 1],
        showgrid=False,
        side='right',
        title_standoff=8
    )
    fig.update_yaxes(
        title_text="Alpha",
        row=3, col=2,
        rangemode='tozero',
        showgrid=True,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Bid Premium (%)",
        row=3, col=2, secondary_y=True,
        rangemode='tozero',
        showgrid=False,
        side='right',
        title_standoff=8
    )
    
    scatter_fig = plot_pairwise_scatterplots(metrics, ["avg_bids", "avg_auction_price", "win_rates", "costs"])
    return fig, scatter_fig


def plot_score_distribution_by_agent(
    df_results: pl.DataFrame,
    agent_col: str = "agent",
    score_col: str = "score",
    relative: bool = False,
    quantile: float = 0.25,
)-> go.Figure:
    """
    Plot score distribution by agent using violin plots.

    Args:
        df_results: Polars DataFrame with columns [agent_col, score_col].
        agent_col: Name of the column containing the agent names.
        score_col: Name of the column containing the score values.
        relative: If True, plot relative score distribution (%).
        quantile: Quantile to use as baseline for relative score (default 0.25).

    Returns:
        Plotly Figure.
    """

    if relative:
        # Use configurable quantile as baseline
        baseline = df_results.select(pl.col(score_col).quantile(quantile, "nearest")).item()
        max_score = df_results.select(pl.col(score_col).max()).item()

        # Create modified dataframe with relative scores
        df_plot = df_results.with_columns(
            rel_score_pct=((pl.col(score_col) - baseline) / (max_score - baseline) * 100)
        )

        y_col = "rel_score_pct"
        y_label = "Relative Score (%)"
        title = f"Relative Score Distribution by Agent (baseline: {int(quantile*100)}% quantile)"
    else:
        df_plot = df_results
        y_col = score_col
        y_label = "Score"
        title = "Score Distribution by Agent"

    fig = px.violin(
        df_plot,
        x=agent_col,
        y=y_col,
        box=True,
        points="all",
        color=agent_col,
        labels={agent_col: "Agent", y_col: y_label},
        title=title
    )
    fig.update_layout(
        xaxis_title="Agent",
        yaxis_title=y_label,
        font=dict(size=14)
    )
    return fig


def plot_score_by_agent(
    df_results: pl.DataFrame,
    agent_col: str = "agent",
    score_col: str = "score", 
    statistic: str = "mean",
    relative: bool = False,
    show_error: bool = True,
    quantile: float = 0.25,
)-> go.Figure:
    """
    Plot score statistics by agent using Plotly.

    Args:
        df_results: Polars DataFrame with columns [agent_col, score_col].
        agent_col: Name of the column containing the agent names.
        score_col: Name of the column containing the score values.
        statistic: Statistic to plot ("mean" or "median").
        relative: If True, plot relative score (%).
        show_error: If True, show error bars (std for mean, IQR for median).
        quantile: Quantile to use as baseline for relative score (default 0.25).
    
    Returns:
        Plotly Figure.
    """

    if statistic == "mean":
        stat_iqr = (
            df_results.group_by(agent_col, maintain_order=True)
            .agg(
                stat=pl.col(score_col).mean(),
                std=pl.col(score_col).std(),
            )
        )
        stat_label = "Mean"
        error_label = "std"
    elif statistic == "median":
        stat_iqr = (
            df_results.group_by(agent_col, maintain_order=True)
            .agg(
                stat=pl.col(score_col).median(),
                q25=pl.col(score_col).quantile(0.25),
                q75=pl.col(score_col).quantile(0.75),
            )
            .with_columns(
                iqr=pl.col("q75") - pl.col("q25")
            )
        )
        stat_label = "Median"
        error_label = "iqr"
    else:
        raise ValueError(f"Unsupported statistic: {statistic}. Use 'mean' or 'median'.")

    if relative:
        # Use configurable quantile as baseline
        baseline = df_results.select(pl.col(score_col).quantile(quantile, "nearest")).item()
        max_stat = stat_iqr["stat"].max()
        stat_iqr = (
            stat_iqr
            .with_columns(
                rel_stat_score_pct=((pl.col("stat") - baseline) / (max_stat - baseline) * 100)
            )
            .with_columns(
                text=(
                    pl.col("rel_stat_score_pct").round(1).cast(pl.Utf8) + "%<br>(" +
                    pl.col("stat").round(2).cast(pl.Utf8) + ")"  # TODO: Replace pl.Utf8 with pl.String?
                )
            )
        )
        y_col = "rel_stat_score_pct"
        y_label = f"Relative {stat_label} Score"
        title = f"Relative {stat_label} Score by Agent (baseline: {int(quantile*100)}% quantile)"
        text_col = "text"
    else:
        y_col = "stat"
        y_label = f"{stat_label} Score"
        title = f"{stat_label} Score by Agent"
        text_col = "stat"

    fig = px.bar(
        stat_iqr,
        x=agent_col,
        y=y_col,
        color=agent_col,
        error_y=error_label if show_error else None,
        labels={agent_col: "Agent", y_col: y_label},
        title=title,
        text=stat_iqr[text_col],
        category_orders={agent_col: stat_iqr[agent_col].to_list()},
    )
    fig.update_layout(
        xaxis_title="Agent",
        yaxis_title=y_label,
        font=dict(size=14),
        showlegend=False,
    )
    if show_error:
        fig.update_traces(
            error_y_thickness=1.2,
            error_y_width=4,
            error_y_color='rgba(0,0,0,0.4)'
        )
    return fig


def plot_interval_estimates(
    point_estimates: dict[str, list[float] | np.ndarray],
    interval_estimates: dict[str, np.ndarray],
    metric_names: list[str],
    algorithms: list[str] | None = None,
    colors: dict[str, str] | None = None,
    xlabel: str = 'Normalized Score',
    fig_width: int = 920,
    fig_height: int | None = None,
    bar_thickness: int = 10,
    row_spacing: float = 0.5,
    show_grid: bool = True,
    grid_opacity: float = 0.2,
    font_size: int = 12,
    title_font_size: int = 12,
    xlabel_font_size: int = 13,
    margin_config: dict[str, int] | None = None,
    subplot_spacing: float = 0.01,
    column_widths: list[float] | None = None,
    max_ticks: int = 5,
    **kwargs: Any
) -> go.Figure:
    """Plots various metrics with confidence intervals using horizontal bar format.

    Args:
        point_estimates: Dictionary mapping algorithm to a list or array of point
            estimates of the metrics to plot.
        interval_estimates: Dictionary mapping algorithms to interval estimates
            corresponding to the `point_estimates`. Typically, consists of stratified
            bootstrap CIs.
        metric_names: Names of the metrics corresponding to `point_estimates`.
        algorithms: List of methods used for plotting. If None, defaults to all the
            keys in `point_estimates`.
        colors: Maps each method to a color. If None, then this mapping is created
            automatically using plotly colors.
        xlabel: Label for the x-axis.
        fig_width: Width of the figure in pixels.
        fig_height: Height of the figure in pixels. If None, calculated automatically.
        bar_thickness: Thickness of the confidence interval bars.
        row_spacing: Vertical spacing between algorithm rows.
        show_grid: Whether to show grid lines on x-axis.
        grid_opacity: Opacity of grid lines (0-1).
        font_size: Base font size for tick labels.
        title_font_size: Font size for subplot titles.
        xlabel_font_size: Font size for main x-axis label.
        margin_config: Dictionary with margin configuration (l, r, t, b).
        subplot_spacing: Horizontal spacing between subplots.
        column_widths: List of relative column widths. If None, uses default.
        max_ticks: Maximum number of ticks on x-axis.
        **kwargs: Additional keyword arguments.

    Returns:
        fig: A plotly Figure.
    """
    
    # Default parameter handling
    if algorithms is None:
        algorithms = list(point_estimates.keys())
    
    if margin_config is None:
        margin_config = dict(l=3, r=3, t=20, b=40)
    
    if column_widths is None:
        column_widths = [0.24, 0.24, 0.26, 0.26]
    
    if fig_height is None:
        fig_height = max(200, len(algorithms) * 10 + 60)
    
    num_metrics = len(point_estimates[algorithms[0]])
    
    # Generate colors if not provided
    if colors is None:
        plotly_colors = plotly.colors.qualitative.Plotly
        if len(algorithms) > len(plotly_colors):
            # Cycle through colors if we need more than available
            plotly_colors = plotly_colors * ((len(algorithms) // len(plotly_colors)) + 1)
        colors = {alg: plotly_colors[i % len(plotly_colors)] for i, alg in enumerate(algorithms)}
    
    # Create subplot structure
    fig = make_subplots(
        rows=1, 
        cols=num_metrics,
        subplot_titles=metric_names,
        horizontal_spacing=subplot_spacing,
        shared_yaxes=False,
        column_widths=column_widths
    )
    
    # Add traces for each metric
    for metric_idx, metric_name in enumerate(metric_names):
        col_idx = metric_idx + 1
        
        for alg_idx, algorithm in enumerate(algorithms):
            y_pos = (len(algorithms) - 1 - alg_idx) * row_spacing
            
            # Get estimates
            lower, upper = interval_estimates[algorithm][:, metric_idx]
            point_est = point_estimates[algorithm][metric_idx]
            
            # Add confidence interval bar
            fig.add_trace(
                go.Scatter(
                    x=[lower, upper],
                    y=[y_pos, y_pos],
                    mode='lines',
                    line=dict(color=colors[algorithm], width=bar_thickness),
                    showlegend=False,
                    opacity=0.8,
                    hovertemplate=(
                        f'<b>{algorithm}</b><br>{metric_name}<br>'
                        f'Point: {point_est:.3f}<br>CI: [{lower:.3f}, {upper:.3f}]<extra></extra>'
                    ),
                    name=f'{algorithm}_{metric_name}'
                ),
                row=1, col=col_idx
            )
            
            # Add point estimate marker
            fig.add_trace(
                go.Scatter(
                    x=[point_est],
                    y=[y_pos],
                    mode='markers',
                    marker=dict(
                        symbol='line-ns',
                        size=8,
                        color='black',
                        line=dict(width=2)
                    ),
                    showlegend=False,
                    hoverinfo='skip',
                    name=f'{algorithm}_{metric_name}_point'
                ),
                row=1, col=col_idx
            )
        
        # Configure y-axis
        y_positions = [i * row_spacing for i in range(len(algorithms))]
        algorithm_display_names = algorithms[::-1] if metric_idx == 0 else [''] * len(algorithms)
        
        fig.update_yaxes(
            tickvals=y_positions,
            ticktext=algorithm_display_names,
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=font_size),
            showline=False,
            range=[-0.15, (len(algorithms) - 1) * row_spacing + 0.15],
            fixedrange=True,
            row=1, col=col_idx
        )
        
        # Configure x-axis
        all_data = []
        for algorithm in algorithms:
            lower, upper = interval_estimates[algorithm][:, metric_idx]
            all_data.extend([lower, upper])
        
        data_min, data_max = min(all_data), max(all_data)
        data_range = data_max - data_min
        x_min = data_min - 0.05 * data_range
        x_max = data_max + 0.1 * data_range
        
        fig.update_xaxes(
            showgrid=show_grid,
            gridcolor=f'rgba(128,128,128,{grid_opacity})',
            gridwidth=0.5,
            zeroline=False,
            tickfont=dict(size=font_size),
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True,
            nticks=max_ticks,
            range=[x_min, x_max],
            row=1, col=col_idx
        )
    
    # Apply final layout configuration
    fig.update_layout(
        height=fig_height,
        width=fig_width,
        margin=margin_config,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        font=dict(size=font_size),
    )
    
    # Update subplot titles
    for annotation in fig.layout.annotations:
        annotation.update(
            font=dict(size=title_font_size, color='black'),
            y=0.99
        )
    
    # Add main x-axis label
    fig.add_annotation(
        text=xlabel,
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        xanchor='center', yanchor='top',
        font=dict(size=xlabel_font_size),
        showarrow=False
    )
    
    return fig


def plot_budget_pacing_trajectories(budget: np.ndarray, budget_left_ratio: np.ndarray, agent_name: str) -> go.Figure:
    """
    Plot budget pacing trajectories as a line plot.

    Args:
        budget: Array of budget values for each trajectory
        budget_left_ratio: 2D array of budget left ratios (trajectories x timesteps)
        agent_name: Name of the agent to include in the title
    
    Returns:
        Plotly Figure with the trajectories.
    """
    num_trajectories, num_timesteps = budget_left_ratio.shape
    timesteps = np.arange(num_timesteps)

    # Map budgets to colors on a continuous colorscale
    budgets = budget.astype(float)
    bmin, bmax = float(np.min(budgets)), float(np.max(budgets))
    denom = (bmax - bmin) if bmax > bmin else 1.0
    norm = (budgets - bmin) / denom
    colors = plotly.colors.sample_colorscale('RdYlBu_r', norm)

    fig = go.Figure()

    # Add one semi-transparent line per trajectory
    for i in range(num_trajectories):
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=budget_left_ratio[i],
            mode='lines',
            line=dict(color=colors[i], width=1.5),
            opacity=0.35,  # good opacity for overlapping lines
            showlegend=False,
            customdata=np.full((num_timesteps, 1), budgets[i]),
            hovertemplate='<b>Time Step:</b> %{x}<br>' +
                          '<b>Budget Left Ratio:</b> %{y:.3f}<br>' +
                          '<b>Budget:</b> %{customdata[0]:.0f}<br>' +
                          '<extra></extra>',
        ))

    # Add an invisible marker trace to display a single shared colorbar
    fig.add_trace(go.Scatter(
        x=budgets,
        y=np.zeros_like(budgets),
        mode='markers',
        marker=dict(
            color=budgets,
            colorscale='RdYlBu_r',
            showscale=True,
            opacity=0,
            size=0,
            colorbar=dict(title="Budget", thickness=15, len=0.8),
            cmin=bmin, cmax=bmax
        ),
        hoverinfo='skip',
        showlegend=False
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text=f"Budget Pacing Trajectories - {agent_name}",
            font=dict(size=20, family="Arial Black"),
            x=0.5
        ),
        xaxis=dict(
            title="Time Step",
            title_font=dict(size=14, family="Arial"),
            tickfont=dict(size=12),
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            range=[0, num_timesteps - 1],
        ),
        yaxis=dict(
            title="Budget Left Ratio",
            title_font=dict(size=14, family="Arial"),
            tickfont=dict(size=12),
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=500,
        margin=dict(l=60, r=60, t=80, b=60)
    )

    return fig


def plot_auction_market_overview(
    history: StepwiseAuctionHistory,
    *,
    bid_threshold: float = 0.01,
    gap_threshold: float = 0.0,
    win_threshold: float = 0.0,
    pct: float = 0.05,
    vol_window: int = 4,
    use_log: bool = True,
    smooth_window: int | None = 3,
    template: str = "plotly",
    title: str | None = None,
)-> go.Figure:
    """
    Overview of market-level metrics over time.

    Args:
        history: Stepwise auction history to summarize.
        bid_threshold: Minimum bid to consider a bidder active.
        gap_threshold: Minimum bid threshold when computing top-2 gap.
        win_threshold: Threshold for considering wins when computing Jaccard similarity.
        pct: Proximity band around clearing price for top-of-book pressure.
        vol_window: Rolling window for realized volatility.
        use_log: Whether to use log returns for volatility.
        smooth_window: Window for NaN-aware smoothing of some series (visual only).
        template: Plotly template name.
        title: Optional figure title.

    Returns:
        Plotly Figure.
    """
    T = history.avg_bids.shape[0]
    N = history.avg_bids.shape[1]
    t = np.arange(T)

    # --- metrics ---
    part = participation_rate_per_tick(history, bid_threshold=bid_threshold)
    gap = bid_gap_ratio_per_tick(history, bid_threshold=gap_threshold)
    hhi, enc = hhi_cumulative_by_wins(history, return_enc=True)
    gini = gini_cumulative_by_wins(history)
    pressure = top_of_book_pressure_per_tick(history, pct=pct, bid_threshold=bid_threshold)
    vol = realized_volatility(history, window=vol_window, use_log=use_log)
    aggr = bid_aggressiveness_mean_per_tick(history, bid_threshold=bid_threshold)
    jacc = winner_set_jaccard_per_tick(history, win_threshold=win_threshold)

    # optional smoothing (visual only)
    part_s = _nan_rolling_mean(part, smooth_window)
    gap_s = _nan_rolling_mean(gap, smooth_window)
    gini_s = _nan_rolling_mean(gini, smooth_window)
    press_s = _nan_rolling_mean(pressure.astype(float), max(3, (smooth_window or 1)))  # bars + smooth line
    vol_s = vol  # already rolling
    aggr_s = _nan_rolling_mean(aggr, smooth_window)
    jacc_s = _nan_rolling_mean(jacc, smooth_window)

    # --- layout ---
    fig = make_subplots(
        rows=4, cols=2,
        specs=[
            [{}, {}],
            [{"secondary_y": True}, {}],
            [{}, {}],
            [{}, {}],
        ],
        subplot_titles=(
            "Participation Rate",
            "Top‑2 Bid Gap Ratio",
            "HHI (primary) · ENC (secondary)",
            "Gini (Cumulative Wins)",
            f"Top‑of‑Book Pressure (±{int(pct*100)}% of clearing price)",
            "Realized Volatility (Clearing Price)",
            "Bid Aggressiveness (mean)",
            "Winner‑Set Jaccard (t vs t‑1)",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.10,
    )

    # Row 1
    fig.add_trace(go.Scatter(
        x=t, y=part_s, mode="lines",
        name="Participation",
        hovertemplate="t=%{x}<br>rate=%{y:.3f}<extra></extra>",
        line=dict(color="#2CA02C", width=2.2),
    ), row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=gap_s, mode="lines",
        name="(top1/top2) - 1",
        hovertemplate="t=%{x}<br>gap=%{y:.3f}<extra></extra>",
        line=dict(color="#D62728", width=2.2),
    ), row=1, col=2)
    fig.add_hline(y=0.0, line_dash="dot", line_color="gray", row=1, col=2)

    # Row 2 (HHI + ENC)
    fig.add_trace(go.Scatter(
        x=t, y=hhi, mode="lines",
        name="HHI",
        hovertemplate="t=%{x}<br>HHI=%{y:.3f}<extra></extra>",
        line=dict(color="#9467BD", width=2.2),
    ), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=t, y=enc, mode="lines",
        name="ENC",
        hovertemplate="t=%{x}<br>ENC=%{y:.2f}<extra></extra>",
        line=dict(color="#1F77B4", width=2.0, dash="dash"),
    ), row=2, col=1, secondary_y=True)
    # HHI guideline lines
    for thr, c in [(0.15, "rgba(0,0,0,0.25)"), (0.25, "rgba(0,0,0,0.25)")]:
        fig.add_hline(y=thr, line_dash="dot", line_color=c, row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=gini_s, mode="lines",
        name="Gini",
        hovertemplate="t=%{x}<br>Gini=%{y:.3f}<extra></extra>",
        line=dict(color="#FF7F0E", width=2.2),
    ), row=2, col=2)

    # Row 3
    fig.add_trace(go.Bar(
        x=t, y=pressure, name="# bids in band",
        marker_color="#8C6BB1", opacity=0.45,
        hovertemplate="t=%{x}<br>bids=%{y}<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=press_s, mode="lines",
        name="pressure (smooth)",
        line=dict(color="#8C6BB1", width=2.0),
        hoverinfo="skip",
        showlegend=True,
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=vol_s, mode="lines",
        name="Realized Vol",
        hovertemplate="t=%{x}<br>vol=%{y:.3f}<extra></extra>",
        line=dict(color="#17BECF", width=2.2),
    ), row=3, col=2)

    # Row 4
    fig.add_trace(go.Scatter(
        x=t, y=aggr_s, mode="lines",
        name="Aggressiveness",
        hovertemplate="t=%{x}<br>(v-b)/v=%{y:.3f}<extra></extra>",
        line=dict(color="#1F77B4", width=2.2),
    ), row=4, col=1)
    fig.add_hline(y=0.0, line_dash="dot", line_color="gray", row=4, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=jacc_s, mode="lines",
        name="Jaccard",
        hovertemplate="t=%{x}<br>Jaccard=%{y:.3f}<extra></extra>",
        line=dict(color="#7F7F7F", width=2.2),
    ), row=4, col=2)

    # Axes and layout
    fig.update_layout(
        template=template,
        title=title or f"Auction Market Metrics • T={T}, N={N}",
        height=1400,
        width=1180,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.02, xanchor="center", x=0.5, font=dict(size=12)),
        margin=dict(l=50, r=30, t=80, b=60),
    )
    # Ranges, labels, grids
    fig.update_yaxes(title_text="Rate", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Gap", rangemode="tozero", row=1, col=2)

    hhi_max = float(np.nanmax(hhi)) if np.isfinite(np.nanmax(hhi)) else 0.0
    hhi_ymax = max(1e-3, min(1.0, hhi_max * 1.10))  # 110% of max, capped at 1, avoid zero-height
    fig.update_yaxes(title_text="HHI", range=[0, hhi_ymax], row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="ENC", range=[1, max(1, float(np.nanmax(enc)) * 1.1)], row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Gini", range=[0, 1], row=2, col=2)

    fig.update_yaxes(title_text="# bids", rangemode="tozero", row=3, col=1)
    fig.update_yaxes(title_text="Vol", rangemode="tozero", row=3, col=2)

    fig.update_yaxes(title_text="(v-b)/v", row=4, col=1)
    fig.update_yaxes(title_text="Similarity", range=[0, 1], row=4, col=2)

    fig.update_xaxes(title_text="Timestep", row=4, col=1)
    fig.update_xaxes(title_text="Timestep", row=4, col=2)

    # Global rangeslider on bottom row; link others
    for (r, c) in [(4, 1), (4, 2)]:
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06), row=r, col=c)
    for (r, c) in [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2)]:
        fig.update_xaxes(matches="x", row=r, col=c)

    return fig
