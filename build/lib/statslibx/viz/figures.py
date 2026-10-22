"""
Plotly figure builders used by statslibx ViewX export payloads.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DistributionKind = Literal["histogram", "box", "violin"]


def distribution_figure(
    series: Union[pd.Series, np.ndarray],
    kind: DistributionKind = "histogram",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Build a Plotly distribution figure for a numeric series.

    Parameters
    ----------
    series : Series or ndarray
        Numeric values to visualize.
    kind : {'histogram', 'box', 'violin'}
        Chart style.
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure ready for ViewX ``add_chart(fig=...)``.
    """
    values = pd.Series(series).dropna()
    name = getattr(series, "name", None) or "value"
    fig = go.Figure()

    if kind == "histogram":
        fig.add_trace(go.Histogram(x=values, name=str(name)))
    elif kind == "box":
        fig.add_trace(go.Box(y=values, name=str(name)))
    else:
        fig.add_trace(go.Violin(y=values, name=str(name), box_visible=True, meanline_visible=True))

    fig.update_layout(
        title=title or f"Distribution — {name}",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def correlation_heatmap_figure(
    df: pd.DataFrame,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Build a correlation heatmap for numeric columns.

    Parameters
    ----------
    df : DataFrame
        Input data with numeric columns.
    method : {'pearson', 'spearman', 'kendall'}
        Correlation method.
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Heatmap figure.
    """
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr(method=method)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
        )
    )
    fig.update_layout(
        title=title or f"Correlation ({method})",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def linear_regression_diagnostics_figure(result: Any) -> go.Figure:
    """
    Build regression diagnostic subplots for ``LinearRegressionResult``.

    Parameters
    ----------
    result : LinearRegressionResult
        Fitted descriptive regression result.

    Returns
    -------
    plotly.graph_objects.Figure
        Multi-panel diagnostic figure.
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Fit", "Residuals"))

    if len(result.X_names) == 1:
        x_vals = np.asarray(result.X).flatten()
        fig.add_trace(
            go.Scatter(x=x_vals, y=result.y, mode="markers", name="Actual"),
            row=1,
            col=1,
        )
        order = np.argsort(x_vals)
        fig.add_trace(
            go.Scatter(
                x=x_vals[order],
                y=result.predictions[order],
                mode="lines",
                name="Predicted",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(y=result.y, mode="markers", name="Actual"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(y=result.predictions, mode="lines", name="Predicted"),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(x=result.predictions, y=result.residuals, mode="markers", name="Residuals"),
        row=1,
        col=2,
    )
    fig.update_layout(
        title=f"Regression: {result.y_name}",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def regression_diagnostics_figure(result: Any) -> go.Figure:
    """
    Build regression diagnostic subplots for ``RegressionResult``.

    Parameters
    ----------
    result : RegressionResult
        Fitted computational regression result.

    Returns
    -------
    plotly.graph_objects.Figure
        Multi-panel diagnostic figure.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Fit", "Residuals", "Q-Q", "Residual histogram"),
    )

    if result.n_features == 1:
        x_vals = result.X_values.flatten()
        fig.add_trace(
            go.Scatter(x=x_vals, y=result.y_values, mode="markers", name="Actual"),
            row=1,
            col=1,
        )
        x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_range = result.predict(x_range)
        fig.add_trace(
            go.Scatter(x=x_range, y=y_range, mode="lines", name="Predicted"),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(y=result.y_values, mode="markers", name="Actual"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(y=result.y_pred, mode="lines", name="Predicted"),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(x=result.y_pred, y=result.residuals, mode="markers", name="Residuals"),
        row=1,
        col=2,
    )

    from scipy import stats as scipy_stats

    osm, osr = scipy_stats.probplot(result.residuals, dist="norm", fit=False)
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Q-Q"), row=2, col=1)
    fig.add_trace(go.Histogram(x=result.residuals, name="Residuals"), row=2, col=2)

    fig.update_layout(
        title="Regression diagnostics",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def bootstrap_distribution_figure(result: Any, title: Optional[str] = None) -> go.Figure:
    """Histogram of bootstrap statistics with CI markers."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=result.bootstrap_stats, nbinsx=40, name="bootstrap"))
    fig.add_vline(x=result.original_stat, line_dash="dash", line_color="red")
    if hasattr(result, "percentile_ci"):
        fig.add_vline(x=result.percentile_ci[0], line_dash="dot", line_color="green")
        fig.add_vline(x=result.percentile_ci[1], line_dash="dot", line_color="green")
    fig.update_layout(
        title=title or f"Bootstrap — {getattr(result, 'statistic', 'stat')}",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def monte_carlo_figure(result: Any, title: Optional[str] = None) -> go.Figure:
    """Histogram of Monte Carlo simulations."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=result.simulations, nbinsx=40, name="simulations"))
    fig.add_vline(x=result.point_estimate, line_dash="dash", line_color="red")
    if hasattr(result, "ci"):
        fig.add_vline(x=result.ci[0], line_dash="dot", line_color="green")
        fig.add_vline(x=result.ci[1], line_dash="dot", line_color="green")
    fig.update_layout(
        title=title or getattr(result, "name", "Monte Carlo"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def power_curve_figure(
    effect_sizes: Union[list, np.ndarray],
    powers: Union[list, np.ndarray],
    title: Optional[str] = None,
) -> go.Figure:
    """Simple power curve (effect size vs power)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(effect_sizes), y=list(powers), mode="lines+markers", name="power"))
    fig.update_layout(
        title=title or "Power curve",
        xaxis_title="Effect size",
        yaxis_title="Power",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def permutation_test_figure(
    null_dist: Union[list, np.ndarray],
    observed: float,
    title: Optional[str] = None,
) -> go.Figure:
    """Null distribution from a permutation test with observed statistic."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=list(null_dist), nbinsx=40, name="null"))
    fig.add_vline(x=observed, line_dash="dash", line_color="red")
    fig.update_layout(
        title=title or "Permutation null distribution",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
