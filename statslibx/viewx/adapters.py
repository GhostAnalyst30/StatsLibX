"""
ViewX integration layer — serializers and optional re-exports for statslibx results.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from ..computational import (
    BootstrappingResult,
    JackknifeResult,
    MonteCarloResult,
    RegressionResult,
)
from ..descriptive import DescriptiveSummary, LinearRegressionResult
from ..inferential import PowerResult, TestResult

# ViewX classes are optional; only imported when available.
try:
    from viewx import HTML, Report, DataMatrix, Presentation, Slide

    _VIEWX_CLASSES = True
except ImportError:
    HTML = Report = DataMatrix = Presentation = Slide = None  # type: ignore[misc, assignment]
    _VIEWX_CLASSES = False

__all__ = [
    "HTML",
    "Report",
    "DataMatrix",
    "Presentation",
    "Slide",
    "to_report_data",
    "VIEWX_CLASSES_AVAILABLE",
]

VIEWX_CLASSES_AVAILABLE: bool = _VIEWX_CLASSES


def _regression_payload(result: Union[LinearRegressionResult, RegressionResult]) -> Dict[str, Any]:
    """Build a normalized payload for regression result types."""
    if isinstance(result, LinearRegressionResult):
        formula = f"{result.y_name} ~ " + " + ".join(result.X_names)
        metrics = {
            "R2": result.r_squared,
            "Adjusted R2": result.adj_r_squared,
        }
        if result.aic is not None:
            metrics["AIC"] = result.aic
            metrics["BIC"] = result.bic
        coef_rows = []
        coef_rows.append({"term": "intercept", "coefficient": result.intercept_})
        for i, name in enumerate(result.X_names):
            row = {"term": name, "coefficient": float(result.coef_[i])}
            if result.p_values is not None:
                row["p_value"] = float(result.p_values[i])
            coef_rows.append(row)
        valueboxes = [
            {"label": "R²", "value": f"{result.r_squared:.4f}"},
            {"label": "Adj. R²", "value": f"{result.adj_r_squared:.4f}"},
        ]
        figures: List[Dict[str, Any]] = []
        return {
            "title": "Regression Analysis",
            "sections": [{"heading": "Model", "body": formula}],
            "tables": [
                {"name": "metrics", "data": [metrics]},
                {"name": "coefficients", "data": coef_rows},
            ],
            "valueboxes": valueboxes,
            "figures": figures,
            "metadata": {"result_type": "LinearRegressionResult", "engine": result.engine},
        }

    summary = result.summary()
    metrics = summary.get("metrics", {})
    coef_df = summary.get("coefficients")
    coef_rows = []
    if coef_df is not None:
        tmp = coef_df.reset_index().rename(columns={"index": "term"})
        coef_rows = tmp.to_dict(orient="records")
    valueboxes = [
        {"label": "R²", "value": f"{metrics.get('R2', 0):.4f}"},
        {"label": "RMSE", "value": f"{metrics.get('RMSE', 0):.4f}"},
    ]
    tables = [{"name": "metrics", "data": [metrics]}]
    if coef_rows:
        tables.append({"name": "coefficients", "data": coef_rows})
    formula_body = summary.get("model_info", {}).get("formula") or result.get_formula()
    return {
        "title": "Regression Analysis",
        "sections": [{"heading": "Model", "body": formula_body}],
        "tables": tables,
        "valueboxes": valueboxes,
        "figures": [],
        "metadata": {"result_type": "RegressionResult"},
    }


def _attach_figures(
    payload: Dict[str, Any],
    result: Any,
    include_figures: bool,
    data: Optional[Any],
) -> Dict[str, Any]:
    """Attach Plotly figures to payload when requested."""
    if not include_figures:
        return payload

    from ..viz.figures import (
        bootstrap_distribution_figure,
        correlation_heatmap_figure,
        distribution_figure,
        linear_regression_diagnostics_figure,
        monte_carlo_figure,
        regression_diagnostics_figure,
    )

    figures: List[Dict[str, Any]] = list(payload.get("figures", []))

    if isinstance(result, LinearRegressionResult):
        figures.append(
            {
                "title": "Regression diagnostics",
                "kind": "regression",
                "fig": linear_regression_diagnostics_figure(result),
            }
        )
    elif isinstance(result, RegressionResult):
        figures.append(
            {
                "title": "Regression diagnostics",
                "kind": "regression",
                "fig": regression_diagnostics_figure(result),
            }
        )
    elif isinstance(result, BootstrappingResult):
        figures.append(
            {
                "title": "Bootstrap distribution",
                "kind": "bootstrap",
                "fig": bootstrap_distribution_figure(result),
            }
        )
    elif isinstance(result, MonteCarloResult):
        figures.append(
            {
                "title": "Monte Carlo distribution",
                "kind": "montecarlo",
                "fig": monte_carlo_figure(result),
            }
        )
    elif isinstance(result, DescriptiveSummary) and data is not None:
        import pandas as pd

        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        max_figs = int(payload.get("metadata", {}).get("max_distribution_figures", 3))
        for col in numeric_cols[:max_figs]:
            figures.append(
                {
                    "title": f"Distribution — {col}",
                    "kind": "distribution",
                    "fig": distribution_figure(df[col], title=f"Distribution — {col}"),
                }
            )
        if len(numeric_cols) >= 2:
            figures.append(
                {
                    "title": "Correlation heatmap",
                    "kind": "correlation",
                    "fig": correlation_heatmap_figure(df[numeric_cols]),
                }
            )

    payload["figures"] = figures
    return payload


def to_report_data(
    result: Any,
    include_figures: bool = False,
    data: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Serialize a statslibx result into a ViewX-friendly report payload.

    This function does **not** require ViewX to be installed.
    """
    if isinstance(result, dict):
        return result

    if isinstance(result, DescriptiveSummary):
        df = result.to_dataframe(format="compact")
        payload: Dict[str, Any] = {
            "title": "Descriptive Statistics Summary",
            "sections": [{"heading": "Overview", "body": "Descriptive statistics by variable."}],
            "tables": [{"name": "descriptive_stats", "data": df.reset_index().to_dict(orient="records")}],
            "valueboxes": [
                {"label": "Variables", "value": str(len(result.results))},
            ],
            "figures": [],
            "metadata": {
                "result_type": "DescriptiveSummary",
                "max_distribution_figures": kwargs.get("max_distribution_figures", 3),
            },
        }
        return _attach_figures(payload, result, include_figures, data)

    if isinstance(result, TestResult):
        valueboxes = [
            {"label": "Statistic", "value": f"{result.statistic:.6f}"},
        ]
        if result.pvalue is not None:
            valueboxes.append({"label": "p-value", "value": f"{result.pvalue:.6e}"})
        tables: List[Dict[str, Any]] = []
        if isinstance(result.params, dict) and result.params:
            tables.append({"name": "params", "data": [result.params]})
        if result.critical_values is not None and result.significance_levels is not None:
            tables.append({
                "name": "critical_values",
                "data": [
                    {"alpha": a, "critical": c}
                    for a, c in zip(result.significance_levels, result.critical_values)
                ],
            })
        if isinstance(result.homo_result, dict):
            tables.append({"name": "homoscedasticity", "data": [result.homo_result]})
        payload = {
            "title": result.test_name,
            "sections": [
                {
                    "heading": "Results",
                    "body": result.interpretation,
                    "statistic": result.statistic,
                    "pvalue": result.pvalue,
                    "alpha": result.alpha,
                }
            ],
            "tables": tables,
            "valueboxes": valueboxes,
            "figures": [],
            "metadata": {"result_type": "TestResult", "alternative": result.alternative},
        }
        return _attach_figures(payload, result, include_figures, data)

    if isinstance(result, PowerResult):
        payload = {
            "title": result.test_name,
            "sections": [{"heading": "Power analysis", "body": f"Power={result.power:.4f}"}],
            "tables": [{"name": "power", "data": [result.to_dict()]}],
            "valueboxes": [
                {"label": "Power", "value": f"{result.power:.4f}"},
                {"label": "Effect size", "value": f"{result.effect_size:.4f}"},
                {"label": "Sample size", "value": str(result.sample_size)},
            ],
            "figures": [],
            "metadata": {"result_type": "PowerResult"},
        }
        return payload

    if isinstance(result, BootstrappingResult):
        payload = {
            "title": f"Bootstrap ({result.statistic})",
            "sections": [{"heading": "Summary", "body": f"n_samples={result.n_samples}"}],
            "tables": [{"name": "bootstrap", "data": [{
                "original": result.original_stat,
                "bias": result.bias,
                "std_error": result.std_error,
                "ci_low": result.percentile_ci[0],
                "ci_high": result.percentile_ci[1],
            }]}],
            "valueboxes": [
                {"label": "Original", "value": f"{result.original_stat:.4f}"},
                {"label": "SE", "value": f"{result.std_error:.4f}"},
            ],
            "figures": [],
            "metadata": {"result_type": "BootstrappingResult"},
        }
        return _attach_figures(payload, result, include_figures, data)

    if isinstance(result, MonteCarloResult):
        payload = {
            "title": result.name,
            "sections": [{"heading": "Monte Carlo", "body": f"n={len(result.simulations)}"}],
            "tables": [{"name": "montecarlo", "data": [{
                "point_estimate": result.point_estimate,
                "mean": result.mean,
                "std": result.std,
                "ci_low": result.ci[0],
                "ci_high": result.ci[1],
            }]}],
            "valueboxes": [
                {"label": "Mean", "value": f"{result.mean:.4f}"},
                {"label": "Std", "value": f"{result.std:.4f}"},
            ],
            "figures": [],
            "metadata": {"result_type": "MonteCarloResult"},
        }
        return _attach_figures(payload, result, include_figures, data)

    if isinstance(result, JackknifeResult):
        payload = {
            "title": f"Jackknife ({result.statistic})",
            "sections": [{"heading": "Jackknife", "body": "Bias and SE via leave-one-out"}],
            "tables": [{"name": "jackknife", "data": [result.summary()]}],
            "valueboxes": [
                {"label": "Estimate", "value": f"{result.point_estimate:.4f}"},
                {"label": "Bias", "value": f"{result.bias:.4f}"},
                {"label": "SE", "value": f"{result.std_error:.4f}"},
            ],
            "figures": [],
            "metadata": {"result_type": "JackknifeResult"},
        }
        return payload

    if isinstance(result, (LinearRegressionResult, RegressionResult)):
        payload = _regression_payload(result)
        return _attach_figures(payload, result, include_figures, data)

    raise TypeError(
        f"Unsupported result type: {type(result).__name__}. "
        "Expected DescriptiveSummary, TestResult, PowerResult, LinearRegressionResult, "
        "RegressionResult, BootstrappingResult, MonteCarloResult, JackknifeResult, or dict."
    )
