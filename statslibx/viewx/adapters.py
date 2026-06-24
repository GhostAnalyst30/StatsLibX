"""
ViewX integration layer — re-exports and adapters for statslibx results.
"""

from __future__ import annotations

from typing import Any, Dict, Union

try:
    from viewx import HTML, Slides, Report, DataMatrix
except ImportError as exc:
    raise ImportError(
        "ViewX is not installed. Install with: pip install viewx"
    ) from exc

from ..computational import RegressionResult
from ..descriptive import DescriptiveSummary
from ..inferential import TestResult

__all__ = [
    "HTML",
    "Slides",
    "Report",
    "DataMatrix",
    "to_report_data",
]


def to_report_data(result: Union[DescriptiveSummary, TestResult, RegressionResult, dict]) -> Dict[str, Any]:
    """
    Serialize a statslibx result into a ViewX-friendly report payload.

    Returns a dict with keys: title, sections, tables.
    """
    if isinstance(result, dict):
        return result

    if isinstance(result, DescriptiveSummary):
        df = result.to_dataframe(format="compact")
        return {
            "title": "Descriptive Statistics Summary",
            "sections": [{"heading": "Overview", "body": str(result)}],
            "tables": [{"name": "descriptive_stats", "data": df.to_dict(orient="records")}],
        }

    if isinstance(result, TestResult):
        return {
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
            "tables": [],
        }

    if isinstance(result, RegressionResult):
        summary = result.summary()
        return {
            "title": "Regression Analysis",
            "sections": [{"heading": "Model", "body": result.get_formula()}],
            "tables": [{"name": "metrics", "data": [summary.get("metrics", {})]}],
        }

    raise TypeError(
        f"Unsupported result type: {type(result).__name__}. "
        "Expected DescriptiveSummary, TestResult, RegressionResult, or dict."
    )
