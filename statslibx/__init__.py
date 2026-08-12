"""
StatsLibx — Statistical library for Python.

Provides descriptive, inferential, and computational statistics,
data preprocessing, dataset loading, and optional ViewX visualization.
"""

from __future__ import annotations

import logging
import sys

__version__ = "0.3.1"
__author__ = "Emmanuel Ascendra"

# ── Logging configuration ─────────────────────────────────────────────

_log_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(_log_fmt, datefmt="%H:%M:%S"))

logger = logging.getLogger("statslibx")
logger.addHandler(_handler)
logger.setLevel(logging.WARNING)

# ── Public API exports ────────────────────────────────────────────────

from .backend import Backend, BackendType, resolve_backend
from .descriptive import DescriptiveStats, DescriptiveSummary, LinearRegressionResult
from .inferential import (
    InferentialStats,
    TestResult,
    PowerResult,
    ConfidenceIntervalResult,
    PairwiseResult,
    adjust_pvalues,
)
from .computational import (
    ComputationalStats,
    RegressionResult,
    BootstrappingResult,
    MonteCarloResult,
    JackknifeResult,
    InterpolationResult,
)
from .utils import UtilsStats
from .preprocessing import Preprocessing
from .datasets import load_dataset, generate_dataset, load_iris, load_penguins

# Core serializer — always available (does not require viewx installed).
from .viewx.adapters import to_report_data
from .viewx._availability import VIEWX_AVAILABLE

# Optional ViewX bridge (None when viewx is not installed).
from .viewx.adapters import (
    HTML,
    Report,
    DataMatrix,
    Presentation,
    Slide,
    VIEWX_CLASSES_AVAILABLE,
)

from .viewx.renderers import render_html, render_report, render_presentation, render

__all__ = [
    "DescriptiveStats",
    "InferentialStats",
    "ComputationalStats",
    "UtilsStats",
    "Preprocessing",
    "DescriptiveSummary",
    "LinearRegressionResult",
    "TestResult",
    "PowerResult",
    "ConfidenceIntervalResult",
    "PairwiseResult",
    "adjust_pvalues",
    "RegressionResult",
    "BootstrappingResult",
    "MonteCarloResult",
    "JackknifeResult",
    "InterpolationResult",
    "Backend",
    "BackendType",
    "resolve_backend",
    "load_dataset",
    "generate_dataset",
    "load_iris",
    "load_penguins",
    "to_report_data",
    "VIEWX_AVAILABLE",
    "VIEWX_CLASSES_AVAILABLE",
    "HTML",
    "Report",
    "DataMatrix",
    "Presentation",
    "Slide",
    "render_html",
    "render_report",
    "render_presentation",
    "render",
    "logger",
]

# ── Welcome message ───────────────────────────────────────────────────

def welcome() -> None:
    """Display library information."""
    print(f"StatsLibx v{__version__}")
    print("Descriptive and inferential statistics library")
    print(f"Author: {__author__}")
    print()
    print("Classes:")
    print("  - DescriptiveStats   Descriptive statistics")
    print("  - InferentialStats   Inferential statistics")
    print("  - ComputationalStats Computational statistics")
    print("  - UtilsStats         Utilities and visualization")
    print("  - Preprocessing      Data preparation")
    print()
    print("Modules:")
    print("  - datasets           Built-in datasets and generators")
    print("  - viewx              Optional ViewX export (pip install statslibx[viewx])")
    print()
    if VIEWX_AVAILABLE:
        print("ViewX: installed — use .to_html() / .to_presentation() on results")
    else:
        print("ViewX: not installed — core analysis works; pip install statslibx[viewx] to export")
    print()
    print(f"Docs: https://github.com/GhostAnalyst30/StatsLibX")
    print(f"Web:  https://statslibx.vercel.app/")
