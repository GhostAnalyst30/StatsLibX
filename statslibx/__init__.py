"""
StatsLibx — Statistical library for Python.

Provides descriptive, inferential, and computational statistics,
data preprocessing, dataset loading, and visualization utilities.
"""

from __future__ import annotations

import logging
import sys

__version__ = "0.2.9"
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
from .descriptive import DescriptiveStats, DescriptiveSummary
from .inferential import InferentialStats, TestResult
from .computational import ComputationalStats
from .utils import UtilsStats
from .preprocessing import Preprocessing
from .datasets import load_dataset, generate_dataset

try:
    from .viewx import HTML, Slides, Report, DataMatrix, to_report_data
except ImportError:
    HTML = Slides = Report = DataMatrix = to_report_data = None  # type: ignore[misc, assignment]

__all__ = [
    "DescriptiveStats",
    "InferentialStats",
    "ComputationalStats",
    "UtilsStats",
    "Preprocessing",
    "DescriptiveSummary",
    "TestResult",
    "Backend",
    "BackendType",
    "resolve_backend",
    "load_dataset",
    "generate_dataset",
    "HTML",
    "Slides",
    "Report",
    "DataMatrix",
    "to_report_data",
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
    print()
    print(f"Docs: https://github.com/GhostAnalyst30/StatsLibX")
    print(f"Web:  https://statslibx.vercel.app/")
