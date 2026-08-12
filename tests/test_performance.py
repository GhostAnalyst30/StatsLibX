"""Lightweight performance regression checks for v0.4 hot paths."""

import time

import numpy as np
import pandas as pd
import pytest

from statslibx import ComputationalStats, DescriptiveStats
from statslibx._stats_utils import bootstrap_ci, vectorized_bootstrap


def test_vectorized_bootstrap_faster_than_loop():
    rng = np.random.default_rng(0)
    vals = rng.normal(size=400)

    t0 = time.perf_counter()
    vectorized_bootstrap(vals, np.mean, n_resamples=2000, random_state=0)
    vec_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(2000):
        np.mean(rng.choice(vals, size=len(vals), replace=True))
    loop_ms = (time.perf_counter() - t0) * 1000

    # Vectorized path should be meaningfully faster on typical hardware.
    assert vec_ms < loop_ms


def test_summary_column_cache_completes_quickly():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({f"c{i}": rng.normal(size=5000) for i in range(8)})
    t0 = time.perf_counter()
    summary = DescriptiveStats(df).summary()
    elapsed = time.perf_counter() - t0
    assert len(summary.results) == 8
    # Soft budget — CI machines vary; keep generous.
    assert elapsed < 5.0


def test_bootstrap_ci_returns_finite():
    vals = np.arange(50, dtype=float)
    result = bootstrap_ci(vals, n_resamples=1000, random_state=3)
    assert np.isfinite(result["lower"])
    assert result["lower"] < result["upper"]


def test_bootstrap_result_vectorized():
    df = pd.DataFrame({"x": np.random.default_rng(2).normal(size=300)})
    t0 = time.perf_counter()
    result = ComputationalStats(df).bootstrap("x", n_samples=3000)
    elapsed = time.perf_counter() - t0
    assert len(result.bootstrap_stats) == 3000
    assert elapsed < 5.0
