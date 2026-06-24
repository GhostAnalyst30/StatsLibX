"""
Command-line interface for statslibx.

Usage:
    statslibx describe <dataset> [--numeric] [--categorical]
    statslibx data <dataset> [--summary] [--types] [--missing]
    statslibx preview <dataset> [--rows N] [--sample]
    statslibx quality <dataset> [--verbose]
    statslibx info <dataset> [--detailed]
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from statslibx.datasets import load_dataset
from statslibx.preprocessing import Preprocessing

logger = logging.getLogger(__name__)


# ── Dataclass-based argument model ─────────────────────────────────────

@dataclass
class CliArgs:
    command: str
    file: str
    numeric: bool = False
    categorical: bool = False
    verbose: bool = False
    rows: int = 5
    sample: bool = False
    detailed: bool = False
    summary: bool = False
    types: bool = False
    missing: bool = False


# ── Pretty-print helpers ──────────────────────────────────────────────

def _header(title: str, char: str = "=") -> None:
    print(f"\n{title}")
    print(char * min(len(title), 80))


def _fmt_col(col: str, dtype: str, nulls: int, unique: int, width: int = 22) -> str:
    return (
        f"  {col:<{width}} | Tipo: {dtype:<14} | Nulos: {nulls:>6} | "
        f"Unicos: {unique:>6,}"
    )


def _line(char: str = "-", n: int = 80) -> None:
    print(char * n)


# ── Subcommand handlers ───────────────────────────────────────────────

def _handle_describe(df: pd.DataFrame, args: CliArgs) -> None:
    pp = Preprocessing(df)
    _header(f"DESCRIPTIVE STATISTICS  -  {args.file}")
    if args.numeric:
        print("\n  Numeric variables:")
        _line("-")
        print(pp.describe_numeric())
    elif args.categorical:
        print("\n  Categorical variables:")
        _line("-")
        print(pp.describe_categorical() if hasattr(pp, "describe_categorical")
              else df.describe(include=["object", "category"]))
    else:
        print("\n  Numeric variables:")
        _line("-")
        print(pp.describe_numeric())
        print("\n  Categorical variables:")
        _line("-")
        print(pp.describe_categorical() if hasattr(pp, "describe_categorical")
              else df.describe(include=["object", "category"]))
    print()


def _handle_data(df: pd.DataFrame, args: CliArgs) -> None:
    _header(f"DATASET SUMMARY  -  {args.file}")
    print(f"\n  Rows:    {df.shape[0]:>8,}")
    print(f"  Columns: {df.shape[1]:>8,}")
    print(f"\n  Columns:")
    _line("-")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        nulls = int(df[col].isnull().sum())
        unique = int(df[col].nunique())
        print(f"  {i:2d}. {_fmt_col(col, str(dtype), nulls, unique)}")
    if args.types:
        print(f"\n  Data types:")
        _line("-")
        print(df.dtypes.to_string())
    if args.missing:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            pct = (missing / len(df)) * 100
            print(f"\n  Missing values:")
            _line("-")
            for col, n in missing.items():
                print(f"    * {col}: {n:,} ({pct[col]:.1f}%)")
        else:
            print(f"\n  No missing values")
    if args.summary:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            print(f"\n  Numeric summary:")
            _line("-")
            for col in numeric_cols[:5]:
                print(f"    * {col}:")
                print(f"        Min: {df[col].min():>10.2f}  Max: {df[col].max():>10.2f}")
                print(f"        Mean: {df[col].mean():>9.2f}  Median: {df[col].median():>8.2f}")
            if len(numeric_cols) > 5:
                print(f"    ... and {len(numeric_cols) - 5} more numeric columns")
    print()


def _handle_preview(df: pd.DataFrame, args: CliArgs) -> None:
    _header(f"DATA PREVIEW  -  {args.file}")
    if args.sample:
        n = min(args.rows, len(df))
        print(f"\n  Random sample ({n} rows):")
        _line("-")
        print(df.sample(n).to_string(index=False))
    else:
        n = min(args.rows, len(df))
        print(f"\n  First {n} rows:")
        _line("-")
        print(df.head(n).to_string(index=False))
    print()


def _handle_quality(df: pd.DataFrame, args: CliArgs) -> None:
    pp = Preprocessing(df)
    _header(f"DATA QUALITY  -  {args.file}")
    report = pp.data_quality()
    print(report)
    if args.verbose and hasattr(pp, "missing_details"):
        print("\n  Missing value details:")
        _line("-")
        print(pp.missing_details())
    print()


def _handle_info(df: pd.DataFrame, args: CliArgs) -> None:
    _header(f"DATASET INFO  -  {args.file}")
    print(f"\n  Dimensions:  {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    print(f"  Memory:      {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"\n  Columns ({len(df.columns)}):")
    _line("-")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:3d}. {col}")
    if args.detailed:
        print(f"\n  Data types:")
        _line("-")
        print(df.dtypes.to_string())
        nulls = df.isnull().sum()
        if nulls.sum() > 0:
            pct = (nulls / len(df)) * 100
            print(f"\n  Null values:")
            _line("-")
            null_df = pd.DataFrame({"Nulos": nulls, "Porcentaje": pct})
            print(null_df[null_df["Nulos"] > 0].to_string())
        else:
            print(f"\n  No null values")
        print(f"\n  Unique values per column:")
        _line("-")
        for col in df.columns:
            print(f"    {col}: {df[col].nunique():,}")
    print()


# ── Command dispatcher ────────────────────────────────────────────────

_HANDLERS = {
    "describe": _handle_describe,
    "data": _handle_data,
    "preview": _handle_preview,
    "quality": _handle_quality,
    "info": _handle_info,
}


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="statslibx",
        description="Statslibx - Data analysis from the terminal",
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("describe", help="Descriptive statistics")
    p.add_argument("file")
    p.add_argument("-n", "--numeric", action="store_true")
    p.add_argument("-c", "--categorical", action="store_true")

    p = sub.add_parser("data", help="Dataset summary")
    p.add_argument("file")
    p.add_argument("-s", "--summary", action="store_true")
    p.add_argument("-t", "--types", action="store_true")
    p.add_argument("-m", "--missing", action="store_true")

    p = sub.add_parser("preview", help="Preview rows")
    p.add_argument("file")
    p.add_argument("-n", "--rows", type=int, default=5)
    p.add_argument("--sample", action="store_true")

    p = sub.add_parser("quality", help="Data quality report")
    p.add_argument("file")
    p.add_argument("-v", "--verbose", action="store_true")

    p = sub.add_parser("info", help="Dataset information")
    p.add_argument("file")
    p.add_argument("-d", "--detailed", action="store_true")

    parsed = parser.parse_args()

    if not parsed.command:
        import statslibx
        statslibx.welcome()
        return

    args = CliArgs(
        command=parsed.command,
        file=parsed.file,
        numeric=getattr(parsed, "numeric", False),
        categorical=getattr(parsed, "categorical", False),
        verbose=getattr(parsed, "verbose", False),
        rows=getattr(parsed, "rows", 5),
        sample=getattr(parsed, "sample", False),
        detailed=getattr(parsed, "detailed", False),
        summary=getattr(parsed, "summary", False),
        types=getattr(parsed, "types", False),
        missing=getattr(parsed, "missing", False),
    )

    try:
        df = load_dataset(args.file)
    except FileNotFoundError:
        print(f"Error: dataset '{args.file}' not found.")
        return

    if df is None or df.empty:
        print(f"Error: could not load data from '{args.file}'.")
        return

    handler = _HANDLERS.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}")
        return

    handler(df, args)


if __name__ == "__main__":
    main()
