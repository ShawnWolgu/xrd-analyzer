#!/usr/bin/env python3
"""Compare two tabular regression artifacts with explicit numeric tolerances."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _read_table(path: Path, sheet: str | None) -> pd.DataFrame:
    """读取 CSV、TSV 或 Excel 表格。"""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _sort_frame(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """按唯一键排序并重置行号。"""
    if not keys:
        return frame.reset_index(drop=True)

    missing = [key for key in keys if key not in frame.columns]
    if missing:
        raise ValueError(f"Missing key columns: {missing}")
    if frame.duplicated(keys).any():
        raise ValueError(f"Key columns are not unique: {keys}")
    return frame.sort_values(keys).reset_index(drop=True)


def compare_frames(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    keys: list[str],
    rtol: float,
    atol: float,
) -> list[str]:
    """比较两个表格并返回可读的差异列表。"""
    problems: list[str] = []

    if list(baseline.columns) != list(candidate.columns):
        return [
            "Column mismatch:\n"
            f"  baseline={list(baseline.columns)}\n"
            f"  candidate={list(candidate.columns)}"
        ]

    baseline = _sort_frame(baseline, keys)
    candidate = _sort_frame(candidate, keys)

    if baseline.shape != candidate.shape:
        return [f"Shape mismatch: baseline={baseline.shape}, candidate={candidate.shape}"]

    for column in baseline.columns:
        left = baseline[column]
        right = candidate[column]

        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            left_values = left.to_numpy(dtype=float)
            right_values = right.to_numpy(dtype=float)
            matches = np.isclose(
                left_values,
                right_values,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
            if not np.all(matches):
                finite = np.isfinite(left_values) & np.isfinite(right_values)
                absolute = np.abs(left_values[finite] - right_values[finite])
                denominator = np.maximum(np.abs(left_values[finite]), atol)
                relative = absolute / denominator
                problems.append(
                    f"{column}: {int((~matches).sum())} numeric mismatches; "
                    f"max_abs={absolute.max() if absolute.size else float('nan'):.6g}; "
                    f"max_rel={relative.max() if relative.size else float('nan'):.6g}"
                )
            continue

        matches = left.fillna("<NA>").astype(str).eq(right.fillna("<NA>").astype(str))
        if not matches.all():
            problems.append(f"{column}: {int((~matches).sum())} text mismatches")

    return problems


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tabular XRD outputs without treating the baseline as ground truth."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--sheet", help="Excel sheet name; defaults to the first sheet")
    parser.add_argument("--key", nargs="*", default=[], help="Unique columns used to sort rows")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    return parser.parse_args()


def main() -> int:
    """运行命令行比较并返回适合 CI 的退出码。"""
    args = _parse_args()
    try:
        baseline = _read_table(args.baseline, args.sheet)
        candidate = _read_table(args.candidate, args.sheet)
        problems = compare_frames(
            baseline,
            candidate,
            keys=args.key,
            rtol=args.rtol,
            atol=args.atol,
        )
    except (OSError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if problems:
        print("REGRESSION DIFFERENCES DETECTED")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print(f"REGRESSION-ONLY MATCH: tables agree within rtol={args.rtol:g}, atol={args.atol:g}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
