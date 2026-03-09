#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter results by MASE and num_variates, then compute a global geometric mean "
            "for numeric metric columns on the filtered rows."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV path (e.g., gift-eval/results/TimesFM-2.5_original/all_results.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required_cols = {"dataset", "domain", "num_variates", "eval_metrics/MASE[0.5]"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    filtered = df[(df["eval_metrics/MASE[0.5]"] <= 10) & (df["num_variates"] == 1)].copy()

    metric_cols = [
        c
        for c in filtered.columns
        if pd.api.types.is_numeric_dtype(filtered[c])
        and c != "num_variates"
    ]

    geometric_metrics = {}
    for col in metric_cols:
        values = filtered[col].dropna().astype(float)
        if values.empty:
            geometric_metrics[col] = np.nan
            continue
        if (values <= 0).any():
            geometric_metrics[col] = np.nan
            continue
        geometric_metrics[col] = float(np.exp(np.log(values).mean()))

    output_df = pd.DataFrame(
        [
            {
                "aggregation_level": "global_geometric_mean",
                "rows_before": len(df),
                "rows_after": len(filtered),
                **geometric_metrics,
            }
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)

    print(f"rows_before={len(df)}")
    print(f"rows_after={len(filtered)}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
