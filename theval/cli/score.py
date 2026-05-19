from __future__ import annotations

import argparse
import sys
from pathlib import Path

from theval.scoring import DEFAULT_GT_NAME, METRICS, score_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute THEval final scores from a raw aggregate metric CSV. "
            "The CSV must contain one row named GT and one row per method. "
            "Higher final scores are better."
        )
    )
    parser.add_argument("--metrics", required=True, help="Input CSV with raw THEval metrics.")
    parser.add_argument(
        "--output",
        help="Optional output CSV for normalized metric scores, group scores, final scores, and ranks.",
    )
    parser.add_argument(
        "--gt-name",
        default=DEFAULT_GT_NAME,
        help=f"Name of the ground-truth row. Default: {DEFAULT_GT_NAME}",
    )
    parser.add_argument(
        "--model-column",
        default="Model",
        help="Column containing method names when the CSV is not already indexed.",
    )
    parser.add_argument(
        "--weights",
        help=(
            "Optional metric weights as a JSON file, inline JSON object, or "
            "comma-separated Metric=weight pairs. By default every metric has weight 1. "
            "Weights are applied as a weighted average."
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of decimals to print/write. Default: 4.",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="Print the required metric columns and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_metrics:
        for metric in METRICS:
            print(metric)
        return 0

    try:
        scores = score_csv(
            args.metrics,
            gt_name=args.gt_name,
            model_column=args.model_column,
            weights=args.weights,
        )
    except Exception as exc:
        print(f"theval-score: {exc}", file=sys.stderr)
        return 1

    rounded = scores.round(args.precision)
    print(rounded[["Final score", "Rank"]].to_string())

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rounded.to_csv(output_path)
        print(f"\nSaved detailed scores to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
