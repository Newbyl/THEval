from __future__ import annotations

import argparse
import sys

from theval.collect import append_or_write, collect_metric_row
from theval.scoring import METRICS


ARGUMENTS = {
    "Global aesthetic": "global_aesthetic",
    "Mouth quality": "mouth_quality",
    "Face quality": "face_quality",
    "Lip dynamics": "lip_dynamics",
    "Head motion dynamics": "head_motion_dynamics",
    "Eyebrow dynamics": "eyebrow_dynamics",
    "Silent lip stability": "silent_lip_stability",
    "Lip sync": "lip_sync",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect eight THEval metric output files into one raw metric CSV row."
    )
    parser.add_argument("--model", required=True, help="Method/model name for the output row.")
    parser.add_argument("--output", required=True, help="Raw metric CSV to write.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to --output if it already exists instead of overwriting it.",
    )
    for metric, argument in ARGUMENTS.items():
        parser.add_argument(f"--{argument.replace('_', '-')}", required=True, help=f"{metric} output file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = {metric: getattr(args, argument) for metric, argument in ARGUMENTS.items()}
    try:
        row = collect_metric_row(args.model, files)
        append_or_write(row, args.output, append=args.append)
    except Exception as exc:
        print(f"theval-collect-metrics: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote metrics for {args.model} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
