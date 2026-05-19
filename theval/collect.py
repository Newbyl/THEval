"""Collect aggregate metric values from THEval metric output files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from theval.scoring import METRICS


SUMMARY_PATTERNS = {
    "Global aesthetic": [r"Overall Average Aesthetics:\s*([-+0-9.eE]+)"],
    "Mouth quality": [r"Overall Average Quality:\s*([-+0-9.eE]+)"],
    "Face quality": [r"Overall Average Quality:\s*([-+0-9.eE]+)"],
    "Lip dynamics": [r"Average STD Over Time:\s*([-+0-9.eE]+)"],
    "Head motion dynamics": [r"Mean head motion dynamics:\s*([-+0-9.eE]+)"],
    "Eyebrow dynamics": [r"Average Micro-Expression Intensity:\s*([-+0-9.eE]+)"],
    "Silent lip stability": [r"Average Silent MAD:\s*,?\s*([-+0-9.eE]+)"],
    "Lip sync": [r"Average Mean Difference:\s*([-+0-9.eE]+)"],
}


def extract_summary_value(metric: str, output_file: str | Path) -> float:
    """Extract one aggregate metric value from a metric output file."""

    path = Path(output_file)
    text = path.read_text()
    for pattern in SUMMARY_PATTERNS[metric]:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    raise ValueError(f"Could not find a '{metric}' summary value in {path}")


def collect_metric_row(model_name: str, files: dict[str, str | Path]) -> pd.DataFrame:
    """Build a one-row raw metric table for a method."""

    missing = [metric for metric in METRICS if metric not in files or files[metric] is None]
    if missing:
        raise ValueError("Missing output file(s) for: " + ", ".join(missing))

    row = {"Model": model_name}
    for metric in METRICS:
        row[metric] = extract_summary_value(metric, files[metric])
    return pd.DataFrame([row])


def append_or_write(row: pd.DataFrame, output_csv: str | Path, append: bool = False) -> None:
    """Write a collected metric row to CSV."""

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if append and output_path.exists():
        existing = pd.read_csv(output_path)
        merged = pd.concat([existing, row], ignore_index=True)
        merged.to_csv(output_path, index=False)
    else:
        row.to_csv(output_path, index=False)
