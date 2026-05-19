"""Final-score computation for THEval metric tables.

THEval compares each method to the ground-truth row metric by metric. Following
Eq. 12 of the paper, each raw metric is converted into a normalized score:

    score = 1 - abs(method_value - gt_value) / gt_value

The final score is the unweighted average of the eight normalized metric scores.
Higher is better.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


METRICS = [
    "Global aesthetic",
    "Mouth quality",
    "Face quality",
    "Lip dynamics",
    "Head motion dynamics",
    "Eyebrow dynamics",
    "Silent lip stability",
    "Lip sync",
]

DEFAULT_GT_NAME = "GT"


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).strip().lower()).strip()


METRIC_ALIASES = {_key(metric): metric for metric in METRICS}
METRIC_ALIASES.update(
    {
        "global aesthetics": "Global aesthetic",
        "global aesthetic score": "Global aesthetic",
        "mouth quality score": "Mouth quality",
        "face quality score": "Face quality",
        "std over time": "Lip dynamics",
        "lip dynamics std": "Lip dynamics",
        "head motion": "Head motion dynamics",
        "head motion dynamic": "Head motion dynamics",
        "eyebrow": "Eyebrow dynamics",
        "eyebrow dynamics score": "Eyebrow dynamics",
        "micro expression intensity": "Eyebrow dynamics",
        "silent mad": "Silent lip stability",
        "silent lip": "Silent lip stability",
        "silence lip stability": "Silent lip stability",
        "mean difference": "Lip sync",
        "lip synchronization": "Lip sync",
        "lipsync": "Lip sync",
    }
)


def canonical_metric_name(name: str) -> str | None:
    """Return the canonical THEval metric name for a CSV column."""

    return METRIC_ALIASES.get(_key(name))


def read_metric_table(path: str | Path, model_column: str = "Model") -> pd.DataFrame:
    """Read a raw aggregate metric CSV.

    The CSV should contain one row per method and a ground-truth row named
    ``GT`` by default. Method names can either live in a ``Model`` column or in
    the first CSV column/index.
    """

    path = Path(path)
    df = pd.read_csv(path)

    if model_column in df.columns:
        df = df.set_index(model_column)
    elif df.columns[0].lower().startswith("unnamed"):
        df = df.set_index(df.columns[0])
    else:
        first_col = df.columns[0]
        if canonical_metric_name(first_col) is None:
            df = df.set_index(first_col)

    df.index = df.index.astype(str)
    return df


def prepare_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rename metric columns to THEval canonical names and validate them."""

    rename = {}
    for column in df.columns:
        canonical = canonical_metric_name(column)
        if canonical is not None:
            rename[column] = canonical

    metrics_df = df.rename(columns=rename)
    missing = [metric for metric in METRICS if metric not in metrics_df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required metric column(s): {missing_text}")

    metrics_df = metrics_df[METRICS].apply(pd.to_numeric, errors="coerce")
    if metrics_df.isna().any().any():
        bad = metrics_df.columns[metrics_df.isna().any()].tolist()
        raise ValueError(
            "All metric values must be numeric. Columns with invalid values: "
            + ", ".join(bad)
        )

    return metrics_df


def compute_normalized_scores(
    metrics_df: pd.DataFrame,
    gt_name: str = DEFAULT_GT_NAME,
    epsilon: float = 1e-9,
) -> pd.DataFrame:
    """Compute per-metric normalized scores relative to the GT row."""

    if gt_name not in metrics_df.index:
        raise ValueError(f"Ground-truth row '{gt_name}' was not found.")

    gt = metrics_df.loc[gt_name].astype(float)
    errors = (metrics_df.astype(float) - gt).abs()

    denominators = gt.abs()
    if (denominators <= epsilon).any():
        zero_metrics = denominators[denominators <= epsilon].index.tolist()
        raise ValueError(
            "GT values must be non-zero for THEval normalization. "
            "Invalid metric(s): " + ", ".join(zero_metrics)
        )

    relative_errors = errors.divide(denominators, axis="columns")
    return 1.0 - relative_errors


def parse_weights(weights: str | Path | Mapping[str, float] | None) -> dict[str, float]:
    """Parse optional metric weights.

    ``weights`` may be a mapping, a JSON file path, an inline JSON object, or a
    comma-separated ``Metric=weight`` string. Unspecified metrics default to 1.
    """

    parsed: Mapping[str, float] = {}
    if weights is None:
        parsed = {}
    elif isinstance(weights, Mapping):
        parsed = weights
    else:
        value = str(weights).strip()
        possible_path = Path(value)
        if possible_path.exists():
            parsed = json.loads(possible_path.read_text())
        elif value.startswith("{"):
            parsed = json.loads(value)
        elif value:
            items = [item.strip() for item in value.split(",") if item.strip()]
            parsed = dict(item.split("=", 1) for item in items)

    canonical_weights = {metric: 1.0 for metric in METRICS}
    unknown = []
    for name, weight in parsed.items():
        canonical = canonical_metric_name(name)
        if canonical is None:
            unknown.append(str(name))
            continue
        canonical_weights[canonical] = float(weight)

    if unknown:
        raise ValueError("Unknown metric weight(s): " + ", ".join(unknown))

    return canonical_weights


def compute_final_scores(
    metrics_df: pd.DataFrame,
    gt_name: str = DEFAULT_GT_NAME,
    weights: str | Path | Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Return normalized metric scores, group scores, final score, and rank."""

    metrics_df = prepare_metric_table(metrics_df)
    normalized = compute_normalized_scores(metrics_df, gt_name=gt_name)
    metric_weights = parse_weights(weights)
    weights_series = pd.Series(metric_weights)

    result = normalized.copy()
    result["Video quality"] = result[
        ["Global aesthetic", "Mouth quality", "Face quality"]
    ].mean(axis=1)
    result["Naturalness"] = result[
        [
            "Lip dynamics",
            "Head motion dynamics",
            "Eyebrow dynamics",
        ]
    ].mean(axis=1)
    result["Synchronization"] = result[["Silent lip stability", "Lip sync"]].mean(axis=1)
    weighted_scores = result[METRICS].multiply(weights_series, axis=1)
    result["Final score"] = weighted_scores.sum(axis=1) / weights_series.sum()
    result["Rank"] = result["Final score"].rank(method="min", ascending=False).astype(int)
    result = result.sort_values(["Final score", "Rank"], ascending=[False, True])
    result.index.name = "Model"
    return result


def score_csv(
    input_csv: str | Path,
    gt_name: str = DEFAULT_GT_NAME,
    model_column: str = "Model",
    weights: str | Path | Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Convenience wrapper for scoring a CSV file."""

    return compute_final_scores(
        read_metric_table(input_csv, model_column=model_column),
        gt_name=gt_name,
        weights=weights,
    )
