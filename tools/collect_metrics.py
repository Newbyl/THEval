"""Compatibility wrapper for the THEval metric-output collector."""

from theval.cli.collect_metrics import main


if __name__ == "__main__":
    raise SystemExit(main())
