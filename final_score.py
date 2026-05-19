"""Compatibility wrapper for the THEval final-score CLI."""

from theval.cli.score import main


if __name__ == "__main__":
    raise SystemExit(main())
