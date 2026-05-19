"""Compatibility wrapper for the THEval video-list CLI."""

from theval.cli.list_videos import main


if __name__ == "__main__":
    raise SystemExit(main())
