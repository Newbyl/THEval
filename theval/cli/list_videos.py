from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a text file containing video paths for metric scripts."
    )
    parser.add_argument("folders", nargs="+", help="Folders to search recursively.")
    parser.add_argument(
        "-o",
        "--output",
        default="video_paths.txt",
        help="Output text file. Default: video_paths.txt",
    )
    parser.add_argument(
        "--relative-to",
        help="Write paths relative to this directory instead of absolute paths.",
    )
    parser.add_argument(
        "--extensions",
        default=".mp4,.mov,.avi,.mkv,.webm",
        help="Comma-separated video extensions. Default: .mp4,.mov,.avi,.mkv,.webm",
    )
    return parser


def iter_video_paths(folders: list[str], extensions: set[str]) -> list[Path]:
    paths: list[Path] = []
    for folder in folders:
        root = Path(folder).expanduser()
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                paths.append(path.resolve())
    return sorted(paths)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    extensions = {
        ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
        for ext in args.extensions.split(",")
        if ext.strip()
    }
    paths = iter_video_paths(args.folders, extensions)

    relative_to = Path(args.relative_to).resolve() if args.relative_to else None
    output_lines = []
    for path in paths:
        if relative_to is not None:
            output_lines.append(str(path.relative_to(relative_to)))
        else:
            output_lines.append(str(path))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + ("\n" if output_lines else ""))
    print(f"Wrote {len(output_lines)} video path(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
