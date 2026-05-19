from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract WAV audio files from videos listed in a text file."
    )
    parser.add_argument("--video_txt", required=True, help="Text file containing one video path per line.")
    parser.add_argument(
        "--output_dir",
        "--audio_folder",
        dest="output_dir",
        required=True,
        help="Directory where extracted .wav files will be written.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Output audio sample rate in Hz. Default: 16000.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of output audio channels. Default: 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .wav files instead of skipping them.",
    )
    parser.add_argument(
        "--manifest",
        help="Optional CSV manifest mapping each video path to its extracted audio path.",
    )
    return parser


def read_video_paths(txt_file: str | Path) -> list[Path]:
    path = Path(txt_file)
    if not path.is_file():
        raise FileNotFoundError(f"Video list file not found: {path}")
    return [Path(line.strip()).expanduser() for line in path.read_text().splitlines() if line.strip()]


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH. Install ffmpeg before extracting audio.")


def extract_audio(
    video_path: Path,
    wav_path: Path,
    sample_rate: int,
    channels: int,
    overwrite: bool,
) -> tuple[bool, str]:
    if not video_path.is_file():
        return False, "missing video"

    if wav_path.exists() and not overwrite:
        return True, "skipped existing"

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        str(wav_path),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"ffmpeg exited with {result.returncode}"
        return False, message
    return True, "extracted"


def write_manifest(rows: list[dict[str, str]], manifest_path: str | Path) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Video Path", "Audio Path", "Status"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        check_ffmpeg()
        video_paths = read_video_paths(args.video_txt)
    except Exception as exc:
        print(f"theval-extract-audio: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem_counts = Counter(path.stem for path in video_paths)
    duplicate_stems = sorted(stem for stem, count in stem_counts.items() if count > 1)
    if duplicate_stems:
        print(
            "Warning: duplicate video basenames will map to the same WAV filename: "
            + ", ".join(duplicate_stems),
            file=sys.stderr,
        )

    rows: list[dict[str, str]] = []
    extracted = skipped = failed = 0

    for video_path in video_paths:
        wav_path = output_dir / f"{video_path.stem}.wav"
        ok, status = extract_audio(
            video_path=video_path,
            wav_path=wav_path,
            sample_rate=args.sample_rate,
            channels=args.channels,
            overwrite=args.overwrite,
        )

        if ok and status == "extracted":
            extracted += 1
        elif ok:
            skipped += 1
        else:
            failed += 1
            print(f"Failed: {video_path} -> {status}", file=sys.stderr)

        rows.append(
            {
                "Video Path": str(video_path),
                "Audio Path": str(wav_path),
                "Status": status,
            }
        )

    if args.manifest:
        write_manifest(rows, args.manifest)

    print(
        f"Audio extraction complete: {extracted} extracted, "
        f"{skipped} skipped, {failed} failed. Output: {output_dir}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
