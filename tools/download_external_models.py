"""Download optional external model code and checkpoints used by THEval metrics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
ADAFACE_DIR = MODELS_DIR / "AdaFace"
FACEXFORMER_DIR = MODELS_DIR / "facexformer"


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


def ensure_adaface() -> None:
    if not ADAFACE_DIR.exists():
        run(["git", "clone", "https://github.com/mk-minchul/AdaFace.git", str(ADAFACE_DIR)])

    checkpoint_dir = ADAFACE_DIR / "pretrained"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "adaface_ir50_ms1mv2.ckpt"
    if checkpoint_path.exists():
        print(f"AdaFace checkpoint already exists: {checkpoint_path}")
        return

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install gdown first: pip install gdown") from exc

    print(f"Downloading AdaFace R50 MS1MV2 checkpoint to {checkpoint_path}")
    gdown.download(id="1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI", output=str(checkpoint_path), quiet=False)


def ensure_facexformer() -> None:
    FACEXFORMER_DIR.mkdir(parents=True, exist_ok=True)

    if not (FACEXFORMER_DIR / "network").exists():
        run(["git", "clone", "https://github.com/Kartik-3004/facexformer.git", str(FACEXFORMER_DIR)])

    checkpoint_path = FACEXFORMER_DIR / "ckpts" / "model.pt"
    if checkpoint_path.exists():
        print(f"FaceXFormer checkpoint already exists: {checkpoint_path}")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub first: pip install huggingface-hub") from exc

    print(f"Downloading FaceXFormer checkpoint to {checkpoint_path}")
    hf_hub_download(
        repo_id="kartiknarayan/facexformer",
        filename="ckpts/model.pt",
        local_dir=str(FACEXFORMER_DIR),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaface", action="store_true", help="Download AdaFace code/checkpoint.")
    parser.add_argument("--facexformer", action="store_true", help="Download FaceXFormer code/checkpoint.")
    parser.add_argument("--all", action="store_true", help="Download all optional external models.")
    args = parser.parse_args(argv)

    if not (args.all or args.adaface or args.facexformer):
        parser.error("Choose --all, --adaface, or --facexformer.")

    try:
        if args.all or args.adaface:
            ensure_adaface()
        if args.all or args.facexformer:
            ensure_facexformer()
    except Exception as exc:
        print(f"download_external_models: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
