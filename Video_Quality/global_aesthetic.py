import argparse
import os
import sys

import numpy as np
import pyiqa
import torch
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm


BATCH_SIZE = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_video(video_file):
    vr = VideoReader(video_file, ctx=cpu(0))
    frames = vr.get_batch(range(len(vr))).asnumpy()
    return [Image.fromarray(frame) for frame in frames]


def score_images(images, metric, batch_size=BATCH_SIZE):
    scores = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]
        batch = torch.stack([T.ToTensor()(img) for img in batch_images]).to(device)
        with torch.no_grad():
            batch_scores = metric(batch)
        scores.extend(batch_scores.detach().cpu().reshape(-1).tolist())
    return scores


def compute_aesthetics_score(video_path, metric):
    try:
        frames = load_video(video_path)
        valid_frames = [img for img in frames if img is not None]
        if not valid_frames:
            return None

        scores = score_images(valid_frames, metric)
        return float(np.mean(scores)) if scores else None
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None


def read_video_paths(txt_file):
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"File '{txt_file}' not found.")
    with open(txt_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def evaluate_videos(video_paths, output_path, metric):
    metrics = []
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        f.write("Video Path,Aesthetic_Score\n")

        for video_path in tqdm(video_paths, desc="Processing Videos"):
            score = compute_aesthetics_score(video_path, metric)
            if score is None:
                f.write(f"{video_path},NaN\n")
                continue

            metrics.append(score)
            f.write(f"{video_path},{score:.4f}\n")
            f.flush()

    with open(output_path, "a") as f:
        f.write("\n=== Summary ===\n")
        f.write(f"Overall Average Aesthetics: {np.nanmean(metrics):.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate global aesthetic quality in videos.")
    parser.add_argument("--video_txt", required=True, help="Text file containing video paths.")
    parser.add_argument("--output_txt", required=True, help="Output file for results.")
    args = parser.parse_args()

    print("Loading TOPIQ IAA metric...")
    metric = pyiqa.create_metric("topiq_iaa", device=device)

    try:
        video_paths = read_video_paths(args.video_txt)
    except Exception as e:
        print(e)
        sys.exit(1)

    evaluate_videos(video_paths, args.output_txt, metric)
    print(f"\nResults saved to {args.output_txt}")


if __name__ == "__main__":
    main()
