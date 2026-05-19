import argparse
import os
import sys

import mediapipe as mp
import numpy as np
import pyiqa
import torch
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm


BATCH_SIZE = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


FACEMESH_LIPS = [
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82,
    84, 87, 88, 91, 95, 146, 178, 181, 185, 191,
    267, 269, 270, 291, 308, 310, 311, 312, 314,
    317, 318, 321, 324, 375, 402, 405, 409, 415,
]


def load_video(video_file):
    vr = VideoReader(video_file, ctx=cpu(0))
    frames = vr.get_batch(range(len(vr))).asnumpy()
    return [Image.fromarray(frame) for frame in frames]


def calculate_bounding_box(face_landmarks, indices, width, height):
    x_coords = [int(face_landmarks.landmark[i].x * width) for i in indices]
    y_coords = [int(face_landmarks.landmark[i].y * height) for i in indices]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


def crop_region(frame, bbox):
    x_min, y_min, x_max, y_max = bbox
    padding = 10
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, frame.width)
    y_max = min(y_max + padding, frame.height)
    return frame.crop((x_min, y_min, x_max, y_max))


def extract_mouth_regions(frames):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    cropped_mouths = []
    for frame in frames:
        frame_np = np.array(frame)
        results = face_mesh.process(frame_np)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                width, height = frame.size
                mouth_bbox = calculate_bounding_box(face_landmarks, FACEMESH_LIPS, width, height)
                cropped_mouths.append(crop_region(frame, mouth_bbox))
        else:
            cropped_mouths.append(None)

    face_mesh.close()
    return cropped_mouths


def pad_images(images):
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)

    padded = []
    for img in images:
        canvas = Image.new(img.mode, (max_w, max_h), (0, 0, 0))
        canvas.paste(img, (0, 0))
        padded.append(canvas)
    return padded


def score_images(images, metric, batch_size=BATCH_SIZE):
    scores = []
    for start in range(0, len(images), batch_size):
        batch_images = pad_images(images[start:start + batch_size])
        batch = torch.stack([T.ToTensor()(img) for img in batch_images]).to(device)
        with torch.no_grad():
            batch_scores = metric(batch)
        scores.extend(batch_scores.detach().cpu().reshape(-1).tolist())
    return scores


def compute_mouth_quality(video_path, metric):
    try:
        frames = load_video(video_path)
        cropped_mouths = extract_mouth_regions(frames)
        valid_mouths = [img for img in cropped_mouths if img is not None]
        if not valid_mouths:
            return None

        scores = score_images(valid_mouths, metric)
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
        f.write("Video Path,Quality_Score\n")

        for video_path in tqdm(video_paths, desc="Processing Videos"):
            score = compute_mouth_quality(video_path, metric)
            if score is None:
                f.write(f"{video_path},NaN\n")
                continue

            metrics.append(score)
            f.write(f"{video_path},{score:.4f}\n")
            f.flush()

    with open(output_path, "a") as f:
        f.write("\n=== Summary ===\n")
        f.write(f"Overall Average Quality: {np.nanmean(metrics):.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate mouth region quality in videos.")
    parser.add_argument("--video_txt", required=True, help="Text file containing video paths.")
    parser.add_argument("--output_txt", required=True, help="Output file for results.")
    args = parser.parse_args()

    print("Loading MUSIQ metric...")
    metric = pyiqa.create_metric("musiq", device=device)

    try:
        video_paths = read_video_paths(args.video_txt)
    except Exception as e:
        print(e)
        sys.exit(1)

    evaluate_videos(video_paths, args.output_txt, metric)
    print(f"\nResults saved to {args.output_txt}")


if __name__ == "__main__":
    main()
