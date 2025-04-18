import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional
from facenet_pytorch import MTCNN
from PIL import Image
from itertools import islice

# Add AdaFace to path and imports
sys.path.append('../models/AdaFace')
#print(sys.path.remove("/lustre/fswork/projects/rech/vra/uuf71dx/talking_heads/THEval/models/facexformer"))
from inference import load_pretrained_model, to_input

MAX_FRAMES_PER_SEGMENT = 500
NUM_SEGMENTS = 1
FRAME_THRESHOLD = MAX_FRAMES_PER_SEGMENT * NUM_SEGMENTS

BATCH_SIZE = 128
THRESHOLD = 0.4  # New threshold for frame‐wise identity check (similarity)


def read_video_paths(txt_file: str) -> List[str]:
    if not os.path.exists(txt_file):
        print(f"Error: The file '{txt_file}' does not exist.")
        sys.exit(1)
    
    with open(txt_file, 'r') as file:
        video_paths = [line.strip() for line in file if line.strip()]
    
    if not video_paths:
        print(f"Error: No video paths found in '{txt_file}'.")
        sys.exit(1)
    
    return video_paths

def initialize_models(device: torch.device) -> Tuple[MTCNN, torch.nn.Module]:
    mtcnn = MTCNN(keep_all=False, device=device, post_process=False)
    model = load_pretrained_model('ir_50').to(device).eval()
    return mtcnn, model


def extract_face_embeddings_batch(frames: List[np.ndarray], mtcnn: MTCNN, model: torch.nn.Module, device: torch.device) -> List[Optional[np.ndarray]]:
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    pil_images = [Image.fromarray(img) for img in frames_rgb]
    
    with torch.no_grad():
        boxes, _ = mtcnn.detect(pil_images)

    embeddings = [None] * len(frames)
    for idx, box in enumerate(boxes):
        try:
            if box is None or len(box) == 0:
                continue
            x1, y1, x2, y2 = map(int, box[0])
            face_rgb = frames_rgb[idx][y1:y2, x1:x2]
            if face_rgb.size == 0:
                continue
            face_pil = Image.fromarray(face_rgb)  # PIL expects RGB
            # resize to model's expected input
            face_pil = face_pil.resize((112, 112))
            inp = to_input(face_pil).to(device)   # normalizes to BGR internally
            with torch.no_grad():
                feat, _ = model(inp)
            embeddings[idx] = feat.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
            embeddings[idx] = None
    return embeddings


def compute_cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    if embedding1 is None or embedding2 is None:
        return np.nan
    norm1 = embedding1 / np.linalg.norm(embedding1)
    norm2 = embedding2 / np.linalg.norm(embedding2)

    cosine_similarity = np.dot(norm1, norm2)

    cosine_distance = 1 - cosine_similarity
    return cosine_distance

def batched(iterable, n=1):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

def process_video_segment(video_path: str, start_frame: int, end_frame: int, mtcnn: MTCNN, model: torch.nn.Module, device: torch.device, batch_size: int = BATCH_SIZE) -> float:
    if not os.path.exists(video_path):
        print(f"Warning: Video file '{video_path}' does not exist. Skipping segment.")
        return np.nan
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Unable to open video file '{video_path}'. Skipping segment.")
        return np.nan

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    same_count = 0
    total_count = 0
    key_emb = None  # first valid embedding as reference

    with tqdm(total=end_frame - start_frame, leave=False) as pbar:
        while frame_idx < end_frame:
            batch = []
            for _ in range(min(batch_size, end_frame - frame_idx)):
                ret, frm = cap.read()
                if not ret:
                    break
                batch.append(frm)
                frame_idx += 1
            if not batch:
                break

            embs = extract_face_embeddings_batch(batch, mtcnn, model, device)
            for emb in embs:
                if emb is not None:
                    if key_emb is None:
                        key_emb = emb  # set reference at first valid detection
                    else:
                        # compare current to key frame
                        sim = np.dot(emb / np.linalg.norm(emb), key_emb / np.linalg.norm(key_emb))
                        if sim >= THRESHOLD:
                            same_count += 1
                        total_count += 1
                pbar.update(1)

    cap.release()
    if total_count == 0:
        print(f"Warning: No valid comparisons in {video_path} [{start_frame}-{end_frame}]")
        return float('nan')
    # return percent of frames above threshold
    return (same_count / total_count) * 100.0


def evaluate_videos(video_paths: List[str], output_path: str, mtcnn: MTCNN, model: torch.nn.Module, device: torch.device, max_frames_per_segment: int = MAX_FRAMES_PER_SEGMENT, num_segments: int = NUM_SEGMENTS, batch_size: int = BATCH_SIZE):
    metrics_list = []

    with open(output_path, 'w') as outfile:
        outfile.write("Video Path,Segment,Start,End,Score(%)\n")
        with tqdm(total=len(video_paths), desc="Evaluating Videos") as pbar_videos:
            for video_path in video_paths:
                if not os.path.exists(video_path):
                    print(f"Error: Video file '{video_path}' does not exist. Skipping.")
                    outfile.write(f"{video_path},N/A,N/A,N/A,NaN\n")
                    pbar_videos.update(1)
                    continue

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error: Cannot open video '{video_path}'. Skipping.")
                    outfile.write(f"{video_path},N/A,N/A,N/A,NaN\n")
                    pbar_videos.update(1)
                    continue
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                actual_num_segments = num_segments
                if total_frames > FRAME_THRESHOLD:
                    actual_num_segments = num_segments
                else:
                    actual_num_segments = max(1, total_frames // max_frames_per_segment)
                
                if total_frames <= max_frames_per_segment:
                    segments = [(0, total_frames)]
                else:
                    segments = []
                    for i in range(actual_num_segments):
                        start_frame = i * max_frames_per_segment
                        end_frame = start_frame + max_frames_per_segment
                        if end_frame > total_frames:
                            end_frame = total_frames
                        segments.append((start_frame, end_frame))
                
                for idx, (start, end) in enumerate(segments, 1):
                    score = process_video_segment(video_path, start, end, mtcnn, model, device, batch_size)
                    outfile.write(f"{video_path},Segment_{idx},{start},{end},{score:.2f}\n")
                    print(f"Video: {video_path}, Segment {idx}, Score: {score:.2f}%")
                    metrics_list.append(score)
                
                pbar_videos.update(1)

    if not metrics_list:
        print("No valid metrics to compute. Exiting.")
        return

    overall_mean_score = np.nanmean(metrics_list)

    with open(output_path, 'a') as outfile:
        outfile.write("\n=== Evaluation Summary ===\n")
        outfile.write(f"Number of video segments processed: {len(metrics_list)}\n")
        outfile.write(f"Average Score (%): {overall_mean_score:.2f}\n")

    print("\n=== Evaluation Results ===")
    print(f"Number of video segments processed: {len(metrics_list)}")
    print(f"Average Score (%): {overall_mean_score:.2f}")
    print(f"\nDetailed metrics have been saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compute Identity Preservation Scores Across Multiple Videos.')
    parser.add_argument('--video_txt', type=str, required=True, help="Path to the text file containing video paths.")
    parser.add_argument('--output_txt', type=str, required=True, help="Path to the output text file to save metrics.")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    video_paths = read_video_paths(args.video_txt)
    
    mtcnn, model = initialize_models(device)
    
    evaluate_videos(
        video_paths=video_paths,
        output_path=args.output_txt,
        mtcnn=mtcnn,
        model=model,
        device=device,
        max_frames_per_segment=MAX_FRAMES_PER_SEGMENT,
        num_segments=NUM_SEGMENTS,
        batch_size=BATCH_SIZE
    )

if __name__ == "__main__":
    main()

# python identity_consistency.py --video_txt ../input_files/input.txt --output_txt ../output_files/output.txt