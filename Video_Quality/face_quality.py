import torch
import pyiqa
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from decord import VideoReader, cpu

MAX_FRAMES_PER_SEGMENT = 500
NUM_SEGMENTS = 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_video_segment(video_file, start_frame, end_frame):
    vr = VideoReader(video_file, ctx=cpu(0))
    total_frames = len(vr)
    end_frame = min(end_frame, total_frames)
    frames = vr.get_batch(range(start_frame, end_frame)).asnumpy()
    return [Image.fromarray(frame) for frame in frames]

def compute_face_quality(video_path, start_frame, end_frame, metric):
    try:
        frames = load_video_segment(video_path, start_frame, end_frame)
        if not frames:
            return None
        # model only supports batch size = 1, run per frame
        scores = []
        for img in frames:
            tensor = T.ToTensor()(img).to(device).unsqueeze(0)
            with torch.no_grad():
                score = metric(tensor)
            scores.append(score.item())
        return float(np.mean(scores))
    except Exception as e:
        print(f"Error processing segment {start_frame}-{end_frame}: {e}")
        return None

def read_video_paths(txt_file):
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"File '{txt_file}' not found.")
    with open(txt_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def evaluate_videos(video_paths, output_path, metric):
    metrics = []
    video_scores = {}

    with open(output_path, 'w') as f:
        f.write("Video Path,Segment,Start Frame,End Frame,Quality_Score\n")
        
        for video_path in tqdm(video_paths, desc="Processing Videos"):
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)
                del vr
            except Exception as e:
                print(f"Error reading {video_path}: {e}")
                f.write(f"{video_path},N/A,N/A,N/A,NaN\n")
                continue

            segment_scores = []
            for seg_num in range(NUM_SEGMENTS):
                start = seg_num * MAX_FRAMES_PER_SEGMENT
                end = start + MAX_FRAMES_PER_SEGMENT
                if start >= total_frames:
                    break
                end = min(end, total_frames)
                
                score = compute_face_quality(video_path, start, end, metric)
                if score is None:
                    continue
                
                segment_scores.append(score)
                metrics.append(score)
                f.write(f"{video_path},Segment_{seg_num+1},{start},{end},{score:.4f}\n")
                f.flush()

            if segment_scores:
                video_scores[video_path] = np.mean(segment_scores)
            else:
                video_scores[video_path] = np.nan

    with open(output_path, 'a') as f:
        f.write("\n=== Summary ===\n")
        f.write(f"Total Segments Processed: {len(metrics)}\n")
        f.write(f"Overall Average Quality: {np.nanmean(metrics):.4f}\n")

def main():
    parser = argparse.ArgumentParser(description='Evaluate face quality in video segments.')
    parser.add_argument('--video_txt', required=True, help='Text file containing video paths.')
    parser.add_argument('--output_txt', required=True, help='Output file for results.')
    args = parser.parse_args()

    print("Loading topiq_nr-face metric...")
    metric = pyiqa.create_metric('topiq_nr-face', device=device)

    try:
        video_paths = read_video_paths(args.video_txt)
    except Exception as e:
        print(e)
        sys.exit(1)

    evaluate_videos(video_paths, args.output_txt, metric)
    print(f"\nResults saved to {args.output_txt}")

if __name__ == "__main__":
    main()

# python face_quality.py --video_txt ../input_files/input.txt --output_txt test.txt