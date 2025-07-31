import torch
import pyiqa
import torchvision.transforms as T
from PIL import Image
import mediapipe as mp
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

def extract_mouth_regions(frames):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    
    FACEMESH_LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 
                    84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 
                    267, 269, 270, 291, 308, 310, 311, 312, 314, 
                    317, 318, 321, 324, 375, 402, 405, 409, 415]
    
    cropped_mouths = []
    for frame in frames:
        frame_np = np.array(frame)
        results = face_mesh.process(frame_np)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                width, height = frame.size
                mouth_bbox = calculate_bounding_box(face_landmarks, FACEMESH_LIPS, width, height)
                mouth_crop = crop_region(frame, mouth_bbox)
                cropped_mouths.append(mouth_crop)
        else:
            cropped_mouths.append(None)
    face_mesh.close()
    return cropped_mouths

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

def compute_mouth_quality(video_path, start_frame, end_frame, metric):
    try:
        frames = load_video_segment(video_path, start_frame, end_frame)
        cropped_mouths = extract_mouth_regions(frames)
        valid_mouths = [img for img in cropped_mouths if img is not None]
        if not valid_mouths:
            return None

        # --- added: pad all mouth crops to same size ---
        widths  = [img.width  for img in valid_mouths]
        heights = [img.height for img in valid_mouths]
        max_w, max_h = max(widths), max(heights)
        padded_mouths = []
        for img in valid_mouths:
            canvas = Image.new(img.mode, (max_w, max_h), (0,0,0))
            canvas.paste(img, (0,0))
            padded_mouths.append(canvas)
        # --- end added ---

        # convert PIL -> tensor batch (N,3,H,W), run musiQ
        batch = torch.stack([T.ToTensor()(img) for img in padded_mouths]).to(device)
        with torch.no_grad():
            scores = metric(batch)
        return float(scores.mean().item())
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
                
                score = compute_mouth_quality(video_path, start, end, metric)
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
    parser = argparse.ArgumentParser(description='Evaluate mouth region quality in video segments.')
    parser.add_argument('--video_txt', required=True, help='Text file containing video paths.')
    parser.add_argument('--output_txt', required=True, help='Output file for results.')
    args = parser.parse_args()

    print("Loading musiQ metric...")
    metric = pyiqa.create_metric('musiq', device=device)

    try:
        video_paths = read_video_paths(args.video_txt)
    except Exception as e:
        print(e)
        sys.exit(1)

    evaluate_videos(video_paths, args.output_txt, metric)
    print(f"\nResults saved to {args.output_txt}")

if __name__ == "__main__":
    main()

# python mouth_quality_other.py --video_txt ../input_files/input.txt --output_txt test.txt