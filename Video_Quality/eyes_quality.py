import torch
from transformers import AutoModelForCausalLM
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

def load_video_segment(video_file, start_frame, end_frame):
    vr = VideoReader(video_file, ctx=cpu(0))
    total_frames = len(vr)
    end_frame = min(end_frame, total_frames)
    frames = vr.get_batch(range(start_frame, end_frame)).asnumpy()
    return [Image.fromarray(frame) for frame in frames]

def extract_eyes_region(frames):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    
    # Landmark indices for left and right eyes (combined)
    FACEMESH_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
    FACEMESH_RIGHT_EYE = [362, 263, 387, 373, 380, 381, 382, 384, 385, 386, 387, 388, 390, 466, 388]
    FACEMESH_EYES = FACEMESH_LEFT_EYE + FACEMESH_RIGHT_EYE
    
    cropped_eyes = []
    for frame in frames:
        frame_np = np.array(frame)
        results = face_mesh.process(frame_np)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                width, height = frame.size
                eyes_bbox = calculate_bounding_box(face_landmarks, FACEMESH_EYES, width, height)
                eyes_crop = crop_region(frame, eyes_bbox)
                cropped_eyes.append(eyes_crop)
        else:
            cropped_eyes.append(None)
    face_mesh.close()
    return cropped_eyes

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

def compute_eyes_quality(video_path, start_frame, end_frame, model):
    try:
        frames = load_video_segment(video_path, start_frame, end_frame)
        cropped_eyes = extract_eyes_region(frames)
        valid_eyes = [img for img in cropped_eyes if img is not None]
        if not valid_eyes:
            return None
        scores = model.score([valid_eyes], task_="quality", input_="video")
        return float(scores[0]) if scores else None
    except Exception as e:
        print(f"Error processing segment {start_frame}-{end_frame}: {e}")
        return None

def read_video_paths(txt_file):
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"File '{txt_file}' not found.")
    with open(txt_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def evaluate_videos(video_paths, output_path, model):
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
                
                score = compute_eyes_quality(video_path, start, end, model)
                if score is None:
                    continue
                
                segment_scores.append(score)
                metrics.append(score)
                f.write(f"{video_path},Segment_{seg_num+1},{start},{end},{score:.4f}\n")
                f.flush()

            print(f"mean quality score : {np.nanmean(metrics)}")
            
            if segment_scores:
                video_scores[video_path] = np.mean(segment_scores)
            else:
                video_scores[video_path] = np.nan

    with open(output_path, 'a') as f:
        f.write("\n=== Summary ===\n")
        f.write(f"Total Segments Processed: {len(metrics)}\n")
        f.write(f"Overall Average Quality: {np.nanmean(metrics):.4f}\n")

def main():
    parser = argparse.ArgumentParser(description='Evaluate eye region quality in video segments.')
    parser.add_argument('--video_txt', required=True, help='Text file containing video paths.')
    parser.add_argument('--output_txt', required=True, help='Output file for results.')
    args = parser.parse_args()

    print("Loading qalign model...")
    model = AutoModelForCausalLM.from_pretrained(
        "q-future/one-align",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    try:
        video_paths = read_video_paths(args.video_txt)
    except Exception as e:
        print(e)
        sys.exit(1)

    evaluate_videos(video_paths, args.output_txt, model)
    print(f"\nResults saved to {args.output_txt}")

if __name__ == "__main__":
    main()
    
# python eyes_quality_HDTF.py --video_txt ../input.txt --output_txt output.txt
