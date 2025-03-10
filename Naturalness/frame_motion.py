import cv2
import numpy as np
import os
from typing import List
import sys
from tqdm import tqdm
import argparse


# Segmentation parameters
MAX_FRAMES_PER_SEGMENT = 500
NUM_SEGMENTS = 1
FRAME_THRESHOLD = MAX_FRAMES_PER_SEGMENT * NUM_SEGMENTS

def read_video_paths(txt_file: str) -> List[str]:
    if not os.path.exists(txt_file):
        print(f"Error: The file {txt_file} does not exist.")
        sys.exit(1)
    
    with open(txt_file, 'r') as file:
        video_paths = [line.strip() for line in file if line.strip()]
    
    if not video_paths:
        print(f"Error: No video paths found in {txt_file}.")
        sys.exit(1)
    
    return video_paths

def compute_mae_between_frames(frame1: np.ndarray, frame2: np.ndarray) -> float:
    if frame1.shape != frame2.shape:
        print("Warning: Frame dimensions do not match. Resizing frames to match.")
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    
    # Convert frames to float32 for accurate computation
    frame1 = frame1.astype(np.float32)
    frame2 = frame2.astype(np.float32)
    
    # Compute absolute difference
    abs_diff = np.abs(frame1 - frame2)
    
    # Compute MAE
    mae = np.mean(abs_diff)
    
    return mae

def process_video_segment(video_path: str, start_frame: int, end_frame: int) -> float:
    if not os.path.exists(video_path):
        print(f"Warning: Video file {video_path} does not exist. Skipping segment.")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Unable to open video file {video_path}. Skipping segment.")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_to_process = end_frame - start_frame
    mae_values = []
    frame_count = 0

    ret, prev_frame = cap.read()
    if not ret:
        print(f"Warning: Unable to read frame {start_frame} of {video_path}. Skipping segment.")
        cap.release()
        return None

    while frame_count < frames_to_process - 1:
        ret, curr_frame = cap.read()
        if not ret:
            break

        mae = compute_mae_between_frames(prev_frame, curr_frame)
        mae_values.append(mae)
        frame_count += 1
        prev_frame = curr_frame

    cap.release()

    if not mae_values:
        print(f"Warning: No frame pairs found in segment {start_frame}-{end_frame} of {video_path}.")
        return None

    average_mae = np.mean(mae_values)
    return average_mae

def evaluate_videos(video_paths: List[str], output_path: str,
                   max_frames_per_segment: int = MAX_FRAMES_PER_SEGMENT,
                   num_segments: int = NUM_SEGMENTS):
    metrics_list = []

    with open(output_path, 'w') as outfile:
        outfile.write("Video Path,Segment,Start Frame,End Frame,Mean Absolute Error\n")

        with tqdm(total=len(video_paths), desc="Evaluating Videos") as pbar_videos:
            for video_path in video_paths:
                if not os.path.exists(video_path):
                    print(f"Error: Video file {video_path} does not exist. Skipping.")
                    pbar_videos.update(1)
                    continue

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error: Cannot open video {video_path}. Skipping.")
                    pbar_videos.update(1)
                    continue
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                actual_num_segments = num_segments
                if total_frames > max_frames_per_segment * num_segments:
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
                    mae = process_video_segment(video_path, start_frame=start, end_frame=end)
                    if mae is not None:
                        metrics_list.append(mae)
                        segment_label = f"Segment_{idx}"
                        outfile.write(f"{video_path},{segment_label},{start},{end},{mae:.6f}\n")
                    else:
                        outfile.write(f"{video_path},{segment_label},{start},{end},NaN\n")
                
                pbar_videos.update(1)

    if not metrics_list:
        print("No valid MAE metrics computed. Please check your videos and paths.")
        return

    valid_mae = [mae for mae in metrics_list if not np.isnan(mae)]
    if valid_mae:
        overall_mean_mae = np.mean(valid_mae)
    else:
        overall_mean_mae = float('nan')

    with open(output_path, 'a') as outfile:
        outfile.write("\n=== Evaluation Summary ===\n")
        outfile.write(f"Number of video segments processed: {len(metrics_list)}\n")
        outfile.write(f"Average Mean Absolute Error: {overall_mean_mae:.6f}\n")

    print("\n=== Evaluation Results ===")
    print(f"Number of video segments processed: {len(metrics_list)}")
    print(f"Average Mean Absolute Error: {overall_mean_mae:.6f}")
    print(f"\nDetailed metrics have been saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compute Mean Absolute Error (MAE) for Video Segments.")
    parser.add_argument('--video_txt', type=str, required=True, help="Path to the text file containing video paths.")
    parser.add_argument('--output_txt', type=str, required=True, help="Path to the output text file to save metrics.")
    args = parser.parse_args()

    video_paths = read_video_paths(args.video_txt)
    if not video_paths:
        print("No video paths found. Exiting.")
        return

    evaluate_videos(
        video_paths,
        output_path=args.output_txt,
        max_frames_per_segment=MAX_FRAMES_PER_SEGMENT,
        num_segments=NUM_SEGMENTS
    )

if __name__ == "__main__":
    main()
    

# python frame_motion.py --video_txt ../input.txt --output_txt output.txt



