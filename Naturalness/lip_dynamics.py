import os
import sys
import cv2
import numpy as np
from itertools import combinations
from typing import List, Tuple, Optional
from tqdm import tqdm
import argparse
import mediapipe as mp

MAX_FRAMES_PER_SEGMENT = 500
NUM_SEGMENTS = 1
FRAME_THRESHOLD = MAX_FRAMES_PER_SEGMENT * NUM_SEGMENTS

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

FACEMESH_LIPS = frozenset([
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375),
    (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
    (37, 0), (0, 267), (267, 269), (269, 270), (270, 409),
    (409, 291), (78, 95), (95, 88), (88, 178), (178, 87),
    (87, 14), (14, 317), (317, 402), (402, 318), (318, 324),
    (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
    (82, 13), (13, 312), (312, 311), (311, 310), (310, 415),
    (415, 308)
])

def extract_unique_indices(connection_set: frozenset) -> List[int]:
    idxs = set()
    for (i, j) in connection_set:
        idxs.add(i)
        idxs.add(j)
    return sorted(list(idxs))

lip_indices = extract_unique_indices(FACEMESH_LIPS)

def get_landmark_xy(frame_shape: Tuple[int, int, int], face_landmarks, idx: int) -> np.ndarray:
    h, w, _ = (256, 256, 256)
    x_norm = face_landmarks.landmark[idx].x
    y_norm = face_landmarks.landmark[idx].y
    return np.array([x_norm * w, y_norm * h])

def compute_lip_shape_vector(frame_shape: Tuple[int, int, int], face_landmarks, lip_idxs: List[int]) -> np.ndarray:
    points = [get_landmark_xy(frame_shape, face_landmarks, i) for i in lip_idxs]

    dist_list = []
    for (i, j) in combinations(range(len(points)), 2):
        dist = np.linalg.norm(points[i] - points[j])
        dist_list.append(dist)

    return np.array(dist_list)

def compute_mouth_diversity(
    video_path: str,
    start_frame: int,
    end_frame: int,
    method: str = 'std_over_time'
) -> Optional[float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Unable to open video file '{video_path}'. Skipping segment.")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    shape_vectors = []
    frame_count = start_frame

    
    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.resize(rgb_frame, (256, 256))
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            shape_vec = compute_lip_shape_vector(frame.shape, face_landmarks, lip_indices)
            shape_vectors.append(shape_vec)
        else:
            shape_vectors.append(None)

        frame_count += 1
        

    cap.release()

    if len(shape_vectors) == 0 or all(vec is None for vec in shape_vectors):
        return 0.0

    valid_shape_vectors = [vec for vec in shape_vectors if vec is not None]
    if len(valid_shape_vectors) == 0:
        return 0.0

    shape_vectors_np = np.array(valid_shape_vectors)

    if method == 'std_over_time':
        std_per_dimension = np.std(shape_vectors_np, axis=0)
        diversity_std = float(np.mean(std_per_dimension))
        return diversity_std
    else:
        raise ValueError("Unknown method. Use 'std_over_time'.")

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

def evaluate_videos(
    video_paths: List[str],
    output_path: str,
    max_frames_per_segment: int = MAX_FRAMES_PER_SEGMENT,
    num_segments: int = NUM_SEGMENTS
):
    metrics_std = []
    video_scores_dict_std = {}

    with open(output_path, 'w') as outfile:
        outfile.write("Video Path,Segment,Start Frame,End Frame,Std_Over_Time\n")

        for video_path in tqdm(video_paths, desc="Evaluating Videos", unit="video"):
            if not os.path.exists(video_path):
                print(f"Error: Video file '{video_path}' does not exist. Skipping.")
                outfile.write(f"{video_path},N/A,N/A,N/A,NaN\n")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Cannot open video '{video_path}'. Skipping.")
                outfile.write(f"{video_path},N/A,N/A,N/A,NaN\n")
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
            
            video_segment_std = []

            for idx, (start, end) in enumerate(segments, 1):
                std_metric = compute_mouth_diversity(
                    video_path,
                    start_frame=start,
                    end_frame=end,
                    method='std_over_time'
                )
                video_segment_std.append(std_metric)
                metrics_std.append(std_metric)

                segment_label = f"Segment_{idx}"
                outfile.write(f"{video_path},{segment_label},{start},{end},{std_metric if std_metric is not None else 'NaN'}\n")
                print(f"Video: {video_path}, Segment: {idx}, Frames: {start}-{end}, STD: {std_metric:.3f}")

            valid_std = [s for s in video_segment_std if s is not None]
            mean_std = np.mean(valid_std) if valid_std else np.nan

            video_scores_dict_std[video_path] = mean_std
            print(f"current metric mean : {np.nanmean(metrics_std)}")
            
            

        if metrics_std:
            overall_mean_std = np.nanmean(metrics_std)
        else:
            overall_mean_std = np.nan

    with open(output_path, 'a') as outfile:
        outfile.write("\n=== Evaluation Summary ===\n")
        outfile.write(f"Number of video segments processed: {len(metrics_std)}\n")
        outfile.write(f"Average STD Over Time: {overall_mean_std:.6f}\n")

def main():
    parser = argparse.ArgumentParser(description='Compute Mouth Movement Diversity Across Multiple Videos.')
    parser.add_argument('--video_txt', type=str, required=True, help="Path to the text file containing video paths.")
    parser.add_argument('--output_txt', type=str, required=True, help="Path to the output text file to save metrics.")
    args = parser.parse_args()
    
    video_paths = read_video_paths(args.video_txt)
    
    evaluate_videos(
        video_paths=video_paths,
        output_path=args.output_txt,
        max_frames_per_segment=MAX_FRAMES_PER_SEGMENT,
        num_segments=NUM_SEGMENTS
    )
    
    average_std = []
    
    with open(args.output_txt, 'r') as f:
        for line in f:
            if line.startswith("Video Path") or line.startswith("===") or not line.strip():
                continue
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            std = parts[4]
            if std != 'NaN':
                average_std.append(float(std))
    
    overall_avg_std = np.mean(average_std) if average_std else 0.0
    
    print("\n=== Evaluation Results ===")
    print(f"Number of video segments processed: {len(average_std)}")
    print(f"Average STD Over Time: {overall_avg_std:.6f}")
    print(f"\nDetailed metrics have been saved to: {args.output_txt}")

if __name__ == "__main__":
    main()

# python lip_dynamics.py --video_txt ../input.txt --output_txt output.txt
