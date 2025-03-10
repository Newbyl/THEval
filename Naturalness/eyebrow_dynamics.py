import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from tqdm import tqdm
import argparse


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

FACEMESH_LEFT_EYEBROW = frozenset([
    (276, 283), (283, 282), (282, 295),
    (295, 285), (300, 293), (293, 334),
    (334, 296), (296, 336)
])
FACEMESH_LEFT_EYE = frozenset([
    (263, 249), (249, 390), (390, 373), (373, 374),
    (374, 380), (380, 381), (381, 382), (382, 362),
    (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362)
])

FACEMESH_RIGHT_EYEBROW = frozenset([
    (46, 53), (53, 52), (52, 65), (65, 55),
    (70, 63), (63, 105), (105, 66), (66, 107)
])
FACEMESH_RIGHT_EYE = frozenset([
    (33, 7), (7, 163), (163, 144), (144, 145),
    (145, 153), (153, 154), (154, 155), (155, 133),
    (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133)
])

def extract_unique_indices(connection_set):
    idxs = set()
    for (i, j) in connection_set:
        idxs.add(i)
        idxs.add(j)
    return sorted(list(idxs))

left_eyebrow_idxs = extract_unique_indices(FACEMESH_LEFT_EYEBROW)
left_eye_idxs     = extract_unique_indices(FACEMESH_LEFT_EYE)

right_eyebrow_idxs = extract_unique_indices(FACEMESH_RIGHT_EYEBROW)
right_eye_idxs     = extract_unique_indices(FACEMESH_RIGHT_EYE)

def get_avg_xy(frame_shape, face_landmarks, landmark_indices):
    h, w, _ = frame_shape
    coords = []
    for idx in landmark_indices:
        x_norm = face_landmarks.landmark[idx].x
        y_norm = face_landmarks.landmark[idx].y
        x_px = x_norm * w
        y_px = y_norm * h
        coords.append((x_px, y_px))
    coords = np.array(coords)
    return coords.mean(axis=0)

def euclidean_distance(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)

def get_face_scale(frame_shape, face_landmarks):
    left_eye_center  = get_avg_xy(frame_shape, face_landmarks, left_eye_idxs)
    right_eye_center = get_avg_xy(frame_shape, face_landmarks, right_eye_idxs)
    scale = euclidean_distance(left_eye_center, right_eye_center)
    return scale

def compute_micro_expression_intensity(video_path, start_frame=0, end_frame=200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(end_frame, frame_count)
    frames_to_process = end_frame - start_frame

    prev_left_dist = None
    prev_right_dist = None

    sum_of_diffs_left = 0.0
    sum_of_diffs_right = 0.0
    num_transitions_left = 0
    num_transitions_right = 0

    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            scale = get_face_scale(frame.shape, face_landmarks)
            if scale < 1e-3:
                #pbar.update(1)
                current_frame += 1
                continue

            left_eyebrow_center = get_avg_xy(frame.shape, face_landmarks, left_eyebrow_idxs)
            left_eye_center     = get_avg_xy(frame.shape, face_landmarks, left_eye_idxs)
            left_dist_px        = euclidean_distance(left_eyebrow_center, left_eye_center)
            left_dist_norm      = left_dist_px / scale

            right_eyebrow_center = get_avg_xy(frame.shape, face_landmarks, right_eyebrow_idxs)
            right_eye_center     = get_avg_xy(frame.shape, face_landmarks, right_eye_idxs)
            right_dist_px        = euclidean_distance(right_eyebrow_center, right_eye_center)
            right_dist_norm      = right_dist_px / scale

            if prev_left_dist is not None:
                sum_of_diffs_left += abs(left_dist_norm - prev_left_dist)
                num_transitions_left += 1
            if prev_right_dist is not None:
                sum_of_diffs_right += abs(right_dist_norm - prev_right_dist)
                num_transitions_right += 1

            prev_left_dist = left_dist_norm
            prev_right_dist = right_dist_norm

        
        current_frame += 1

    cap.release()

    total_transitions = num_transitions_left + num_transitions_right
    total_sum_of_diffs = sum_of_diffs_left + sum_of_diffs_right

    if total_transitions > 0:
        mei = total_sum_of_diffs / total_transitions
    else:
        mei = 0.0

    return mei

def read_video_paths(txt_file: str):
    if not os.path.exists(txt_file):
        print(f"Error: The file {txt_file} does not exist.")
        sys.exit(1)
    
    with open(txt_file, 'r') as file:
        video_paths = [line.strip() for line in file if line.strip()]
    
    if not video_paths:
        print(f"Error: No video paths found in {txt_file}.")
        sys.exit(1)
    
    return video_paths

def evaluate_videos(video_paths, output_path, max_frames_per_segment=MAX_FRAMES_PER_SEGMENT, num_segments=NUM_SEGMENTS):
    metrics_list = []

    with open(output_path, 'w') as outfile:
        outfile.write("Video Path,Segment,Start Frame,End Frame,Micro-Expression Intensity\n")

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
                    mei = compute_micro_expression_intensity(video_path, start_frame=start, end_frame=end)
                    if mei is not None:
                        metrics_list.append(mei)
                        segment_label = f"Segment_{idx}"
                        outfile.write(f"{video_path},{segment_label},{start},{end},{mei:.6f}\n")
                    else:
                        outfile.write(f"{video_path},{segment_label},{start},{end},0.000000\n")
                
                pbar_videos.update(1)

    if not metrics_list:
        print("No valid metrics to compute. Exiting.")
        return

    mean_mei = np.mean(metrics_list)

    with open(output_path, 'a') as outfile:
        outfile.write("\n=== Evaluation Summary ===\n")
        outfile.write(f"Number of video segments processed: {len(metrics_list)}\n")
        outfile.write(f"Average Micro-Expression Intensity: {mean_mei:.6f}\n")

    print("\n=== Evaluation Results ===")
    print(f"Number of video segments processed: {len(metrics_list)}")
    print(f"Average Micro-Expression Intensity: {mean_mei:.6f}")
    print(f"\nDetailed metrics have been saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compute Micro-Expression Intensity for Video Segments.")
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


# python eyebrow_dynamics.py --video_txt ../input.txt --output_txt output.txt