import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
import mediapipe as mp
import csv
from pydub import AudioSegment
from pydub.silence import detect_silence

MAX_FRAMES_PER_SEGMENT = 500
MIN_SILENCE_DURATION = 750
SILENCE_THRESHOLD = -30

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def extract_audio(video_path):
    try:
        return AudioSegment.from_file(video_path)
    except Exception as e:
        print(f"Audio extraction failed for {video_path}: {e}")
        return None

def detect_silence_frames(audio_segment, frame_rate, total_frames):
    if audio_segment is None:
        return set()

    silence_intervals = detect_silence(
        audio_segment,
        min_silence_len=MIN_SILENCE_DURATION,
        silence_thresh=SILENCE_THRESHOLD
    )

    silence_frames = set()
    for start_ms, end_ms in silence_intervals:
        start_frame = int((start_ms / 1000) * frame_rate)
        end_frame = int((end_ms / 1000) * frame_rate)
        silence_frames.update(range(
            max(0, start_frame),
            min(total_frames, end_frame + 1)
        ))

    return silence_frames

def calculate_lip_average_distance(frame, frame_size):
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = 256, 256
    
    vertical_distances = []
    upper_lip_indices = [191, 80, 81, 82, 13, 312, 311, 310]
    lower_lip_indices = [95, 88, 178, 87, 14, 317, 402, 318]
    
    for upper, lower in zip(upper_lip_indices, lower_lip_indices):
        y_upper = int(landmarks[upper].y * h)
        y_lower = int(landmarks[lower].y * h)
        vertical_distances.append(abs(y_upper - y_lower))

    return np.mean(vertical_distances) if vertical_distances else None

def process_video(video_path, silence_frames):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return float('nan'), 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_process_frames = min(MAX_FRAMES_PER_SEGMENT, total_frames)

        silent_frame_list = sorted(silence_frames)

        avg_distances = []

        for f_idx in silent_frame_list:
            if f_idx >= max_process_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (256, 256))

            
            avg_distance = calculate_lip_average_distance(
                frame,
                (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            )
            
            if avg_distance is not None:
                avg_distances.append(avg_distance)

        cap.release()

        if len(avg_distances) < 2:
            return float('nan'), len(silence_frames)

        return np.var(avg_distances), len(silence_frames)

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return float('nan'), 0



def main():
    parser = argparse.ArgumentParser(description='Silent Mouth Movement Analysis')
    parser.add_argument('--video_txt', required=True, help='Input video list')
    parser.add_argument('--output_txt', required=True, help='Output CSV file')
    
    args = parser.parse_args()

    with open(args.video_txt, 'r') as f, open(args.output_txt, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['Video Path', 'Silence Variance', 'Silence Frames'])
        
        variances = []
        video_count = 0

        for video_path in tqdm([l.strip() for l in f if l.strip()], desc="Processing videos"):
            video_count += 1
            if not os.path.exists(video_path):
                writer.writerow([video_path, 'File not found', 0])
                continue

            audio = extract_audio(video_path)
            if audio is None:
                writer.writerow([video_path, 'Audio error', 0])
                continue

            cap = cv2.VideoCapture(video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            silence_frames = detect_silence_frames(
                audio_segment=audio,
                frame_rate=frame_rate,
                total_frames=total_frames
            )

            if not silence_frames:
                writer.writerow([video_path, 'No silence', 0])
                continue

            variance, sil_count = process_video(video_path, silence_frames)
            
            writer.writerow([
                video_path,
                f"{variance:.4f}" if not np.isnan(variance) else "NaN",
                sil_count
            ])
            
            if not np.isnan(variance):
                variances.append(variance)
                
            #print(f"average var : {np.nanmean(variances)}")

        avg_variance = np.mean(variances) if variances else np.nan
        
        writer.writerow([])
        writer.writerow(['Average Variance:', f"{avg_variance:.4f}", ''])
        writer.writerow(['Processed Frames per Video:', MAX_FRAMES_PER_SEGMENT, ''])
        
        print(f"\nAnalysis Complete")
        print(f"Total videos processed: {video_count}")
        print(f"Average silence variance: {avg_variance:.4f}")
        if variances:
            print(f"Variance range: {np.min(variances):.4f} - {np.max(variances):.4f}")
            print(f"Standard deviation: {np.std(variances):.4f}")
        print(f"Max frames processed per video: {MAX_FRAMES_PER_SEGMENT}")

if __name__ == "__main__":
    main()
    

# python silent_lip_stability.py --video_txt ../input.txt --output_txt output.txt
