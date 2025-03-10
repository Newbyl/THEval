import cv2
import mediapipe as mp
import numpy as np
import librosa
import argparse
import os
import tempfile
from moviepy.editor import VideoFileClip
from scipy.stats import pearsonr
from tqdm import tqdm
from decord import VideoReader, cpu

MAX_FRAMES_PER_SEGMENT = 500
NUM_SEGMENTS = 1

FACEMESH_LIPS = [
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82,
    84, 87, 88, 91, 95, 146, 178, 181, 185, 191,
    267, 269, 270, 291, 308, 310, 311, 312, 314,
    317, 318, 321, 324, 375, 402, 405, 409, 415
]

def extract_audio(filename):
    clip = VideoFileClip(filename)
    audio = clip.audio
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
        audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
        y, sr = librosa.load(temp_audio.name, sr=None)
    return y, sr

def compute_volume(y, sr, frame_times):
    frame_length = int(sr * 0.05)
    hop_length = int(sr * 0.05)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
    return np.interp(frame_times, times, rms)

def compute_mouth_openness(video_path, start_frame, end_frame):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    end_frame = min(end_frame, total_frames)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    openness = []
    frame_times = []
    fps = vr.get_avg_fps()
    
    try:
        frames = vr.get_batch(range(start_frame, end_frame)).asnumpy()
        for i, frame in enumerate(frames):
            results = face_mesh.process(frame)
            frame_idx = start_frame + i
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                upper = np.mean([landmarks[i].y for i in FACEMESH_LIPS if i in [191, 80, 81, 82, 13, 312, 311, 310]])
                lower = np.mean([landmarks[i].y for i in FACEMESH_LIPS if i in [95, 88, 178, 87, 14, 317, 402, 318]])
                openness.append(abs(upper - lower))
            else:
                openness.append(0)
                
            frame_times.append(frame_idx / fps)
        
        openness = (openness-np.min(openness)) / (np.max(openness)-np.min(openness)) 
            
    finally:
        face_mesh.close()
        
    return np.array(frame_times), np.array(openness)

def process_segment_cor(video_path, start_frame, end_frame, y, sr):
    t, o = compute_mouth_openness(video_path, start_frame, end_frame)
    v = compute_volume(y, sr, t)
    return pearsonr(o, v)[0] if len(o) > 1 else np.nan

def process_segment(video_path, start_frame, end_frame, y, sr):
    t, o = compute_mouth_openness(video_path, start_frame, end_frame)
    v = compute_volume(y, sr, t)
    return np.mean(o - v) if len(o) > 1 else np.nan

def read_video_paths(txt_file):
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Video list file not found: {txt_file}")
    with open(txt_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def evaluate_videos(video_paths, output_path):
    metrics = []
    video_scores = {}

    with open(output_path, 'w') as f:
        f.write("Video Path,Segment,Start Frame,End Frame,Correlation_Score\n")
        
        for video_path in tqdm(video_paths, desc="Processing Videos"):
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)
                del vr
                
                y, sr = extract_audio(video_path)
                
                segment_scores = []
                for seg_num in range(NUM_SEGMENTS):
                    start = seg_num * MAX_FRAMES_PER_SEGMENT
                    end = start + MAX_FRAMES_PER_SEGMENT
                    if start >= total_frames:
                        break
                    end = min(end, total_frames)
                    
                    score = process_segment(video_path, start, end, y, sr)
                    if np.isnan(score):
                        continue
                    
                    segment_scores.append(score)
                    metrics.append(score)
                    f.write(f"{video_path},Segment_{seg_num+1},{start},{end},{score:.4f}\n")
                    f.flush()
                    
                    print(f"Current mean diff : {np.mean(metrics)}")
                    print(f"Current std diff : {np.std(metrics)}")

                if segment_scores:
                    video_scores[video_path] = np.mean(segment_scores)
                else:
                    video_scores[video_path] = np.nan

            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                f.write(f"{video_path},N/A,N/A,N/A,NaN\n")
                continue

    with open(output_path, 'a') as f:
        f.write("\n=== Summary ===\n")
        f.write(f"Total Segments Processed: {len(metrics)}\n")
        f.write(f"Overall Average diff: {np.nanmean(metrics):.4f}\n")
        f.write(f"Overall STD diff: {np.nanstd(metrics):.4f}\n")

def main():
    parser = argparse.ArgumentParser(description='Calculate mouth openness-audio volume diff')
    parser.add_argument('--video_txt', required=True, help='Text file containing video paths')
    parser.add_argument('--output_txt', required=True, help='Output text file path')
    args = parser.parse_args()

    try:
        video_paths = read_video_paths(args.video_txt)
    except Exception as e:
        print(f"Error: {e}")
        return

    evaluate_videos(video_paths, args.output_txt)
    print(f"\nResults saved to {args.output_txt}")

if __name__ == "__main__":
    main()

# python lip_sync.py --video_txt ../input.txt --output_txt output.txt
