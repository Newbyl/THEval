import cv2
import mediapipe as mp
import numpy as np
import librosa
import argparse
import os
import tempfile
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from decord import VideoReader, cpu
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

FACEMESH_LIPS_UPPER = [191, 80, 81, 82, 13, 312, 311, 310]
FACEMESH_LIPS_LOWER = [95, 88, 178, 87, 14, 317, 402, 318]
FACEMESH_EYES_LEFT = [33, 133]
FACEMESH_EYES_RIGHT = [362, 263]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

vad_model = load_silero_vad()

def extract_audio(filename):
    clip = VideoFileClip(filename)
    audio = clip.audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
        y, sr = librosa.load(temp_audio.name, sr=None)
        temp_path = temp_audio.name
    return y, sr, temp_path

def detect_speech_segments(wav_path, sr, total_frames, fps):
    wav = read_audio(wav_path)
    speech_timestamps = get_speech_timestamps(wav, vad_model, return_seconds=True)

    speech_frames = set()
    for seg in speech_timestamps:
        start_frame = int(seg['start'] * fps)
        end_frame = int(seg['end'] * fps)
        speech_frames.update(range(start_frame, min(end_frame + 1, total_frames)))

    speech_ratio = len(speech_frames) / total_frames if total_frames > 0 else 0
    return speech_frames, speech_ratio

def compute_volume(y, sr, frame_times):
    frame_length = int(sr * 0.05)
    hop_length = int(sr * 0.05)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
    return np.interp(frame_times, times, rms)

def compute_mouth_openness(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()

    openness = []
    frame_times = []

    frames = vr.get_batch(range(total_frames)).asnumpy()
    for i, frame in enumerate(frames):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            upper = np.mean([landmarks[j].y for j in FACEMESH_LIPS_UPPER])
            lower = np.mean([landmarks[j].y for j in FACEMESH_LIPS_LOWER])

            eye_left = np.mean([landmarks[j].x for j in FACEMESH_EYES_LEFT])
            eye_right = np.mean([landmarks[j].x for j in FACEMESH_EYES_RIGHT])
            interocular = abs(eye_right - eye_left)

            if interocular > 0:
                openness_val = abs(upper - lower) / interocular
                openness.append(openness_val)
            else:
                openness.append(np.nan)
        else:
            openness.append(np.nan)

        frame_times.append(i / fps)

    return np.array(frame_times), np.array(openness), total_frames, fps

def process_video(video_path, y, sr, wav_path):
    t, o, total_frames, fps = compute_mouth_openness(video_path)
    v = compute_volume(y, sr, t)

    speech_frames, speech_ratio = detect_speech_segments(wav_path, sr, total_frames, fps)

    mask = np.array([i in speech_frames for i in range(total_frames)]) & ~np.isnan(o)

    if speech_ratio < 0.1 or np.sum(mask) < 2:
        mean_diff = np.nan
    else:
        o_speech = o[mask]
        v_speech = v[mask]

        # Speech-only normalization
        o_norm = (o_speech - np.min(o_speech)) / (np.max(o_speech) - np.min(o_speech) + 1e-8)
        v_norm = (v_speech - np.min(v_speech)) / (np.max(v_speech) - np.min(v_speech) + 1e-8)

        mean_diff = np.mean(np.abs(o_norm - v_norm))

    return mean_diff

def read_video_paths(txt_file):
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Video list file not found: {txt_file}")
    with open(txt_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def evaluate_videos(video_paths, output_path, audio_folder=None):
    mean_diffs = []
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("Video Path,Mean_Difference\n")

        for video_path in tqdm(video_paths, desc="Processing Videos"):
            try:
                if audio_folder:
                    base = os.path.splitext(os.path.basename(video_path))[0]
                    wav_path = os.path.join(audio_folder, base + '.wav')
                    if os.path.exists(wav_path):
                        y, sr = librosa.load(wav_path, sr=None)
                    else:
                        print(f"WAV file not found ({wav_path}), extracting from video")
                        y, sr, wav_path = extract_audio(video_path)
                else:
                    y, sr, wav_path = extract_audio(video_path)

                mean_diff = process_video(video_path, y, sr, wav_path)
                mean_diffs.append(mean_diff)

                f.write(f"{video_path},{mean_diff if not np.isnan(mean_diff) else 'NaN'}\n")
                f.flush()

                if not audio_folder and os.path.exists(wav_path):
                    os.remove(wav_path)

            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                f.write(f"{video_path},NaN\n")
                continue

    with open(output_path, 'a') as f:
        f.write("\n=== Summary ===\n")
        f.write(f"Average Mean Difference: {np.nanmean(mean_diffs):.4f}\n")

def main():
    parser = argparse.ArgumentParser(description='Compute normalized mouth RMS difference with speech-only frames')
    parser.add_argument('--video_txt', required=True, help='Text file containing video paths')
    parser.add_argument('--output_txt', required=True, help='Output text file path')
    parser.add_argument('--audio_folder', help='Folder with pre-extracted .wav files', default=None)
    args = parser.parse_args()

    try:
        video_paths = read_video_paths(args.video_txt)
    except Exception as e:
        print(f"Error: {e}")
        return

    evaluate_videos(video_paths, args.output_txt, args.audio_folder)
    print(f"\nResults saved to {args.output_txt}")

if __name__ == "__main__":
    main()
