import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
import mediapipe as mp
import csv
from pydub import AudioSegment
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import statistics
import tempfile

MIN_SILENCE_DURATION = 300

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

vad_model = load_silero_vad()

def extract_audio(video_path):
    try:
        return AudioSegment.from_file(video_path)
    except Exception as e:
        print(f"Audio extraction failed for {video_path}: {e}")
        return None

def detect_silence_frames(audio_input, frame_rate, total_frames):
    if audio_input is None:
        return set()
    if isinstance(audio_input, str):
        wav_path = audio_input
        remove_temp = False
    else:
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()
        audio_input.export(temp_wav.name, format="wav")
        wav_path = temp_wav.name
        remove_temp = True

    wav = read_audio(wav_path)
    speech_timestamps = get_speech_timestamps(wav, vad_model, return_seconds=True)

    if remove_temp and os.path.exists(wav_path):
        os.remove(wav_path)

    silence_frames = set(range(total_frames))
    
    for seg in speech_timestamps:
        start_frame = int(seg['start'] * frame_rate)
        end_frame = int(seg['end']   * frame_rate)
        for f in range(start_frame, end_frame+1):
            silence_frames.discard(f)

    filtered_silence_frames = set()
    current_start = None
    count = 0
    for f in sorted(silence_frames):
        if current_start is None:
            current_start, count = f, 1
        elif f == current_start + count:
            count += 1
        else:
            if count >= (MIN_SILENCE_DURATION/1000 * frame_rate):
                filtered_silence_frames.update(range(current_start, current_start+count))
            current_start, count = f, 1
    if count >= (MIN_SILENCE_DURATION/1000 * frame_rate):
        filtered_silence_frames.update(range(current_start, current_start+count))
    
    return filtered_silence_frames

def calculate_lip_average_distance(frame):
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = 256, 256
    
    # Normalize by interocular distance
    left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    eye_distance = np.linalg.norm(left_eye - right_eye)
    if eye_distance < 1e-3:
        return None

    upper_lip_indices = [191, 80, 81, 82, 13, 312, 311, 310]
    lower_lip_indices = [95, 88, 178, 87, 14, 317, 402, 318]

    vertical_distances = []
    for upper, lower in zip(upper_lip_indices, lower_lip_indices):
        y_upper = landmarks[upper].y * h
        y_lower = landmarks[lower].y * h
        vertical_distances.append(abs(y_upper - y_lower) / eye_distance)

    return np.mean(vertical_distances) if vertical_distances else None

def process_video(video_path, silence_frames):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return float('nan')

        silent_frame_list = sorted(silence_frames)
        avg_distances = []

        for f_idx in silent_frame_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (256, 256))

            avg_distance = calculate_lip_average_distance(frame)
            
            if avg_distance is not None:
                avg_distances.append(avg_distance)

        cap.release()

        if len(avg_distances) < 2:
            return float('nan')

        mean_opening = float(np.mean(avg_distances))
        mad = float(statistics.median([abs(x - mean_opening) for x in avg_distances]))
        return mad

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return float('nan')

def main():
    parser = argparse.ArgumentParser(description='Silent Mouth Movement Analysis (Robust)')
    parser.add_argument('--video_txt', required=True, help='Input video list')
    parser.add_argument('--output_txt', required=True, help='Output CSV file')
    parser.add_argument('--audio_folder', help='Folder with pre-extracted .wav files', default=None)
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_txt) or ".", exist_ok=True)

    with open(args.video_txt, 'r') as f, open(args.output_txt, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['Video Path', 'Silent MAD'])
        
        mads = []

        for video_path in tqdm([l.strip() for l in f if l.strip()], desc="Processing videos"):
            try:
                if not os.path.exists(video_path):
                    writer.writerow([video_path, 'NaN'])
                    continue

                if args.audio_folder:
                    base = os.path.splitext(os.path.basename(video_path))[0]
                    wav_file = os.path.join(args.audio_folder, base + '.wav')
                    if not os.path.exists(wav_file):
                        writer.writerow([video_path, 'NaN'])
                        continue
                    audio = wav_file
                else:
                    audio = extract_audio(video_path)

                if audio is None:
                    writer.writerow([video_path, 'NaN'])
                    continue

                cap = cv2.VideoCapture(video_path)
                frame_rate = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                silence_frames = detect_silence_frames(
                    audio_input=audio,
                    frame_rate=frame_rate,
                    total_frames=total_frames
                )

                if not silence_frames:
                    writer.writerow([video_path, 'NaN'])
                    continue

                mad = process_video(video_path, silence_frames)

                writer.writerow([
                    video_path,
                    f"{mad:.4f}" if not np.isnan(mad) else "NaN",
                ])

                if not np.isnan(mad):
                    mads.append(mad)

            except Exception as e:
                print(f"Skipping {video_path} due to error: {e}")
                writer.writerow([video_path, 'NaN'])
                continue

        avg_mad = np.mean(mads) if mads else np.nan
        
        writer.writerow([])
        writer.writerow(['Average Silent MAD:', f"{avg_mad:.4f}"])
        
        print(f"\nAnalysis Complete")
        print(f"Average silent MAD: {avg_mad:.4f}")

if __name__ == "__main__":
    main()
