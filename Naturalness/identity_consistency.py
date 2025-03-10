import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional
from facenet_pytorch import MTCNN
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from itertools import islice


MAX_FRAMES_PER_SEGMENT = 500
NUM_SEGMENTS = 1
FRAME_THRESHOLD = MAX_FRAMES_PER_SEGMENT * NUM_SEGMENTS

BATCH_SIZE = 128


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

def initialize_models(device: torch.device) -> Tuple[MTCNN, AutoModel, AutoImageProcessor]:
    mtcnn = MTCNN(keep_all=False, device=device, post_process=False)
    
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    print(f"Loaded processor type: {type(processor)}")
    
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device).eval()
    return mtcnn, model, processor



def extract_face_embeddings_batch(frames: List[np.ndarray], mtcnn: MTCNN, processor: AutoImageProcessor, 
                                model: AutoModel, device: torch.device) -> List[Optional[np.ndarray]]:
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    pil_images = [Image.fromarray(img) for img in frames_rgb]
    
    with torch.no_grad():
        boxes, _ = mtcnn.detect(pil_images)
    
    embeddings_list = [None] * len(frames)
    
    for idx, box in enumerate(boxes):
        try:
            if box is None or len(box) == 0:
                continue
                
            box = box[0]
            x1, y1, x2, y2 = map(int, box)
            face = frames_rgb[idx][y1:y2, x1:x2]
            
            if face.size == 0:
                continue
                
            if len(face.shape) == 2:
                face = np.stack([face]*3, axis=-1)
            elif face.shape[2] == 1:
                face = np.repeat(face, 3, axis=-1)
            elif face.shape[2] == 4:
                face = cv2.cvtColor(face, cv2.COLOR_RGBA2RGB)
            
            if face.shape[2] != 3:
                continue
                
            with torch.no_grad():
                inputs = processor(images=Image.fromarray(face), return_tensors="pt").to(device)
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            
            embeddings_list[idx] = embedding
            
        except Exception as e:
            print(f"Error processing frame {idx}: {str(e)}")
            embeddings_list[idx] = None
            continue
            
    return embeddings_list


def compute_cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    if embedding1 is None or embedding2 is None:
        return np.nan
    norm1 = embedding1 / np.linalg.norm(embedding1)
    norm2 = embedding2 / np.linalg.norm(embedding2)

    cosine_similarity = np.dot(norm1, norm2)

    cosine_distance = 1 - cosine_similarity
    return cosine_distance

def batched(iterable, n=1):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

def process_video_segment(video_path: str, start_frame: int, end_frame: int, mtcnn: MTCNN, model: AutoModel, processor: AutoImageProcessor, device: torch.device, batch_size: int = BATCH_SIZE) -> float:
    if not os.path.exists(video_path):
        print(f"Warning: Video file '{video_path}' does not exist. Skipping segment.")
        return np.nan
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Unable to open video file '{video_path}'. Skipping segment.")
        return np.nan

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_to_process = end_frame - start_frame
    embedding_distances = []
    prev_embedding = None
    frame_idx = start_frame

    with tqdm(total=frames_to_process, desc=f"Processing {os.path.basename(video_path)} Frames {start_frame}-{end_frame}", leave=False) as pbar:
        while frame_idx < end_frame:
            batch_frames = []
            batch_end = min(frame_idx + batch_size, end_frame)
            num_frames_in_batch = batch_end - frame_idx

            for _ in range(num_frames_in_batch):
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)
                frame_idx += 1

            if not batch_frames:
                break

            embeddings = extract_face_embeddings_batch(batch_frames, mtcnn, processor, model, device)

            for embedding in embeddings:
                if embedding is not None:
                    if prev_embedding is not None:
                        distance = compute_cosine_distance(prev_embedding, embedding)
                        embedding_distances.append(distance)
                    prev_embedding = embedding
                else:
                    prev_embedding = None
                pbar.update(1)
    
    cap.release()

    if not embedding_distances:
        print(f"Warning: No valid embeddings found in segment {start_frame}-{end_frame} of '{video_path}'.")
        return np.nan

    average_distance = np.nanmean(embedding_distances)
    return average_distance

def compute_identity_preservation_scores(video_paths: List[str], mtcnn: MTCNN, model: AutoModel, processor: AutoImageProcessor, device: torch.device, batch_size: int = BATCH_SIZE) -> Tuple[float, dict]:
    all_distances = []
    video_scores_dict = {}
    
    total_videos = len(video_paths)
    print(f"Processing {total_videos} videos...\n")
    
    for video_path in tqdm(video_paths, desc="Processing Videos", unit="video"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open video '{video_path}'. Skipping.")
            video_scores_dict[video_path] = np.nan
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        actual_num_segments = NUM_SEGMENTS
        if total_frames > FRAME_THRESHOLD:
            actual_num_segments = NUM_SEGMENTS
        else:
            actual_num_segments = max(1, total_frames // MAX_FRAMES_PER_SEGMENT)
        
        if total_frames <= MAX_FRAMES_PER_SEGMENT:
            segments = [(0, total_frames)]
        else:
            segments = []
            for i in range(actual_num_segments):
                start_frame = i * MAX_FRAMES_PER_SEGMENT
                end_frame = start_frame + MAX_FRAMES_PER_SEGMENT
                if end_frame > total_frames:
                    end_frame = total_frames
                segments.append((start_frame, end_frame))
        
        video_segment_distances = []
        
        for idx, (start, end) in enumerate(segments, 1):
            distance = process_video_segment(
                video_path,
                start_frame=start,
                end_frame=end,
                mtcnn=mtcnn,
                model=model,
                processor=processor,
                device=device,
                batch_size=batch_size
            )
            video_segment_distances.append(distance)
            all_distances.append(distance)
            print(f"Video: {video_path}, Segment: {idx}, Frames: {start}-{end}, Average Distance: {distance:.4f}")
        
        valid_distances = [d for d in video_segment_distances if not np.isnan(d)]
        if valid_distances:
            mean_video_distance = np.mean(valid_distances)
            video_scores_dict[video_path] = mean_video_distance
        else:
            video_scores_dict[video_path] = np.nan
            print(f"Warning: No valid distances computed for video '{video_path}'.")
    
    if not all_distances:
        print("Error: No identity preservation scores computed. Please check your videos and paths.")
        sys.exit(1)
    
    overall_mean_distance = np.nanmean(all_distances)
    return overall_mean_distance, video_scores_dict

def evaluate_videos(video_paths: List[str], output_path: str, mtcnn: MTCNN, model: AutoModel, processor: AutoImageProcessor, device: torch.device, max_frames_per_segment: int = MAX_FRAMES_PER_SEGMENT, num_segments: int = NUM_SEGMENTS, batch_size: int = BATCH_SIZE):
    metrics_list = []
    video_scores_dict = {}

    with open(output_path, 'w') as outfile:
        outfile.write("Video Path,Segment,Start Frame,End Frame,Mean Cosine Distance\n")
    
        with tqdm(total=len(video_paths), desc="Evaluating Videos") as pbar_videos:
            for video_path in video_paths:
                if not os.path.exists(video_path):
                    print(f"Error: Video file '{video_path}' does not exist. Skipping.")
                    outfile.write(f"{video_path},N/A,N/A,N/A,NaN\n")
                    pbar_videos.update(1)
                    continue

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error: Cannot open video '{video_path}'. Skipping.")
                    outfile.write(f"{video_path},N/A,N/A,N/A,NaN\n")
                    pbar_videos.update(1)
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
                
                video_segment_distances = []
                
                for idx, (start, end) in enumerate(segments, 1):
                    distance = process_video_segment(
                        video_path,
                        start_frame=start,
                        end_frame=end,
                        mtcnn=mtcnn,
                        model=model,
                        processor=processor,
                        device=device,
                        batch_size=batch_size
                    )
                    video_segment_distances.append(distance)
                    metrics_list.append(distance)
                    
                    segment_label = f"Segment_{idx}"
                    outfile.write(f"{video_path},{segment_label},{start},{end},{distance:.6f}\n")
                    print(f"Video: {video_path}, Segment: {idx}, Frames: {start}-{end}, Average Distance: {distance:.4f}")
                
                pbar_videos.update(1)

    if not metrics_list:
        print("No valid metrics to compute. Exiting.")
        return

    overall_mean_distance = np.nanmean(metrics_list)

    with open(output_path, 'a') as outfile:
        outfile.write("\n=== Evaluation Summary ===\n")
        outfile.write(f"Number of video segments processed: {len(metrics_list)}\n")
        outfile.write(f"Average Mean Cosine Distance: {overall_mean_distance:.6f}\n")

    print("\n=== Evaluation Results ===")
    print(f"Number of video segments processed: {len(metrics_list)}")
    print(f"Average Mean Cosine Distance: {overall_mean_distance:.6f}")
    print(f"\nDetailed metrics have been saved to: {output_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compute Identity Preservation Scores Across Multiple Videos.')
    parser.add_argument('--video_txt', type=str, required=True, help="Path to the text file containing video paths.")
    parser.add_argument('--output_txt', type=str, required=True, help="Path to the output text file to save metrics.")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    video_paths = read_video_paths(args.video_txt)
    
    mtcnn, model, processor = initialize_models(device)
    
    evaluate_videos(
        video_paths=video_paths,
        output_path=args.output_txt,
        mtcnn=mtcnn,
        model=model,
        processor=processor,
        device=device,
        max_frames_per_segment=MAX_FRAMES_PER_SEGMENT,
        num_segments=NUM_SEGMENTS,
        batch_size=BATCH_SIZE
    )

if __name__ == "__main__":
    main()

# python identity_consistency.py --video_txt ../input.txt --output_txt output.txt