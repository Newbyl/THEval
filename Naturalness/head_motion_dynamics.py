import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from facexformer.network import FaceXFormer
from PIL import Image
from tqdm import tqdm
import argparse
import os



MAX_FRAMES_PER_SEGMENT = 500
NUM_SEGMENTS = 1
FRAME_THRESHOLD = MAX_FRAMES_PER_SEGMENT * NUM_SEGMENTS
BATCH_SIZE = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

transforms_image = Compose([
    Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_headpose_batch(images, model, device):
    boxes, _ = mtcnn.detect(images)

    cropped_images = []
    valid_indices = []

    for idx, box in enumerate(boxes):
        if box is not None and len(box) > 0:
            x_min, y_min, x_max, y_max = box[0]
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            cropped = images[idx].crop((x_min, y_min, x_max, y_max))
            cropped_images.append(cropped)
            valid_indices.append(idx)
        else:
            valid_indices.append(None)

    if not cropped_images:
        return [None] * len(images)

    image_tensors = torch.stack([transforms_image(img) for img in cropped_images]).to(device)

    task = torch.tensor([2] * len(cropped_images)).to(device)
    label = {"headpose": torch.zeros([len(cropped_images), 3]).to(device)}

    with torch.no_grad():
        _, headpose_output, _, _, _, _, _, _ = model(image_tensors, label, task)

    pitch = (headpose_output[:, 0].cpu().numpy()) * 180 / np.pi
    yaw = (headpose_output[:, 1].cpu().numpy()) * 180 / np.pi
    roll = (headpose_output[:, 2].cpu().numpy()) * 180 / np.pi

    results = [None] * len(images)

    valid_count = 0
    for idx in valid_indices:
        if idx is not None:
            results[idx] = (pitch[valid_count], yaw[valid_count], roll[valid_count])
            valid_count += 1

    return results

def calculate_metrics(pitch, yaw, roll):
    pitch_smoothness = np.var(np.diff(pitch)) if len(pitch) > 1 else 0.0
    yaw_smoothness = np.var(np.diff(yaw)) if len(yaw) > 1 else 0.0
    roll_smoothness = np.var(np.diff(roll)) if len(roll) > 1 else 0.0
    return pitch_smoothness, yaw_smoothness, roll_smoothness


def measure_head_motion_dynamics(pitch, yaw, roll):
    pitch = np.array(pitch)
    yaw   = np.array(yaw)
    roll  = np.array(roll)

    std_pitch = np.std(pitch)
    std_yaw   = np.std(yaw)
    std_roll  = np.std(roll)
    avg_std = (std_pitch + std_yaw + std_roll) / 3.0

    dpitch = np.diff(pitch)
    dyaw   = np.diff(yaw)
    droll  = np.diff(roll)
    var_dpitch = np.var(dpitch) if len(dpitch) > 1 else 0.0
    var_dyaw   = np.var(dyaw)   if len(dyaw)   > 1 else 0.0
    var_droll  = np.var(droll)  if len(droll)  > 1 else 0.0
    avg_deriv_var = (var_dpitch + var_dyaw + var_droll) / 3.0

    complexity = np.sqrt(avg_std * avg_deriv_var)

    return complexity

def analyze_video(video_path, model, device, start_frame=0, end_frame=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    pitch_vals, yaw_vals, roll_vals = [], [], []

    frames_to_process = end_frame - start_frame

    with tqdm(total=frames_to_process, desc=f"Processing {os.path.basename(video_path)} Frames {start_frame}-{end_frame}", leave=False) as pbar:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        batch_images = []

        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            batch_images.append(pil_image)
            current_frame += 1

            if len(batch_images) == BATCH_SIZE or current_frame == end_frame:
                headposes = get_headpose_batch(batch_images, model, device)

                for hp in headposes:
                    if hp is not None:
                        pitch, yaw, roll = hp
                        pitch_vals.append(pitch)
                        yaw_vals.append(yaw)
                        roll_vals.append(roll)
                pbar.update(len(batch_images))

                batch_images = []

    cap.release()

    if not pitch_vals:
        print(f"Warning: No faces detected in video segment {video_path} Frames {start_frame}-{end_frame}. Skipping metrics computation.")
        return None

    pitch_smoothness, yaw_smoothness, roll_smoothness = calculate_metrics(pitch_vals, yaw_vals, roll_vals)
    head_motion_dynamics = measure_head_motion_dynamics(pitch_vals, yaw_vals, roll_vals)

    return {
        "pitch_smoothness": pitch_smoothness,
        "yaw_smoothness": yaw_smoothness,
        "roll_smoothness": roll_smoothness,
        "head_motion_dynamics": head_motion_dynamics,
        "start_frame": start_frame,
        "end_frame": end_frame
    }

def load_model(model_path, device):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = FaceXFormer().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict_backbone'])
    model.eval()
    return model, device

def evaluate_videos(video_paths, model_path, output_path, device="cuda:0"):
    model, device = load_model(model_path, device)

    metrics_list = []

    with open(output_path, 'w') as outfile:
        outfile.write("Video Path,Segment,Start Frame,End Frame,Pitch Smoothness,Yaw Smoothness,Roll Smoothness,Range Smoothness,head motion dynamics\n")

        with tqdm(total=len(video_paths), desc="Evaluating Videos") as pbar_videos:
            for video_idx, video_path in enumerate(video_paths, 1):
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error: Cannot open video {video_path}")
                    pbar_videos.update(1)
                    continue
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if frame_count > FRAME_THRESHOLD:
                    for i in range(NUM_SEGMENTS):
                        start_frame = i * MAX_FRAMES_PER_SEGMENT
                        end_frame = start_frame + MAX_FRAMES_PER_SEGMENT
                        if end_frame > frame_count:
                            end_frame = frame_count

                        metrics = analyze_video(video_path, model, device, start_frame, end_frame)
                        if metrics is not None:
                            metrics_list.append(metrics)
                            segment_label = f"Segment_{i+1}"
                            outfile.write(f"{video_path},{segment_label},{metrics['start_frame']},{metrics['end_frame']},"
                                          f"{metrics['pitch_smoothness']:.6f},{metrics['yaw_smoothness']:.6f},"
                                          f"{metrics['roll_smoothness']:.6f},"
                                          f"{metrics['head_motion_dynamics']:.6f}\n")
                else:
                    metrics = analyze_video(video_path, model, device)
                    if metrics is not None:
                        metrics_list.append(metrics)
                        segment_label = "Full_Video"
                        outfile.write(f"{video_path},{segment_label},0,{frame_count},"
                                      f"{metrics['pitch_smoothness']:.6f},{metrics['yaw_smoothness']:.6f},"
                                      f"{metrics['roll_smoothness']:.6f},"
                                      f"{metrics['head_motion_dynamics']:.6f}\n")

                pbar_videos.update(1)

        if not metrics_list:
            print("No valid metrics to compute. Exiting.")
            return

        head_motion_dynamics_values = [m['head_motion_dynamics'] for m in metrics_list]

        mean_head_motion_dynamics = np.mean(head_motion_dynamics_values) if head_motion_dynamics_values else 0.0

        with open(output_path, 'a') as outfile:
            outfile.write("\n=== Evaluation Summary ===\n")
            outfile.write(f"Number of video segments processed: {len(metrics_list)}\n")
            outfile.write(f"Mean head motion dynamics: {mean_head_motion_dynamics:.6f}\n")

    print("\n=== Evaluation Results ===")
    print(f"Number of video segments processed: {len(metrics_list)}")
    print(f"Mean head motion dynamics: {mean_head_motion_dynamics:.6f}")
    print(f"\nDetailed metrics have been saved to: {output_path}")

def read_video_paths(txt_file):
    if not os.path.isfile(txt_file):
        print(f"Error: The file {txt_file} does not exist.")
        return []
    with open(txt_file, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths

def main():
    parser = argparse.ArgumentParser(description="Evaluate videos for head motion dynamics.")
    parser.add_argument('--video_txt', type=str, required=True, help="Path to the text file containing video paths.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the FaceXFormer model checkpoint.")
    parser.add_argument('--output_txt', type=str, required=True, help="Path to the output text file to save metrics.")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to run the model on. e.g., 'cuda:0' or 'cpu'.")
    args = parser.parse_args()

    video_paths = read_video_paths(args.video_txt)
    if not video_paths:
        print("No video paths found. Exiting.")
        return

    evaluate_videos(video_paths, args.model_path, args.output_txt, args.device)

if __name__ == "__main__":
    main()


# python head_motion_dynamics.py --video_txt ../input.txt --model_path facexformer/ckpts/model.pt --output_txt output.txt --device cuda:0
