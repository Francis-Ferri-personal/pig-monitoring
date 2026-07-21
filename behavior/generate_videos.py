import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import yaml

from behavior.models import BehaviorRNN
from app.backend.services.video_style import draw_pose_annotations, draw_prediction_label, convert_to_web_mp4


def _load_model(exp_dir: str, video_to_eval: str, behavior_classes: Dict[str, int]) -> BehaviorRNN:
    """
    Load BehaviorRNN with hyperparameters that match the training run,
    using summary.txt to capture CLI overrides (e.g., BiLSTM).
    """
    with open(os.path.join(exp_dir, "config_used.yaml"), "r") as f:
        config = yaml.safe_load(f)

    rnn_type_cfg = config.get("rnn_type", "LSTM")
    hidden_size_cfg = config.get("hidden_size", 128)
    num_layers_cfg = config.get("num_layers", 2)

    summary_path = os.path.join(exp_dir, "summary.txt")
    rnn_type = rnn_type_cfg
    hidden_size = hidden_size_cfg
    num_layers = num_layers_cfg
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            for line in f:
                if line.startswith("RNN:"):
                    rnn_type = line.split(":", 1)[1].strip()
                elif line.startswith("Hidden:"):
                    hidden_size = int(line.split(":", 1)[1].strip())
                elif line.startswith("Layers:"):
                    num_layers = int(line.split(":", 1)[1].strip())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_keypoints = config.get("use_keypoints", False)
    feat_root = "data/features_kp" if use_keypoints else "data/features"
    feat_video_dir = os.path.join(feat_root, video_to_eval)
    if not os.path.exists(feat_video_dir):
        raise FileNotFoundError(f"Feature directory not found: {feat_video_dir}")

    input_size = None
    for npz_file in sorted(os.listdir(feat_video_dir)):
        if npz_file.endswith(".npz"):
            data = np.load(os.path.join(feat_video_dir, npz_file))
            feats = data["features"]
            input_size = feats.shape[1]
            break
    if input_size is None:
        raise RuntimeError(f"No .npz visual feature files found in {feat_video_dir}")

    model = BehaviorRNN(
        rnn_type=rnn_type,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=len(behavior_classes),
    ).to(device)

    state = torch.load(os.path.join(exp_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def generate_prediction_videos(exp_name: str, video_to_eval: str = "video3", draw_kp: bool = False) -> None:
    """
    Generate overlay videos for the model using embeddings and behavior annotations.
    """
    exp_dir = os.path.join("out", "results", exp_name)
    with open(os.path.join(exp_dir, "config_used.yaml"), "r") as f:
        config = yaml.safe_load(f)

    behavior_classes = config.get("behavior_classes", {})
    # Build class name list indexed by class id to avoid relying on dict order
    if behavior_classes:
        max_id = max(behavior_classes.values())
        class_names = [None] * (max_id + 1)
        for name, idx in behavior_classes.items():
            if 0 <= idx < len(class_names):
                class_names[idx] = name
    else:
        class_names = []

    skeleton = []
    if draw_kp:
        pose_yaml = "data/pig_pose.yaml"
        if os.path.exists(pose_yaml):
            with open(pose_yaml, "r") as f:
                pose_data = yaml.safe_load(f)
                skeleton = pose_data.get("categories", [{}])[0].get("skeleton", [])
        else:
            print(f"!!! {pose_yaml} not found. Skipping skeleton.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(exp_dir, video_to_eval, behavior_classes)

    # 1. Run Inference on features
    use_keypoints = config.get("use_keypoints", False)
    feat_root = "data/features_kp" if use_keypoints else "data/features"
    feat_video_dir = os.path.join(feat_root, video_to_eval)
    window_size = config.get("window_size", 30)

    pigs_predictions: Dict[int, Dict[int, int]] = {}

    print(f">>> Running inference for {video_to_eval}...")
    for npz_file in sorted(os.listdir(feat_video_dir)):
        if not npz_file.endswith(".npz"):
            continue

        data = np.load(os.path.join(feat_video_dir, npz_file))
        feats = data["features"]
        frames = data["frames"]
        track_id = int(npz_file.replace(".npz", "").replace("track_", ""))

        num_frames = feats.shape[0]
        track_pred_map: Dict[int, int] = {}

        with torch.no_grad():
            for i in range(num_frames - window_size + 1):
                window = torch.from_numpy(feats[i : i + window_size]).float().unsqueeze(0).to(device)
                output = model(window)
                _, pred = torch.max(output, 1)

                frame_idx = int(frames[i + window_size - 1])
                track_pred_map[frame_idx] = int(pred.item())

        pigs_predictions[track_id] = track_pred_map

    # 2. Prepare collection for logs
    inference_logs = []

    # 2. Generate videos overlaying predictions
    out_vid_dir = os.path.join(exp_dir, "videos", video_to_eval)
    os.makedirs(out_vid_dir, exist_ok=True)

    # Prefer the "behavior" annotations, but fall back to "refined" or "sam" if needed
    anns_dir = os.path.join("data", "annotations", "behavior", video_to_eval)
    if not os.path.exists(anns_dir):
        alt = os.path.join("data", "annotations", "refined", video_to_eval)
        sam_alt = os.path.join("data", "annotations", "sam", video_to_eval)
        
        if os.path.exists(alt):
            print(f"Using refined annotations at: {alt}")
            anns_dir = alt
        elif os.path.exists(sam_alt):
            print(f"Using raw SAM annotations at: {sam_alt}")
            anns_dir = sam_alt
        else:
            raise FileNotFoundError(f"Annotations directory not found for {video_to_eval}: tried behavior, refined, and sam folders.")
    frames_dir = os.path.join("data", "images", "frames", video_to_eval)

    # Build global frame offset per clip (must match order used in feature_extractor)
    clip_offsets: Dict[str, int] = {}
    running_offset = 0
    for jf in sorted(os.listdir(anns_dir)):
        if not jf.endswith(".json"):
            continue
        with open(os.path.join(anns_dir, jf), "r") as f:
            d = json.load(f)
        n_frames = len(d.get("images", []))
        clip_offsets[jf.replace(".json", "")] = running_offset
        running_offset += n_frames

    print(f">>> Generating clips in {out_vid_dir}...")

    for json_file in sorted(os.listdir(anns_dir)):
        if not json_file.endswith(".json"):
            continue

        clip_id = json_file.replace(".json", "")
        print(f"  - Creating {clip_id}.mp4...")

        with open(os.path.join(anns_dir, json_file), "r") as f:
            clip_data = json.load(f)

        anns_by_img = {}
        for ann in clip_data["annotations"]:
            img_id = ann["image_id"]
            anns_by_img.setdefault(img_id, []).append(ann)

        clip_frames_dir = os.path.join(frames_dir, clip_id)
        if not os.path.exists(clip_frames_dir):
            print(f"!!! Frames missing for {clip_frames_dir}, skipping.")
            continue

        # Do not assume PNG or a zero-based filename: use the first frame listed
        # in the COCO file that actually exists on disk.
        ordered_images = sorted(
            clip_data["images"],
            key=lambda image: int(image.get("frame_id", image.get("id", 0))),
        )
        first_frame_path = next(
            (
                os.path.join(clip_frames_dir, os.path.basename(image["file_name"]))
                for image in ordered_images
                if os.path.exists(os.path.join(clip_frames_dir, os.path.basename(image["file_name"])))
            ),
            None,
        )
        if first_frame_path is None:
            print(f"!!! No COCO frame files found in {clip_frames_dir}, skipping.")
            continue
        sample_img = cv2.imread(first_frame_path)
        if sample_img is None:
            print(f"!!! Could not read {first_frame_path}, skipping.")
            continue
        h, w, _ = sample_img.shape

        out_mp4 = os.path.join(out_vid_dir, f"{clip_id}.mp4")
        raw_mp4 = os.path.join(out_vid_dir, f"{clip_id}_raw.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = config.get("frames_per_second", 1)

        writer = cv2.VideoWriter(raw_mp4, fourcc, out_fps, (w, h))

        # Images may use "frame_id" or just "id" depending on source; use fallback
        def _get_frame_id(i):
            return int(i.get("frame_id", i.get("id")))

        for img_info in sorted(clip_data["images"], key=_get_frame_id):
            img_id = img_info["id"]
            # Global frame index (same as in NPZ) so predictions align with the right clip/frame
            frame_abs = clip_offsets.get(clip_id, 0) + _get_frame_id(img_info)

            fname = os.path.basename(img_info["file_name"])
            img_path = os.path.join(clip_frames_dir, fname)

            frame_img = cv2.imread(img_path)
            if frame_img is None:
                continue

            # Sort annotations by track_id for orderly logging and consistent drawing
            curr_anns = sorted(anns_by_img.get(img_id, []), key=lambda a: a.get("track_id", 0))

            for ann in curr_anns:
                track_id = ann.get("track_id")
                if track_id is None:
                    continue

                pred_label = "Waiting"
                if track_id in pigs_predictions and frame_abs in pigs_predictions[track_id]:
                    pred_idx = pigs_predictions[track_id][frame_abs]
                    if 0 <= pred_idx < len(class_names) and class_names[pred_idx] is not None:
                        pred_label = class_names[pred_idx]
                    else:
                        pred_label = f"Cls{pred_idx}"

                # Match the web app: green box/ID, red keypoints, cyan skeleton and blue prediction.
                draw_pose_annotations(frame_img, [ann], draw_keypoints=False)
                draw_prediction_label(frame_img, ann["bbox"], pred_label)

                gt_label = ann.get("action") or ann.get("behavior") or "N/A"

                # Collect log data
                inference_logs.append({
                    "video": video_to_eval,
                    "clip": clip_id,
                    "frame": _get_frame_id(img_info),
                    "track_id": track_id,
                    "predicted_action": pred_label,
                    "gt_action": gt_label,
                    "bbox": ann["bbox"],
                    "keypoints": ann.get("keypoints", [])
                })
            writer.write(frame_img)

        writer.release()
        try:
            convert_to_web_mp4(Path(raw_mp4), Path(out_mp4))
        finally:
            if os.path.exists(raw_mp4):
                os.remove(raw_mp4)
        print(f"    Saved {out_mp4}")

    # 3. Save logs
    log_dir = os.path.join(out_vid_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    csv_log = os.path.join(log_dir, "inference_log.csv")
    json_log = os.path.join(log_dir, "inference_log.json")

    print(f">>> Saving inference logs to {log_dir}...")
    
    # Save CSV
    if inference_logs:
        keys = inference_logs[0].keys()
        with open(csv_log, "w", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(inference_logs)

    # Save JSON
    with open(json_log, "w") as f:
        json.dump(inference_logs, f, indent=4)

    print(f"\n>>> FINISHED generating VISUAL video reports and logs for {video_to_eval} in: {out_vid_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment folder name in out/results/")
    parser.add_argument("--video", type=str, default="video3", help="Video folder to evaluate (default: video3)")
    parser.add_argument("--inference", action="store_true", help="Run in inference mode (look for RAW/SAM annotations)")
    parser.add_argument("--draw_kp", action="store_true", help="Deprecated: keypoints are always drawn to match the web app")
    args = parser.parse_args()

    generate_prediction_videos(args.exp, video_to_eval=args.video, draw_kp=args.draw_kp)

