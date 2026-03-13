import argparse
import json
import os
from typing import Dict

import cv2
import numpy as np
import torch
import yaml

from behavior.models import BehaviorRNN


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

    feat_video_dir = os.path.join("data", "features", video_to_eval)
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


def generate_prediction_videos(exp_name: str, video_to_eval: str = "video3") -> None:
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(exp_dir, video_to_eval, behavior_classes)

    # 1. Run Inference on features
    feat_video_dir = os.path.join("data", "features", video_to_eval)
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

    # 2. Generate videos overlaying predictions
    out_vid_dir = os.path.join(exp_dir, "videos", video_to_eval)
    os.makedirs(out_vid_dir, exist_ok=True)

    # Prefer the "behavior" annotations, but fall back to "refined" if needed
    anns_dir = os.path.join("data", "annotations", "behavior", video_to_eval)
    if not os.path.exists(anns_dir):
        alt = os.path.join("data", "annotations", "refined", video_to_eval)
        if os.path.exists(alt):
            print(f"Using refined annotations at: {alt}")
            anns_dir = alt
        else:
            raise FileNotFoundError(f"Annotations directory not found for {video_to_eval}: tried {anns_dir} and {alt}")
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

    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
    ]

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

        sample_img = cv2.imread(os.path.join(clip_frames_dir, "00000.png"))
        if sample_img is None:
            continue
        h, w, _ = sample_img.shape

        out_mp4 = os.path.join(out_vid_dir, f"{clip_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = config.get("frames_per_second", 1)

        writer = cv2.VideoWriter(out_mp4, fourcc, out_fps, (w, h))

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

            for ann in anns_by_img.get(img_id, []):
                track_id = ann.get("track_id")
                if track_id is None:
                    continue

                x, y, w_box, h_box = map(int, ann["bbox"])
                color = colors[track_id % len(colors)]
                cv2.rectangle(frame_img, (x, y), (x + w_box, y + h_box), color, 2)

                pred_label = "Waiting"
                if track_id in pigs_predictions and frame_abs in pigs_predictions[track_id]:
                    pred_idx = pigs_predictions[track_id][frame_abs]
                    if 0 <= pred_idx < len(class_names) and class_names[pred_idx] is not None:
                        pred_label = class_names[pred_idx]
                    else:
                        pred_label = f"Cls{pred_idx}"

                gt_label = ann.get("action", "Unknown")

                text_pred = f"Pred: {pred_label}"
                text_gt = f"GT: {gt_label}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale_pred = 0.8
                font_scale_gt = 0.6
                thick_pred = 2
                thick_gt = 1

                (tw_pred, th_pred), _ = cv2.getTextSize(text_pred, font, font_scale_pred, thick_pred)
                (tw_gt, th_gt), _ = cv2.getTextSize(text_gt, font, font_scale_gt, thick_gt)

                bg_width = max(tw_pred, tw_gt) + 10
                bg_height = th_pred + th_gt + 15

                cv2.rectangle(
                    frame_img,
                    (x, max(0, y - bg_height)),
                    (x + bg_width, y),
                    color,
                    -1,
                )

                cv2.putText(
                    frame_img,
                    text_pred,
                    (x + 5, max(th_pred + 5, y - th_gt - 10)),
                    font,
                    font_scale_pred,
                    (255, 255, 255),
                    thick_pred,
                )
                cv2.putText(
                    frame_img,
                    text_gt,
                    (x + 5, max(bg_height - 5, y - 5)),
                    font,
                    font_scale_gt,
                    (200, 200, 200),
                    thick_gt,
                )

            writer.write(frame_img)

        writer.release()
        print(f"    Saved {out_mp4}")

    print(f"\n>>> FINISHED generating VISUAL video reports for {video_to_eval} in: {out_vid_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment folder name in out/results/")
    parser.add_argument("--video", type=str, default="video3", help="Video folder to evaluate (default: video3)")
    args = parser.parse_args()

    generate_prediction_videos(args.exp, video_to_eval=args.video)

