import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# Add the project root to sys.path to allow imports from 'behavior' package
# when running the script directly as 'python behavior/predict_behavior.py'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from behavior.models import BehaviorRNN


def _load_model(exp_dir: str, video_to_eval: str, behavior_classes: Dict[str, int]) -> Tuple[BehaviorRNN, torch.device, Dict]:
    """
    Load BehaviorRNN and its config from an experiment directory.
    """
    with open(os.path.join(exp_dir, "config_used.yaml"), "r") as f:
        config = yaml.safe_load(f)

    rnn_type_cfg = config.get("rnn_type", "LSTM")
    hidden_size_cfg = config.get("hidden_size", 128)
    num_layers_cfg = config.get("num_layers", 2)

    summary_path = os.path.join(exp_dir, "summary.txt")
    rnn_type, hidden_size, num_layers = rnn_type_cfg, hidden_size_cfg, num_layers_cfg
    
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
            input_size = data["features"].shape[1]
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
    
    return model, device, config


def get_clip_id(frame_idx: int, offsets: Dict[str, int]) -> str:
    """
    Determine which clip a global frame index belongs to.
    """
    sorted_clips = sorted(offsets.keys(), key=lambda x: offsets[x])
    for i in range(len(sorted_clips) - 1):
        curr = sorted_clips[i]
        nxt = sorted_clips[i+1]
        if offsets[curr] <= frame_idx < offsets[nxt]:
            return curr
    return sorted_clips[-1] if sorted_clips else "unknown"


def predict_and_count(exp_name: str, video_to_eval: str, src_anns: str = "data/annotations/refined") -> None:
    exp_dir = os.path.join("out", "results", exp_name)
    
    # Load config for classes
    with open(os.path.join(exp_dir, "config_used.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    behavior_classes = config.get("behavior_classes", {})
    # Sort classes by ID to get consistent column ordering
    sorted_classes = [name for name, idx in sorted(behavior_classes.items(), key=lambda x: x[1])]
    
    model, device, cfg = _load_model(exp_dir, video_to_eval, behavior_classes)
    window_size = cfg.get("window_size", 30)
    
    # 1. Calculate clip offsets from annotations
    anns_dir = os.path.join(src_anns, video_to_eval)
    if not os.path.exists(anns_dir):
        raise FileNotFoundError(f"Annotations directory not found: {anns_dir}")
    
    clip_offsets = {}
    running_offset = 0
    for jf in sorted(os.listdir(anns_dir)):
        if not jf.endswith(".json"): continue
        with open(os.path.join(anns_dir, jf), "r") as f:
            d = json.load(f)
        clip_id = jf.replace(".json", "")
        clip_offsets[clip_id] = running_offset
        running_offset += len(d.get("images", []))

    # 2. Run inference and count
    use_keypoints = cfg.get("use_keypoints", False)
    feat_root = "data/features_kp" if use_keypoints else "data/features"
    feat_video_dir = os.path.join(feat_root, video_to_eval)
    
    # counts[clip_id][track_id][class_id] = count
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    print(f">>> Predicting behaviors for {video_to_eval}...")
    for npz_file in sorted(os.listdir(feat_video_dir)):
        if not npz_file.endswith(".npz"): continue
        
        data = np.load(os.path.join(feat_video_dir, npz_file))
        feats = data["features"]
        frames = data["frames"]
        track_id = int(npz_file.replace(".npz", "").replace("track_", ""))

        with torch.no_grad():
            for i in range(feats.shape[0] - window_size + 1):
                window = torch.from_numpy(feats[i : i + window_size]).float().unsqueeze(0).to(device)
                output = model(window)
                _, pred = torch.max(output, 1)
                
                frame_idx = int(frames[i + window_size - 1])
                clip_id = get_clip_id(frame_idx, clip_offsets)
                counts[clip_id][track_id][int(pred.item())] += 1

    # 3. Save results per clip
    out_dir = os.path.join("out", "predictions", video_to_eval)
    os.makedirs(out_dir, exist_ok=True)
    
    for clip_id, track_counts in counts.items():
        csv_path = os.path.join(out_dir, f"{clip_id}_counts.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Header: track_id, class1, class2, ...
            writer.writerow(["track_id"] + sorted_classes)
            
            # Sort track IDs for consistency
            for tid in sorted(track_counts.keys()):
                row = [tid]
                for class_idx in range(len(sorted_classes)):
                    row.append(track_counts[tid][class_idx])
                writer.writerow(row)
        print(f"    Saved counts for clip {clip_id} to {csv_path}")

    print(f"\n>>> FINISHED. Prediction counts saved in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment folder name in out/results/")
    parser.add_argument("--video", type=str, required=True, help="Video folder to evaluate")
    parser.add_argument("--src_anns", type=str, default="data/annotations/refined", help="Source annotations directory")
    args = parser.parse_args()

    predict_and_count(args.exp, args.video, args.src_anns)
