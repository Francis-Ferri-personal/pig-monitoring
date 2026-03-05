import torch
import numpy as np
import os
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from behavior.models import BehaviorRNN

def generate_full_report(exp_name, video_to_eval='video3'):
    # 1. Load config and model
    exp_dir = os.path.join("out", "results", exp_name)
    with open(os.path.join(exp_dir, "config_used.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = list(config['behavior_classes'].keys())
    
    model = BehaviorRNN(
        rnn_type=config.get('rnn_type', 'LSTM'),
        input_size=57, 
        hidden_size=config.get('hidden_size', 64),
        num_layers=config.get('num_layers', 2),
        num_classes=len(class_names)
    ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pt")))
    model.eval()

    # 2. Process each pig and store predictions
    feat_video_dir = os.path.join("data/features", video_to_eval)
    if not os.path.exists(feat_video_dir):
        print(f"!!! Error: Feature directory {feat_video_dir} not found.")
        return

    window_size = config.get('window_size', 30)
    frames_per_clip = 180 # Standard clip length based on config (1 fps, 3 mins)

    pigs_predictions = {}
    pigs_gt = {}
    max_video_frames = 0

    print(f">>> Running inference for all pigs in {video_to_eval}...")

    for npz_file in sorted(os.listdir(feat_video_dir)):
        if not npz_file.endswith('.npz'): continue
        
        data = np.load(os.path.join(feat_video_dir, npz_file))
        feats = data['features']
        labels = data['labels']
        track_id = int(npz_file.replace('.npz', '').replace('track_', ''))
        
        num_frames = feats.shape[0]
        max_video_frames = max(max_video_frames, num_frames)
        
        preds = np.full(num_frames, -1, dtype=float) # -1 for no data
        gt = np.full(num_frames, -1, dtype=float)

        with torch.no_grad():
            for i in range(num_frames - window_size + 1):
                window = torch.from_numpy(feats[i : i + window_size]).float().unsqueeze(0).to(device)
                output = model(window)
                _, pred = torch.max(output, 1)
                preds[i + window_size - 1] = pred.item()
                gt[i + window_size - 1] = labels[i + window_size - 1]
        
        pigs_predictions[track_id] = preds
        pigs_gt[track_id] = gt

    # 3. Generate one image per Clip (180 frames)
    report_dir = os.path.join(exp_dir, "clip_reports", video_to_eval)
    os.makedirs(report_dir, exist_ok=True)
    
    num_clips = int(np.ceil(max_video_frames / frames_per_clip))
    colors = matplotlib.colormaps['tab10']

    print(f">>> Creating {num_clips} clip reports...")

    for c in range(num_clips):
        start = c * frames_per_clip
        end = (c + 1) * frames_per_clip
        
        # --- PREDICTIONS PLOT ---
        plt.figure(figsize=(15, 8))
        for tid in sorted(pigs_predictions.keys()):
            pred_data = pigs_predictions[tid]
            if start < len(pred_data):
                chunk = pred_data[start:min(end, len(pred_data))]
                valid_idx = np.where(chunk != -1)[0]
                if len(valid_idx) > 0:
                    plt.step(start + valid_idx, chunk[valid_idx], where='post', 
                             label=f'Pig {tid}', color=colors(tid % 10), linewidth=2)

        plt.title(f'Behavior Predictions - {video_to_eval} - Clip {c+1:02d}')
        plt.xlabel('Frame')
        plt.ylabel('Action')
        plt.yticks(range(len(class_names)), class_names)
        plt.legend(loc='upper right', ncol=len(pigs_predictions))
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.ylim(-0.5, len(class_names) - 0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, f"{c+1:02d}_pred.png"))
        plt.close()

        # --- GROUND TRUTH PLOT ---
        plt.figure(figsize=(15, 8))
        for tid in sorted(pigs_gt.keys()):
            gt_data = pigs_gt[tid]
            if start < len(gt_data):
                chunk = gt_data[start:min(end, len(gt_data))]
                valid_idx = np.where(chunk != -1)[0]
                if len(valid_idx) > 0:
                    plt.step(start + valid_idx, chunk[valid_idx], where='post', 
                             label=f'Pig {tid}', color=colors(tid % 10), linewidth=2)

        plt.title(f'Behavior Ground Truth - {video_to_eval} - Clip {c+1:02d}')
        plt.xlabel('Frame')
        plt.ylabel('Action')
        plt.yticks(range(len(class_names)), class_names)
        plt.legend(loc='upper right', ncol=len(pigs_gt))
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.ylim(-0.5, len(class_names) - 0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, f"{c+1:02d}_gt.png"))
        plt.close()

    print(f">>> Successfully generated clip reports in: {report_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment folder name in out/results/")
    args = parser.parse_args()
    generate_full_report(args.exp)
