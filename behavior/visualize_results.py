import os
import json
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from behavior.behavior_lstm import behaviorLSTM

# Color palette for tracks
TRACK_COLORS = plt.cm.tab20.colors 

def extract_vector(ann, config, pose_config):
    bbox = ann['bbox'] # [x, y, w, h]
    kp_raw = ann['keypoints'] # [x, y, v, x, y, v, ...]
    
    # 1. BBox Features
    area = bbox[2] * bbox[3]
    ratio = bbox[2] / (bbox[3] + 1e-6)
    
    # 2. Keypoint Features
    kp_to_use = config.get('keypoints_to_use', [])
    full_kp_names = pose_config['categories'][0]['keypoints']
    
    kp_indices = [full_kp_names.index(name) for name in kp_to_use if name in full_kp_names]
    
    kp_vector = []
    center_x = bbox[0] + bbox[2] / 2
    center_y = bbox[1] + bbox[3] / 2
    norm_w = bbox[2] + 1e-6
    norm_h = bbox[3] + 1e-6
    
    for idx in kp_indices:
        start = idx * 3
        kx, ky, kv = kp_raw[start], kp_raw[start+1], kp_raw[start+2]
        
        nx = (kx - center_x) / norm_w
        ny = (ky - center_y) / norm_h
        conf = 1.0 if kv == 2 else (0.3 if kv == 1 else 0.0)
        kp_vector.extend([nx, ny, conf])
        
    geometry = [area/2e6, ratio/5, center_x/2688, center_y/1520, norm_w/2688, norm_h/1520]
    return np.array(geometry + kp_vector)

def plot_clip_transitions(video_id, clip_id, track_results, class_names, max_frame, output_path, version_type="Standard"):
    print(f">>> Saving {version_type} figure: {output_path}")
    plt.figure(figsize=(16, 9))
    
    sorted_tids = sorted(track_results.keys(), key=int)
    
    for tid in sorted_tids:
        data = track_results[tid]
        frames = data['frames']
        preds = data['preds']
        color = TRACK_COLORS[int(tid) % len(TRACK_COLORS)]
        
        # Plot continuous step line
        plt.step(frames, preds, where='post', color=color, alpha=0.6, linewidth=2.5, label=f'Pig {tid}')
        
        # Mark changes
        change_idx = [0] + [j for j in range(1, len(preds)) if preds[j] != preds[j-1]]
        plt.scatter([frames[j] for j in change_idx], [preds[j] for j in change_idx], 
                    color=color, s=50, edgecolors='black', zorder=10)

    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel('Frame ID')
    plt.ylabel('Behavior')
    plt.title(f'Pig Behavior Transitions ({version_type}) - Video: {video_id} | Clip: {clip_id}')
    
    plt.xlim(0, max_frame)
    plt.grid(True, axis='both', linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="ID")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def generate_paper_figures():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("data/pig_pose.yaml", "r") as f:
        pose_config = yaml.safe_load(f)
    
    ANN_DIR = "data/annotations/behavior"
    MODEL_PATH = "out/models/best_model.pth"
    BASE_TRANS_DIR = "out/results/transitions"
    
    if not os.path.exists(MODEL_PATH):
        print(f"!!! Error: Model path {MODEL_PATH} not found.")
        return

    behavior_classes = config.get('behavior_classes', {})
    class_names = list(behavior_classes.keys())
    window_size = config.get('window_size', 30)
    kp_to_use = config.get('keypoints_to_use', [])
    input_size = 6 + (len(kp_to_use) * 3)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Evaluating on: {device}")

    model = behaviorLSTM(input_size=input_size, hidden_size=128, num_layers=2, num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    for video_dir in sorted(os.listdir(ANN_DIR)):
        video_path = os.path.join(ANN_DIR, video_dir)
        if not os.path.isdir(video_path): continue
        
        # Paths for both versions
        std_dir = os.path.join(BASE_TRANS_DIR, "standard", video_dir)
        pad_dir = os.path.join(BASE_TRANS_DIR, "padded", video_dir)
        os.makedirs(std_dir, exist_ok=True)
        os.makedirs(pad_dir, exist_ok=True)
        
        for clip_file in sorted(os.listdir(video_path)):
            if not clip_file.endswith('.json'): continue
            
            clip_name = clip_file.replace('.json', '')
            
            with open(os.path.join(video_path, clip_file), 'r') as f:
                data = json.load(f)
            
            img_map = {img['id']: img for img in data['images']}
            max_clip_frame = max([img['frame_id'] for img in data['images']]) if data['images'] else 180
            
            tracks_raw = {}
            for ann in data['annotations']:
                tid = ann['track_id']
                if tid not in tracks_raw: tracks_raw[tid] = []
                tracks_raw[tid].append(ann)
            
            results_std = {}
            results_pad = {}

            for tid, anns in tracks_raw.items():
                anns = sorted(anns, key=lambda x: img_map[x['image_id']]['frame_id'])
                if len(anns) < window_size: continue
                
                feats = [extract_vector(a, config, pose_config) for a in anns]
                frames = [img_map[a['image_id']]['frame_id'] for a in anns]
                
                # --- VERSION 1: STANDARD (Starts after 30 frames) ---
                seq_std = []
                frames_std = []
                for j in range(len(feats) - window_size + 1):
                    seq_std.append(feats[j : j + window_size])
                    frames_std.append(frames[j + window_size - 1])
                
                # --- VERSION 2: PADDED (Starts from frame 0) ---
                padded_feats = [feats[0]] * (window_size - 1) + feats
                seq_pad = []
                for j in range(len(padded_feats) - window_size + 1):
                    seq_pad.append(padded_feats[j : j + window_size])
                
                # Inference
                for res_dict, seqs, f_list, v_name in [(results_std, seq_std, frames_std, "Standard"), 
                                                       (results_pad, seq_pad, frames, "Padded")]:
                    seq_tensor = torch.from_numpy(np.array(seqs)).float().to(device)
                    with torch.no_grad():
                        outputs = model(seq_tensor)
                        _, pred_ids = torch.max(outputs.data, 1)
                        res_dict[tid] = {
                            'frames': f_list,
                            'preds': pred_ids.cpu().numpy()
                        }
            
            if results_std:
                plot_clip_transitions(video_dir, clip_name, results_std, class_names, max_clip_frame, 
                                      os.path.join(std_dir, f"{clip_name}.png"), "Standard")
            if results_pad:
                plot_clip_transitions(video_dir, clip_name, results_pad, class_names, max_clip_frame, 
                                      os.path.join(pad_dir, f"{clip_name}.png"), "Padded")

    print(f"\n>>> ALL FIGURES GENERATED in {BASE_TRANS_DIR}")

if __name__ == "__main__":
    generate_paper_figures()
