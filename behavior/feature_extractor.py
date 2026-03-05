import json
import os
import numpy as np
import yaml
from pathlib import Path

def extract_features(src_dir, dst_dir, action_to_id, kp_indices):
    """
    Extracts features from behavior-labeled COCO JSONs.
    kp_indices: List of absolute indices [0-16] to extract from the COCO keypoints array.
    """
    print(f">>> Starting feature extraction from {src_dir}...")
    os.makedirs(dst_dir, exist_ok=True)

    # Iterate through videos
    for video_name in sorted(os.listdir(src_dir)):
        src_video_path = os.path.join(src_dir, video_name)
        if not os.path.isdir(src_video_path):
            continue
            
        print(f"  - Processing {video_name}...")
        
        tracks_data = {}

        for json_file in sorted(os.listdir(src_video_path)):
            if not json_file.endswith('.json'):
                continue
            
            with open(os.path.join(src_video_path, json_file), 'r') as f:
                data = json.load(f)
            
            images = {img['id']: img for img in data['images']}
            
            for ann in data['annotations']:
                track_id = ann.get('track_id')
                if track_id is None: continue
                
                img_meta = images[ann['image_id']]
                img_w, img_h = img_meta['width'], img_meta['height']
                
                # BBox features
                x, y, w, h = ann['bbox']
                cx, cy = x + w/2, y + h/2
                area = ann['area'] / (img_w * img_h)
                aspect_ratio = w / (h + 1e-6)
                
                cx_norm, cy_norm = cx / img_w, cy / img_h
                w_norm, h_norm = w / img_w, h / img_h
                bbox_feats = [cx_norm, cy_norm, w_norm, h_norm, area, aspect_ratio]
                
                # Pose features (Normalization relative to BBox)
                # Raw COCO keypoints: [x0, y0, v0, x1, y1, v1, ...]
                full_kpts = np.array(ann['keypoints']).reshape(-1, 3) 
                
                pose_feats = []
                for idx in kp_indices:
                    if idx < len(full_kpts):
                        kx, ky, kv = full_kpts[idx]
                        rx = (kx - cx) / (w + 1e-6)
                        ry = (ky - cy) / (h + 1e-6)
                        conf = 1.0 if kv == 2 else (0.3 if kv == 1 else 0.0)
                        pose_feats.extend([rx, ry, conf])
                    else:
                        pose_feats.extend([0.0, 0.0, 0.0]) # Padding if index out of range
                
                full_vector = bbox_feats + pose_feats
                
                # Label
                action_str = ann.get('action', list(action_to_id.keys())[0])
                
                # Combine Standing and Walking as requested
                if action_str in ["Standing", "Walking"] and "Standing_Walking" in action_to_id:
                    action_id = action_to_id["Standing_Walking"]
                else:
                    action_id = action_to_id.get(action_str, 0)
                
                frame_id = img_meta['frame_id']
                if track_id not in tracks_data:
                    tracks_data[track_id] = []
                
                tracks_data[track_id].append({
                    'frame_id': frame_id,
                    'features': full_vector,
                    'label': action_id
                })

        video_dst = os.path.join(dst_dir, video_name)
        os.makedirs(video_dst, exist_ok=True)
        
        for tid, instances in tracks_data.items():
            instances.sort(key=lambda x: x['frame_id'])
            
            feats_array = np.array([inst['features'] for inst in instances], dtype=np.float32)
            labels_array = np.array([inst['label'] for inst in instances], dtype=np.int32)
            frames_array = np.array([inst['frame_id'] for inst in instances], dtype=np.int32)
            
            save_path = os.path.join(video_dst, f"track_{tid}.npz")
            np.savez(save_path, features=feats_array, labels=labels_array, frames=frames_array)
            
    print(f"\n>>> FINISHED. Features saved in: {dst_dir}")

if __name__ == "__main__":
    # Load main config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load detailed pose config (index source)
    with open("data/pig_pose.yaml", "r") as f:
        pose_config = yaml.safe_load(f)
    
    action_ids = config.get('behavior_classes', {"Lying": 0})
    kp_to_use = config.get('keypoints_to_use', [])

    # Map name strings to their position in behavior/pig_pose.yaml
    full_kp_names = pose_config['categories'][0]['keypoints']
    kp_indices = []
    
    for name in kp_to_use:
        if name in full_kp_names:
            kp_indices.append(full_kp_names.index(name))
        else:
            print(f"!!! Warning: Keypoint '{name}' not found in pig_pose.yaml")

    SRC = "data/annotations/behavior"
    DST = "data/features"
    extract_features(SRC, DST, action_to_id=action_ids, kp_indices=kp_indices)
