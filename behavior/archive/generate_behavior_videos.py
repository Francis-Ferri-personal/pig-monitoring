import os
import cv2
import json
import torch
import yaml
import numpy as np
import argparse
from behavior.models import BehaviorRNN

def generate_prediction_videos(exp_name, video_to_eval='video3'):
    exp_dir = os.path.join("out", "results", exp_name)
    with open(os.path.join(exp_dir, "config_used.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = list(config['behavior_classes'].keys())
    
    # 1. Load Model
    num_kpts = len(config.get('keypoints_to_use', []))
    input_size = 6 + (num_kpts * 3)
    
    model = BehaviorRNN(
        rnn_type=config.get('rnn_type', 'LSTM'),
        input_size=input_size, 
        hidden_size=config.get('hidden_size', 64),
        num_layers=config.get('num_layers', 2),
        num_classes=len(class_names)
    ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pt")))
    model.eval()

    # 2. Run Inference to get temporal predictions
    feat_video_dir = os.path.join("data/features", video_to_eval)
    window_size = config.get('window_size', 30)
    
    pigs_predictions = {}
    
    print(f">>> Running inference for {video_to_eval}...")
    for npz_file in sorted(os.listdir(feat_video_dir)):
        if not npz_file.endswith('.npz'): continue
        
        data = np.load(os.path.join(feat_video_dir, npz_file))
        feats = data['features']
        frames = data['frames']
        track_id = int(npz_file.replace('.npz', '').replace('track_', ''))
        
        num_frames = feats.shape[0]
        # Map original frame_id to predicted label
        track_pred_map = {}
        
        with torch.no_grad():
            for i in range(num_frames - window_size + 1):
                window = torch.from_numpy(feats[i : i + window_size]).float().unsqueeze(0).to(device)
                output = model(window)
                _, pred = torch.max(output, 1)
                
                # The prediction corresponds to the last frame of the window
                frame_idx = frames[i + window_size - 1]
                track_pred_map[frame_idx] = pred.item()
                
        pigs_predictions[track_id] = track_pred_map

    # 3. Generate Videos
    out_vid_dir = os.path.join(exp_dir, "videos", video_to_eval)
    os.makedirs(out_vid_dir, exist_ok=True)
    
    anns_dir = os.path.join("data/annotations/behavior", video_to_eval)
    frames_dir = os.path.join("data/images/frames", video_to_eval)
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)] # Map to 5 tracks approx
    
    print(f">>> Generating clips in {out_vid_dir}...")
    
    for json_file in sorted(os.listdir(anns_dir)):
        if not json_file.endswith('.json'): continue
        
        clip_id = json_file.replace('.json', '')
        print(f"  - Creating {clip_id}.mp4...")
        
        # Load clip anns
        with open(os.path.join(anns_dir, json_file), 'r') as f:
            clip_data = json.load(f)
            
        images_info = {img['id']: img for img in clip_data['images']}
        img_id_to_frame = {img['id']: img['frame_id'] for img in clip_data['images']}
        
        # Group anns by image_id
        anns_by_img = {}
        for ann in clip_data['annotations']:
            img_id = ann['image_id']
            if img_id not in anns_by_img:
                anns_by_img[img_id] = []
            anns_by_img[img_id].append(ann)
            
        # Video writer
        clip_frames_dir = os.path.join(frames_dir, clip_id)
        if not os.path.exists(clip_frames_dir):
            print(f"!!! Frames missing for {clip_frames_dir}, skipping.")
            continue
            
        sample_img = cv2.imread(os.path.join(clip_frames_dir, "00000.png"))
        if sample_img is None: continue
        h, w, _ = sample_img.shape
        
        out_mp4 = os.path.join(out_vid_dir, f"{clip_id}.mp4")
        # Use mp4v codec for standard mp4 compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = config.get('frames_per_second', 1) # Note: if frames were extracted at 1fps
        
        writer = cv2.VideoWriter(out_mp4, fourcc, out_fps, (w, h))
        
        # Iterate sorted frames
        for img_info in sorted(clip_data['images'], key=lambda x: x['frame_id']):
            img_id = img_info['id']
            frame_abs = img_info['frame_id']
            
            fname = os.path.basename(img_info['file_name'])
            img_path = os.path.join(clip_frames_dir, fname)
            
            frame_img = cv2.imread(img_path)
            if frame_img is None: continue
            
            # Draw predictions
            for ann in anns_by_img.get(img_id, []):
                track_id = ann.get('track_id')
                if track_id is None: continue
                
                # BBox
                x, y, w_box, h_box = map(int, ann['bbox'])
                color = colors[track_id % len(colors)]
                cv2.rectangle(frame_img, (x, y), (x + w_box, y + h_box), color, 2)
                
                # Get Predicted Label
                pred_label = "Waiting"
                if track_id in pigs_predictions and frame_abs in pigs_predictions[track_id]:
                    pred_idx = pigs_predictions[track_id][frame_abs]
                    pred_label = class_names[pred_idx]
                
                # Ground truth
                gt_label = ann.get('action', 'Unknown')
                
                # Draw texts
                text_pred = f"Pred: {pred_label}"
                text_gt = f"GT: {gt_label}"
                
                # Text styling
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale_pred = 0.8
                font_scale_gt = 0.6
                thick_pred = 2
                thick_gt = 1
                
                # Get text bounds
                (tw_pred, th_pred), _ = cv2.getTextSize(text_pred, font, font_scale_pred, thick_pred)
                (tw_gt, th_gt), _ = cv2.getTextSize(text_gt, font, font_scale_gt, thick_gt)
                
                bg_width = max(tw_pred, tw_gt) + 10
                bg_height = th_pred + th_gt + 15
                
                # Draw background box for text
                cv2.rectangle(frame_img, (x, max(0, y - bg_height)), (x + bg_width, y), color, -1)
                
                # Text White/Gray
                cv2.putText(frame_img, text_pred, (x + 5, max(th_pred + 5, y - th_gt - 10)), font, font_scale_pred, (255, 255, 255), thick_pred)
                cv2.putText(frame_img, text_gt, (x + 5, max(bg_height - 5, y - 5)), font, font_scale_gt, (200, 200, 200), thick_gt)

            writer.write(frame_img)
        
        writer.release()
        print(f"    Saved {out_mp4}")
        
    print(f"\n>>> FINISHED generating video reports for {video_to_eval} in: {out_vid_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment folder name in out/results/")
    parser.add_argument("--fps", type=int, default=1, help="Output FPS")
    args = parser.parse_args()
    
    generate_prediction_videos(args.exp)
