import sys
import os
import json
import cv2
import yaml
import numpy as np
import argparse
from tqdm import tqdm
from pycocotools import mask as mask_utils

categories = [
    {
        "id": 1,
        "name": "pig",
        "supercategory": "animal",
        "keypoints": [
            "L_Eye", "R_Eye", "Nose", "Neck", "Root of tail",
            "L_Shoulder", "L_Elbow", "L_F_Paw", "R_Shoulder", "R_Elbow", "R_F_Paw",
            "L_Hip", "L_Knee", "L_B_Paw", "R_Hip", "R_Knee", "R_B_Paw"
        ],
        "skeleton": [
            [1, 2], [1, 3], [2, 3], [3, 4], [4, 5],
            [4, 6], [6, 7], [7, 8],
            [4, 9], [9, 10], [10, 11],
            [5, 12], [12, 13], [13, 14],
            [5, 15], [15, 16], [16, 17]
        ]
    }
]

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="MMPose inference with global batching.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (global across frames)")
    args = parser.parse_args()

    # --- MMPose Setup ---
    sys.path.append(os.path.join(os.getcwd(), 'mmpose'))
    from mmpose.apis import MMPoseInferencer
    inferencer = MMPoseInferencer('animal', device=args.device)

    # --- Load Config ---
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    padding_factor = config.get('bbox_padding_factor', 1.10)
    sam_root = 'data/annotations/sam'
    pose_root = 'data/annotations/pose'
    os.makedirs(pose_root, exist_ok=True)

    # Process videos
    for video_folder in sorted(os.listdir(sam_root)):
        video_sam_path = os.path.join(sam_root, video_folder)
        if not os.path.isdir(video_sam_path): continue
        
        video_pose_path = os.path.join(pose_root, video_folder)
        os.makedirs(video_pose_path, exist_ok=True)
            
        for clip_name in sorted(os.listdir(video_sam_path)):
            if not clip_name.endswith('.json'): continue
            
            pose_ann_file = os.path.join(video_pose_path, clip_name)
            
            # --- RESUME LOGIC ---
            if os.path.exists(pose_ann_file):
                print(f">>> Skipping: {video_folder}/{clip_name} (Already exists)")
                continue

            print(f"\n>>> Processing Clip: {video_folder}/{clip_name}")
            with open(os.path.join(video_sam_path, clip_name), 'r') as f:
                ann_data = json.load(f)
            
            # Update categories
            ann_data['categories'] = categories
            
            # 1. ACCUMULATE ALL CROPS FOR THE ENTIRE CLIP
            all_crops = []
            all_meta = [] # To store (ann_object, offset_x, offset_y)
            
            images_list = ann_data['images']
            print(f"  - Accumulating crops from {len(images_list)} frames...")
            
            for image_info in tqdm(images_list, desc="Prep Frames", unit="frame"):
                img_path = os.path.join('data/images/frames', video_folder, image_info['file_name'])
                img = cv2.imread(img_path)
                if img is None: continue
                h_img, w_img = img.shape[:2]
                
                img_anns = [ann for ann in ann_data.get('annotations', []) if ann['image_id'] == image_info['id']]
                
                for ann in img_anns:
                    # Isolate target pig using SAM mask
                    mask = mask_utils.decode(ann['segmentation'])
                    isolated = img * mask[:, :, np.newaxis]
                    
                    # Calculate padded bbox
                    bbox = ann['bbox'] # [x, y, w, h]
                    x_bb, y_bb, w_bb, h_bb = bbox
                    cx, cy = x_bb + w_bb/2, y_bb + h_bb/2
                    nw, nh = w_bb * padding_factor, h_bb * padding_factor
                    
                    x1 = int(max(0, cx - nw/2))
                    y1 = int(max(0, cy - nh/2))
                    x2 = int(min(w_img, cx + nw/2))
                    y2 = int(min(h_img, cy + nh/2))
                    
                    crop = isolated[y1:y2, x1:x2]
                    
                    all_crops.append(crop)
                    all_meta.append((ann, x1, y1))
            
            if not all_crops:
                print(f"  ! No pigs found in clip {clip_name}")
                with open(pose_ann_file, 'w') as f:
                    json.dump(ann_data, f)
                continue

            # 2. RUN BATCH INFERENCE (Truly global batching)
            print(f"  - Running inference on {len(all_crops)} items (Batch size: {args.batch_size})...", flush=True)
            
            # MMPoseInferencer doesn't natively support tqdm easily during internal loop
            # setting show_progress=True in constructor might show a simple rich bar
            
            result_generator = inferencer(
                all_crops,
                batch_size=args.batch_size,
                show=False,
                return_vis=False
            )
            
            # 3. MAP RESULTS BACK
            print(f"  - Mapping results back to annotations...")
            for i, result in enumerate(tqdm(result_generator, total=len(all_crops), desc="Inference", unit="pig")):
                ann, off_x, off_y = all_meta[i]
                
                try:
                    prediction = result['predictions'][0][0]
                    kp_crop = prediction['keypoints']
                    scores = prediction['keypoint_scores']
                    
                    coco_keypoints = []
                    num_keypoints = 0
                    for k in range(len(kp_crop)):
                        kx, ky = kp_crop[k]
                        s = scores[k]
                        kx_orig, ky_orig = kx + off_x, ky + off_y
                        v = 2 if s > 0.3 else 1
                        coco_keypoints.extend([float(kx_orig), float(ky_orig), v])
                        if v > 0: num_keypoints += 1
                        
                    ann['keypoints'] = coco_keypoints
                    ann['num_keypoints'] = num_keypoints
                except (IndexError, KeyError):
                    ann['keypoints'] = []
                    ann['num_keypoints'] = 0

            # 4. SAVE CLIP DATA
            with open(pose_ann_file, 'w') as f:
                json.dump(ann_data, f)
            print(f"  ✓ Saved: {pose_ann_file}")

    print("\n>>> ALL KEYPOINTS GENERATED.")

if __name__ == "__main__":
    main()