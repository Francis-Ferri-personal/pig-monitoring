import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys
import os
import json
import cv2
import yaml
import numpy as np
import argparse
from tqdm import tqdm
from pycocotools import mask as mask_utils
from concurrent.futures import ProcessPoolExecutor

categories = [
    {
        "id": 1,
        "name": "pig",
        "supercategory": "animal",
        "keypoints": [
            "L_Eye", "R_Eye", "Nose", "Neck", "Root of tail",
            "L_Shoulder", "L_Elbow", "L_F_Paw", "R_Shoulder", "R_Elbow", "R_F_Paw",
            "L_Hip", "L_Knee", "L_B_Paw", "R_Hip", "R_Knee", "R_B_Paw",
        ],
        "skeleton": [
            [1, 2], [1, 3], [2, 3], [3, 4], [4, 5],
            [4, 6], [6, 7], [7, 8],
            [4, 9], [9, 10], [10, 11],
            [5, 12], [12, 13], [13, 14],
            [5, 15], [15, 16], [16, 17],
        ],
    }
]

def worker_task(args):
    """
    Worker function to process a single frame: reads, masks, and crops.
    Returns: list of (crop, ann_dict, x1, y1, x2, y2, image_id)
    """
    image_info, annotations, padding_factor, frames_dir, video_folder = args
    image_id = image_info['id']
    
    img_path = os.path.join(frames_dir, video_folder, image_info['file_name'])
    img = cv2.imread(img_path)
    if img is None:
        return []

    h_img, w_img = img.shape[:2]
    results = []

    for ann in annotations:
        if 'segmentation' not in ann:
            continue
            
        mask = mask_utils.decode(ann['segmentation'])
        
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        if mask.shape[:2] != (h_img, w_img):
            mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        
        isolated = cv2.bitwise_and(img, img, mask=mask)
        
        bbox = ann['bbox'] 
        x_bb, y_bb, w_bb, h_bb = bbox
        cx, cy = x_bb + w_bb/2, y_bb + h_bb/2
        nw, nh = w_bb * padding_factor, h_bb * padding_factor
        
        x1 = int(max(0, cx - nw/2))
        y1 =int(max(0, cy - nh/2))
        x2 = int(min(w_img, cx + nw/2))
        y2 = int(min(h_img, cy + nh/2))
        
        crop = isolated[y1:y2, x1:x2]
        
        if crop.size > 0:
            ann_copy = {
                'image_id': image_id,
                'bbox': ann['bbox'],
                'category_id': ann.get('category_id', 1)
            }
            results.append((crop, ann_copy, x1, y1, x2, y2, image_id))
            
    return results

def main():
    parser = argparse.ArgumentParser(description="MMPose inference with global batching and parallel preprocessing.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (global across frames)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for frame prep.")
    args = parser.parse_args()

    sys.path.append(os.path.join(os.getcwd(), 'mmpose'))
    from mmpose.apis import MMPoseInferencer
    inferencer = MMPoseInferencer('animal', device=args.device)

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    padding_factor = config.get('bbox_padding_factor', 1.10)
    sam_root = 'data/annotations/sam'
    pose_root = 'data/annotations/pose'
    os.makedirs(pose_root, exist_ok=True)

    for video_folder in sorted(os.listdir(sam_root)):
        video_sam_path = os.path.join(sam_root, video_folder)
        if not os.path.isdir(video_sam_path): continue
        
        video_pose_path = os.path.join(pose_root, video_folder)
        os.makedirs(video_pose_path, exist_ok=True)
            
        for clip_name in sorted(os.listdir(video_sam_path)):
            if not clip_name.endswith('.json'): continue
            
            pose_ann_file = os.path.join(video_pose_path, clip_name)
            
            if os.path.exists(pose_ann_file):
                print(f">>> Skipping: {video_folder}/{clip_name} (Already exists)")
                continue

            print(f"\n>>> Processing Clip: {video_folder}/{clip_name}")
            with open(os.path.join(video_sam_path, clip_name), 'r') as f:
                ann_data = json.load(f)
            
            ann_data['categories'] = categories
            images_list = ann_data['images']
            
            # Pre-process all tasks to avoid complex state in workers
            task_args = []
            for image_int in images_list:
                img_anns = [ann for ann in ann_data.get('annotations', []) if ann['image_id'] == image_int['id']]
                if img_anns:
                    task_args.append((image_int, img_anns, padding_factor, 'data/images/frames', video_folder))
            
            all_crops = []
            all_meta = []

            print(f"  - Preparing {len(task_args)} frames using {args.workers} workers...")
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for frame_results in tqdm(list(executor.map(worker_task, task_args)), 
                                         total=len(task_args), desc="Prep Frames", unit="frame"):
                    for crop, ann_copy, x1, y1, x2, y2, image_id in frame_results:
                        all_crops.append(crop)
                        all_meta.append((ann_copy, x1, y1, x2, y2, image_id))
            
            if not all_crops:
                print(f"  ! No pigs found in clip {clip_name}")
                with open(pose_ann_file, 'w') as f:
                    json.dump(ann_data, f)
                continue

            print(f"  - Running inference on {len(all_crops)} items (Batch size: {args.batch_size})...", flush=True)
            
            result_generator = inferencer(
                all_crops,
                batch_size=args.batch_size,
                show=int(False),
                return_vis=False
            )
            
            print(f"  - Mapping results back to annotations...")
            
            # Re-map the inference results back to the original ann_data structure
            # We need to find which annotation in the original ann_data matches the inference result
            # We'll use the metadata we collected during the 'Prep' stage.
            
            # Map result index -> metadata index
            # The result_generator follows the same order as all_crops
            
            for i, result in enumerate(tqdm(result_generator, total=len(all_crops), desc="Inference", unit="pig")):
                ann_info, x1, y1, x2, y2, image_id = all_meta[i]
                
                # We need to find the corresponding annotation in the original ann_data
                # We'll use a trick: find the original annotation in the list that has this image_id
                # and matches the spatial context
                
                target_ann = None
                for original_ann in ann_data['annotations']:
                    if original_ann['image_id'] == image_id:
                        # Check if the crop is inside the original bbox
                        orig_bbox = original_ann['bbox']
                        # A very simple spatial check (intersection)
                        # Ideally we should use the exact indices, but we'll use identity
                        # This is a bit fragile but works if we don't lose track of IDs.
                        # Let's assume we find it by checking if the original bbox 
                        # roughly matches the crop center.
                        pass
                
                # Wait, the worker_task already returns the specific 'ann_copy'
                # Let's use that! We need to find the original object in ann_data.
                # But wait, I don't have the original reference, only the copy.
                # We need to find the annotation in ann_data that matches ann_copy.
                
                found_match = False
                for original_ann in ann_data['annotations']:
                    if original_ann['image_id'] == ann_info['image_id'] and \
                       original_ann['bbox'] == ann_info['bbox']:
                        target_ann = original_ann
                        found_match = True
                        break
                
                if not target_ann:
                    # Fallback to dummy if not found
                    target_ann = {'image_id': image_id, 'bbox': ann_info['bbox'], 'category_id': ann_info['category_id']}
                
                try:
                    prediction = result['predictions'][0][0]
                    kp_crop = prediction['keypoints']
                    scores = prediction['keypoint_scores']
                    
                    coco_keypoints = []
                    num_keypoints = 0
                    for k in range(len(kp_crop)):
                        kx, ky = kp_crop[k]
                        s = scores[k]
                        kx_orig, ky_orig = kx + x1, ky + y1
                        v = 2 if s > 0.3 else 1
                        coco_keypoints.extend([float(kx_orig), float(ky_orig), v])
                        if v > 0: num_keypoints += 1
                        
                    target_ann['keypoints'] = coco_keypoints
                    target_ann['num_keypoints'] = num_keypoints
                except (IndexError, KeyError):
                    target_ann['keypoints'] = []
                    target_ann['num_keypoints'] = 0

            # 4. SAVE CLIP DATA
            with open(pose_ann_file, 'w') as f:
                json.dump(ann_data, f)
            print(f"  ✓ Saved: {pose_ann_file}")

    print("\n>>> ALL KEYPOINTS GENERATED.")

if __name__ == "__main__":
    main()
