import os
import sys
import json
import logging
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pycocotools import mask as mask_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PoseEstimationService:
    """Service to handle MMPose batch keypoint extraction for pigs using SAM-derived data."""
    
    def __init__(self, device: str = "cuda:0"):
        # Aseguramos que el path local de mmpose esté accesible para los imports internos del framework
        sys.path.append(os.path.join(os.getcwd(), 'mmpose'))
        try:
            from mmpose.apis import MMPoseInferencer
            logging.info(f"Initializing MMPoseInferencer on device: {device}...")
            self.inferencer = MMPoseInferencer('animal', device=device)
            logging.info("MMPoseInferencer loaded successfully.")
        except ImportError as e:
            logging.error("MMPose is not installed or the directory wasn't found in system paths.")
            raise e

        self.categories = [
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

    def process_coco_pose(self, coco_data: dict, frames_dir: str, batch_size: int = 32, padding_factor: float = 1.10) -> dict:
        """Processes a COCO-formatted dictionary in memory, crops pigs, and extracts keypoints."""
        coco_data['categories'] = self.categories
        
        all_crops = []
        all_meta = []  # (annotation_reference, offset_x, offset_y)
        
        images_map = {img['id']: img for img in coco_data.get('images', [])}
        annotations = coco_data.get('annotations', [])
        
        logging.info(f"Extracting crops from {len(images_map)} cached frames...")
        
        for ann in annotations:
            image_info = images_map.get(ann['image_id'])
            if not image_info:
                continue
                
            img_filename = Path(image_info['file_name']).name
            img_path = os.path.join(frames_dir, img_filename)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Frame file could not be read: {img_path}")
                continue
                
            h_img, w_img = img.shape[:2]
            
            mask = mask_utils.decode(ann['segmentation'])
            isolated = img * mask[:, :, np.newaxis]
            
            x_bb, y_bb, w_bb, h_bb = ann['bbox']
            cx, cy = x_bb + w_bb / 2, y_bb + h_bb / 2
            nw, nh = w_bb * padding_factor, h_bb * padding_factor
            
            x1 = int(max(0, cx - nw / 2))
            y1 = int(max(0, cy - nh / 2))
            x2 = int(min(w_img, cx + nw / 2))
            y2 = int(min(h_img, cy + nh / 2))
            
            crop = isolated[y1:y2, x1:x2]
            
            if crop.size == 0:
                ann['keypoints'] = []
                ann['num_keypoints'] = 0
                continue
                
            all_crops.append(crop)
            all_meta.append((ann, x1, y1))
            
        if not all_crops:
            logging.warning("No pigs were found across any annotations in this batch.")
            return coco_data

        logging.info(f"Running global MMPose inference on {len(all_crops)} items (Batch Size: {batch_size})...")
        result_generator = self.inferencer(
            all_crops,
            batch_size=batch_size,
            show=False,
            return_vis=False
        )
        
        logging.info("Mapping calculated keypoints back to global video frame coordinates...")
        for i, result in enumerate(tqdm(result_generator, total=len(all_crops), desc="MMPose Batch")):
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
                    if v > 0: 
                        num_keypoints += 1
                        
                ann['keypoints'] = coco_keypoints
                ann['num_keypoints'] = num_keypoints
                
            except (IndexError, KeyError, TypeError):
                ann['keypoints'] = []
                ann['num_keypoints'] = 0
                
        logging.info("Keypoints extraction pipeline complete.")
        return coco_data

# =====================================================================
# ADDED: EXECUTION INTERFACE AS A SUBPROCESS
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MMPose inference from an isolated environment subprocess.")
    parser.add_argument("--input_json", required=True, help="Path to temporary SAM COCO JSON.")
    parser.add_argument("--output_json", required=True, help="Path where the final JSON with keypoints will be saved.")
    parser.add_argument("--frames_dir", required=True, help="Directory containing the target raw frames.")
    parser.add_argument("--batch_size", type=int, default=32, help="Global batch size.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device index.")
    args = parser.parse_args()

    logging.info(f"Subprocess triggered. Loading input JSON: {args.input_json}")
    with open(args.input_json, 'r') as f:
        input_coco = json.load(f)

    service = PoseEstimationService(device=args.device)
    output_coco = service.process_coco_pose(
        coco_data=input_coco,
        frames_dir=args.frames_dir,
        batch_size=args.batch_size
    )

    logging.info(f"Saving outputs back to: {args.output_json}")
    with open(args.output_json, 'w') as f:
        json.dump(output_coco, f)
        
    logging.info("Subprocess executed successfully.")