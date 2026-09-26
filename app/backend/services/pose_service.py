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

# Ensure the app/backend directory is importable so `utils.*` resolves both in
# the main process and when this file is launched as a subprocess (CWD is
# app/backend but the script dir is `services/`).
_APP_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_BACKEND not in sys.path:
    sys.path.insert(0, _APP_BACKEND)

from utils.frame_cache import FrameReader

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

    def process_coco_pose(self, coco_data: dict, video_path: str, batch_size: int = 32, padding_factor: float = 1.10,
                            checkpoint_path: str = None, resume: bool = False, chunk_frames: int = 25) -> dict:
        """Processes a COCO-formatted dictionary in memory, crops pigs, and extracts keypoints.

        Frame-chunked: annotations are grouped by frame and processed in chunks
        of `chunk_frames` frames so peak RAM stays bounded (the old version held
        all ~26k crops at once). After each chunk the partial COCO is flushed to
        `checkpoint_path` (if given) so a killed run can resume with `resume=True`
        instead of starting from zero.
        """
        coco_data['categories'] = self.categories

        images_map = {img['id']: img for img in coco_data.get('images', [])}
        annotations = coco_data.get('annotations', [])

        # Group annotation indices by image, preserving video order.
        ordered_image_ids = [img['id'] for img in coco_data.get('images', [])]
        by_image: dict = {}
        for idx, ann in enumerate(annotations):
            by_image.setdefault(ann['image_id'], []).append(idx)

        done_ids: set = set()
        if resume and checkpoint_path and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    ckpt = json.load(f)
                done_ids = set(ckpt.get('done_image_ids', []))
                ckpt_coco = ckpt.get('coco')
                if ckpt_coco and len(ckpt_coco.get('annotations', [])) == len(annotations):
                    coco_data = ckpt_coco
                    coco_data['categories'] = self.categories
                    annotations = coco_data.get('annotations', [])
                    by_image = {}
                    for idx, ann in enumerate(annotations):
                        by_image.setdefault(ann['image_id'], []).append(idx)
                logging.info(f"Resuming pose from checkpoint: {len(done_ids)} frames already done.")
            except Exception as e:
                logging.warning(f"Could not load pose checkpoint ({e}); starting from scratch.")
                done_ids = set()

        pending_image_ids = [iid for iid in ordered_image_ids if iid not in done_ids]
        logging.info(f"Pose chunks: {len(pending_image_ids)} pending frames "
                     f"({len(ordered_image_ids)} total, {len(done_ids)} resumed).")

        def _save_checkpoint():
            if not checkpoint_path:
                return
            try:
                with open(checkpoint_path, 'w') as f:
                    json.dump({"done_image_ids": sorted(done_ids), "coco": coco_data}, f)
            except Exception as e:
                logging.warning(f"Failed to write pose checkpoint: {e}")

        frame_reader = FrameReader(video_path)
        try:
            for start in range(0, len(pending_image_ids), chunk_frames):
                chunk_ids = pending_image_ids[start:start + chunk_frames]
                chunk_set = set(chunk_ids)
                all_crops = []
                all_meta = []  # (annotation_reference, offset_x, offset_y)

                for image_id in chunk_ids:
                    image_info = images_map.get(image_id)
                    if not image_info:
                        done_ids.add(image_id)
                        continue
                    frame_idx = image_info.get('frame_id', image_info.get('id'))
                    img = frame_reader.read(int(frame_idx))
                    if img is None:
                        logging.warning(f"Frame {frame_idx} could not be read from {video_path}")
                        for idx in by_image.get(image_id, []):
                            annotations[idx]['keypoints'] = []
                            annotations[idx]['num_keypoints'] = 0
                        done_ids.add(image_id)
                        continue

                    h_img, w_img = img.shape[:2]
                    for idx in by_image.get(image_id, []):
                        ann = annotations[idx]
                        seg = ann.get('segmentation')
                        if not isinstance(seg, dict) or 'counts' not in seg:
                            ann['keypoints'] = []
                            ann['num_keypoints'] = 0
                            continue
                        try:
                            mask = mask_utils.decode(seg)
                        except Exception:
                            ann['keypoints'] = []
                            ann['num_keypoints'] = 0
                            continue
                        if mask.shape[:2] == (h_img, w_img):
                            isolated = img * mask[:, :, np.newaxis]
                        else:
                            mask_r = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST).astype(bool)
                            isolated = img * mask_r[:, :, np.newaxis]

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

                if all_crops:
                    logging.info(f"Pose chunk frames {start + 1}-{start + len(chunk_ids)}: "
                                 f"inference on {len(all_crops)} crops...")
                    result_generator = self.inferencer(
                        all_crops,
                        batch_size=batch_size,
                        show=False,
                        return_vis=False
                    )
                    for i, result in enumerate(tqdm(result_generator, total=len(all_crops),
                                                   desc=f"MMPose {start + 1}-{start + len(chunk_ids)}")):
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

                done_ids.update(chunk_set)
                # Free chunk crops before the next iteration.
                del all_crops, all_meta
                _save_checkpoint()
                logging.info(f"Pose progress: {len(done_ids)}/{len(ordered_image_ids)} frames done.")
        finally:
            frame_reader.close()

        logging.info("Keypoints extraction pipeline complete.")
        return coco_data

# =====================================================================
# ADDED: EXECUTION INTERFACE AS A SUBPROCESS
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MMPose inference from an isolated environment subprocess.")
    parser.add_argument("--input_json", required=True, help="Path to temporary SAM COCO JSON.")
    parser.add_argument("--output_json", required=True, help="Path where the final JSON with keypoints will be saved.")
    parser.add_argument("--video_path", required=True, help="Path to the prepared video (frames are read from it).")
    parser.add_argument("--batch_size", type=int, default=32, help="Global batch size.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device index.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a JSON checkpoint file (partial progress, resume-safe).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from --checkpoint instead of starting from scratch.")
    parser.add_argument("--chunk_frames", type=int, default=25,
                        help="Frames per pose chunk (bounds peak RAM).")
    args = parser.parse_args()

    logging.info(f"Subprocess triggered. Loading input JSON: {args.input_json}")
    with open(args.input_json, 'r') as f:
        input_coco = json.load(f)

    service = PoseEstimationService(device=args.device)
    output_coco = service.process_coco_pose(
        coco_data=input_coco,
        video_path=args.video_path,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        chunk_frames=args.chunk_frames,
    )

    logging.info(f"Saving outputs back to: {args.output_json}")
    with open(args.output_json, 'w') as f:
        json.dump(output_coco, f)
        
    logging.info("Subprocess executed successfully.")