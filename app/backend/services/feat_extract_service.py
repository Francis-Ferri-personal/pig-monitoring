import os
import numpy as np
import torch
import logging
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List, Tuple
from torchvision import models, transforms


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureExtractionService:
    def __init__(self, model_name: str = "resnet18", image_size: int = 224, batch_size: int = 16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.image_size = image_size
        
        # Initialize the CNN backbone once during service startup
        logging.info(f"Loading Feature Extraction Backbone: {model_name} on {self.device}")
        if model_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feature_dim = backbone.fc.in_features
            backbone.fc = torch.nn.Identity() # Strip the classification head
        elif model_name == "resnet34":
            backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.feature_dim = backbone.fc.in_features
            backbone.fc = torch.nn.Identity()
        else:
            raise ValueError(f"Model {model_name} not supported.")
            
        backbone.eval()
        self.backbone = backbone.to(self.device)
        
        # Standard ImageNet pre-processing transformation
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features_from_coco(
        self, 
        coco_data: Dict[str, Any], 
        frames_directory: str, 
        output_npz_path: str,
        padding_factor: float = 1.1
    ) -> str:
        """
        Processes an in-memory COCO dictionary, extracts spatio-temporal features 
        via batched CNN inferencing, and saves a consolidated .npz file structured by track.
        """
        logging.info(f"Starting feature extraction for frames in: {frames_directory}")
        
        backend_root = Path(__file__).resolve().parents[1]
        if not os.path.isabs(frames_directory):
            frames_dir_path = backend_root / frames_directory
        else:
            frames_dir_path = Path(frames_directory)
        
        # Fast lookup mapping for image metadata
        images_map = {img["id"]: img for img in coco_data.get("images", [])}
        tracks_data: Dict[int, List[Dict]] = {}

        # 1. Group individual object annotations by track_id
        for ann in coco_data.get("annotations", []):
            track_id = ann.get("track_id")
            if track_id is None:
                continue

            img_meta = images_map.get(ann["image_id"])
            if not img_meta:
                continue

            frame_id = img_meta.get("frame_id", img_meta.get("id"))
            
            if track_id not in tracks_data:
                tracks_data[track_id] = []

            tracks_data[track_id].append({
                "frame_id": int(frame_id),
                "bbox": ann["bbox"],
                "keypoints": ann.get("keypoints", []),
                "file_name": img_meta.get("file_name"),
                "img_size": (img_meta["width"], img_meta["height"])
            })

        # Master dictionary to be exported as a multi-keyed .npz archive
        npz_save_dict = {}

        # 2. Process each distinct tracked individual sequentially
        for tid, instances in tracks_data.items():
            # Ensure strict chronological frame ordering for the sequence
            instances.sort(key=lambda x: x["frame_id"])

            feats_list: List[np.ndarray] = []
            frames_list: List[int] = []
            
            prev_cx_norm, prev_cy_norm = None, None

            # Intermediate buffers for batch CNN inferencing
            batch_images: List[torch.Tensor] = []
            batch_bbox_feats: List[List[float]] = []
            batch_kp_feats: List[List[float]] = []
            batch_frames: List[int] = []

            def _process_batch():
                if not batch_bbox_feats:
                    return
                
                # Determine device type string safely for autocast control
                dev_type = "cuda" if "cuda" in str(self.device) else "cpu"

                # Execute batch CNN inference on target crops
                with torch.no_grad():

                    with torch.autocast(device_type=dev_type, enabled=False):
                        batch_tensor = torch.stack(batch_images, dim=0).to(self.device).float()
                        self.backbone = self.backbone.float()
                        emb_batch = self.backbone(batch_tensor).cpu().numpy().astype(np.float32)

                # Unpack and merge features for each instance within the current batch
                for j in range(len(batch_bbox_feats)):
                    # Concatenation map: [CNN (512) + BBox/Motion (11) + Keypoint Geometry (40)]
                    full_vector = np.concatenate([
                        emb_batch[j],
                        np.array(batch_bbox_feats[j], dtype=np.float32),
                        np.array(batch_kp_feats[j], dtype=np.float32)
                    ], axis=0)
                    
                    feats_list.append(full_vector)
                    frames_list.append(batch_frames[j])

                # Reset batch buffers
                batch_images.clear()
                batch_bbox_feats.clear()
                batch_kp_feats.clear()
                batch_frames.clear()

            # Dynamic kinematic calculation and crop aggregation loop
            for inst in instances:
                x, y, w, h = inst["bbox"]
                img_w, img_h = inst["img_size"]
                
                # Process structural BBox features and raw motion displacements
                bbox_feats, prev_cx_norm, prev_cy_norm = self._compute_bbox_features(
                    x, y, w, h, img_w, img_h, prev_cx_norm, prev_cy_norm
                )
                
                # Extract fixed-size engineering features from posture keypoints
                kp_feats = self._compute_keypoint_features(inst["keypoints"], img_w, img_h, (x, y, w, h))

                # Handle crop initialization from disk
                img_path = frames_dir_path / os.path.basename(inst["file_name"])
                if not img_path.exists():
                    logging.warning(f"Frame file not found on disk: {img_path}")
                    continue

                try:
                    with Image.open(img_path).convert("RGB") as img:
                        x1, y1, x2, y2 = self._pad_and_clip_bbox(x, y, w, h, img_w, img_h, padding_factor)
                        crop = img.crop((x1, y1, x2, y2))
                        tensor = self.transform(crop)
                    
                    batch_images.append(tensor)
                    batch_bbox_feats.append(bbox_feats)
                    batch_kp_feats.append(kp_feats)
                    batch_frames.append(inst["frame_id"])
                except Exception as e:
                    logging.error(f"Failed to process object crop from {img_path}: {e}")
                    continue

                # Trigger inference when the target batch size is reached
                if len(batch_images) >= self.batch_size:
                    _process_batch()

            # Process remaining trailing frames for the track sequence
            if batch_images:
                _process_batch()

            # Save arrays under distinct track references inside the file map
            if feats_list:
                npz_save_dict[f"track_{tid}_features"] = np.stack(feats_list, axis=0)
                npz_save_dict[f"track_{tid}_frames"] = np.array(frames_list, dtype=np.int32)

        # 3. Secure output synchronization
        if npz_save_dict:
            np.savez(output_npz_path, **npz_save_dict)
            logging.info(f"Multimodal features saved successfully at: {output_npz_path}")
            return output_npz_path
        else:
            raise ValueError("Feature vector construction empty. Check tracking inputs.")

    def _compute_bbox_features(self, x, y, w, h, img_w, img_h, prev_cx, prev_cy):
        """Calculates normalized spatial dimensions and scaled frame-to-frame displacement vectors."""
        cx, cy = x + w / 2.0, y + h / 2.0
        cx_n, cy_n, w_n, h_n = cx / img_w, cy / img_h, w / img_w, h / img_h
        area_n = (w * h) / (img_w * img_h)
        ratio = w / (h + 1e-6)
        dx, dy = (cx_n - prev_cx, cy_n - prev_cy) if prev_cx is not None else (0.0, 0.0)
        
        # Scaling factor (*10) prevents decimal underflow in sequential network layers
        return [cx_n, cy_n, w_n, h_n, area_n, ratio, dx*10, dy*10, abs(dx)*10, abs(dy)*10, (dx**2 + dy**2)**0.5 * 10], cx_n, cy_n

    def _compute_keypoint_features(self, kps, img_w, img_h, bbox):
        """Converts raw coordinate keypoints into 40 relative geometric invariant attributes."""
        if len(kps) < 51: return [0.0] * 40
        kp_arr = np.array(kps).reshape(-1, 3)
        ks, vis = kp_arr[:, :2], kp_arr[:, 2]
        diag = (bbox[2]**2 + bbox[3]**2)**0.5 + 1e-6
        
        # Structural distances relative to neck (ID 3) and tail-base (ID 4)
        dist_neck = np.linalg.norm(ks - ks[3], axis=1) / diag
        dist_tail = np.linalg.norm(ks - ks[4], axis=1) / diag
        body_v = ks[4] - ks[3]
        
        return dist_neck.tolist() + dist_tail.tolist() + [np.linalg.norm(body_v)/diag, np.arctan2(body_v[1], body_v[0])/np.pi] + (vis / 2.0)[[3, 4, 13, 16]].tolist()

    def _pad_and_clip_bbox(self, x, y, w, h, img_w, img_h, factor):
        """Applies configuration padding to bounding box coordinates while clipping edges inside image bounds."""
        cx, cy = x + w / 2.0, y + h / 2.0
        x1 = int(max(0, cx - (w * factor) / 2.0))
        y1 = int(max(0, cy - (h * factor) / 2.0))
        x2 = int(min(img_w, cx + (w * factor) / 2.0))
        y2 = int(min(img_h, cy + (h * factor) / 2.0))
        return x1, y1, max(x1+1, x2), max(y1+1, y2)