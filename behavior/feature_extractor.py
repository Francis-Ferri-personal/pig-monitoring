import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import models, transforms

#TODO: ADD keypoints features


def load_cnn_device(model_name: str = "resnet18") -> Tuple[torch.nn.Module, torch.device, int]:
    """
    Load a pretrained CNN backbone and return it without the classification head.
    Returns (model, device, feature_dim).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "resnet18":
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feature_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
    elif model_name == "resnet34":
        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        feature_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported model_name={model_name}. Use 'resnet18' or 'resnet34'.")

    backbone.eval()
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    return backbone, device, feature_dim


def get_image_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def compute_bbox_features(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    prev_cx_norm: float = None,
    prev_cy_norm: float = None,
) -> Tuple[List[float], float, float]:
    """
    Geometry + enriched motion features from bbox.
    Returns (features, cx_norm, cy_norm).
    """
    cx = x + w / 2.0
    cy = y + h / 2.0

    cx_norm = cx / img_w
    cy_norm = cy / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    area_norm = (w * h) / (img_w * img_h)
    aspect_ratio = w / (h + 1e-6)

    if prev_cx_norm is None or prev_cy_norm is None:
        dx = 0.0
        dy = 0.0
    else:
        dx = cx_norm - prev_cx_norm
        dy = cy_norm - prev_cy_norm

    # Enriched motion representation
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    speed = (dx**2 + dy**2) ** 0.5

    # Optional scaling to give more weight to motion
    motion_scale = 10.0
    dx_s = dx * motion_scale
    dy_s = dy * motion_scale
    abs_dx_s = abs_dx * motion_scale
    abs_dy_s = abs_dy * motion_scale
    speed_s = speed * motion_scale

    feats = [
        cx_norm,
        cy_norm,
        w_norm,
        h_norm,
        area_norm,
        aspect_ratio,
        dx_s,
        dy_s,
        abs_dx_s,
        abs_dy_s,
        speed_s,
    ]
    return feats, cx_norm, cy_norm


def pad_and_clip_bbox(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    padding_factor: float,
) -> Tuple[int, int, int, int]:
    """
    Apply a padding factor to bbox and clip to image boundaries.
    """
    cx = x + w / 2.0
    cy = y + h / 2.0
    padded_w = w * padding_factor
    padded_h = h * padding_factor

    x1 = int(max(0, cx - padded_w / 2.0))
    y1 = int(max(0, cy - padded_h / 2.0))
    x2 = int(min(img_w, cx + padded_w / 2.0))
    y2 = int(min(img_h, cy + padded_h / 2.0))

    # Ensure at least 1 pixel width/height
    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)

    return x1, y1, x2, y2


ALLOWED_TRACK_IDS = {0, 1, 2, 3, 4}


def extract_features(
    src_dir: str,
    dst_dir: str,
    frames_root: str,
    action_to_id: Dict[str, int],
    cnn_name: str = "resnet18",
    image_size: int = 224,
    bbox_padding_factor: float = 1.1,
    batch_size: int = 16,
    use_keypoints: bool = False,
) -> None:
    """
    Extract embeddings + bbox geometry + motion features from behavior-labeled COCO JSONs.

    - Uses crops from data/images/frames (no masked images).
    - For each track, stores:
        features: [T, D_visual + 11]  (D_visual from CNN, + 11 for bbox geom + enriched motion)
        labels:   [T]
        frames:   [T]
    """
    print(f">>> Starting feature extraction from {src_dir}...")
    os.makedirs(dst_dir, exist_ok=True)

    backbone, device, feat_dim = load_cnn_device(cnn_name)
    transform = get_image_transform(image_size=image_size)

    src_dir_path = Path(src_dir)
    frames_root_path = Path(frames_root)

    total_videos = 0
    total_tracks = 0

    # Iterate through videos (e.g. video1, video2, video3)
    for video_name in sorted(os.listdir(src_dir)):
        src_video_path = src_dir_path / video_name
        if not src_video_path.is_dir():
            continue
        total_videos += 1

        print(f"\n  >>> Processing {video_name}...")

        # Precompute global frame offset per clip (same order as sorted json_file)
        clip_offsets: Dict[str, int] = {}
        running_offset = 0
        for jf in sorted(os.listdir(src_video_path)):
            if not jf.endswith(".json"):
                continue
            with open(src_video_path / jf, "r") as f:
                d = json.load(f)
            n_frames = len(d.get("images", []))
            clip_offsets[jf.replace(".json", "")] = running_offset
            running_offset += n_frames

        # First pass: gather per-track sequences (bbox + label + img_meta + global_frame_id)
        tracks_data: Dict[int, List[Dict]] = {}

        for json_file in sorted(os.listdir(src_video_path)):
            if not json_file.endswith(".json"):
                continue

            clip_id = json_file.replace(".json", "")
            json_path = src_video_path / json_file
            with open(json_path, "r") as f:
                data = json.load(f)

            images = {img["id"]: img for img in data["images"]}
            offset = clip_offsets.get(clip_id, 0)

            for ann in data["annotations"]:
                track_id = ann.get("track_id")
                if track_id is None:
                    continue
                if track_id not in ALLOWED_TRACK_IDS:
                    # Skip any track_id outside the expected 5 pigs
                    continue

                img_meta = images[ann["image_id"]]
                img_w, img_h = img_meta["width"], img_meta["height"]

                x, y, w, h = ann["bbox"]

                action_str = ann.get("action", list(action_to_id.keys())[0])
                if action_str in ["Standing", "Walking"] and "Standing_Walking" in action_to_id:
                    action_id = action_to_id["Standing_Walking"]
                else:
                    action_id = action_to_id.get(action_str, 0)

                frame_id = img_meta.get("frame_id", img_meta.get("id"))
                global_frame_id = offset + int(frame_id)

                file_name = img_meta.get("file_name")
                if file_name is None:
                    continue

                if track_id not in tracks_data:
                    tracks_data[track_id] = []

                tracks_data[track_id].append(
                    {
                        "frame_id": frame_id,
                        "global_frame_id": global_frame_id,
                        "bbox": (x, y, w, h),
                        "keypoints": ann.get("keypoints", []),
                        "img_meta": img_meta,
                        "label": action_id,
                        "file_name": file_name,
                        "img_size": (img_w, img_h),
                    }
                )

        # Second pass: for each track, sort by global_frame_id, compute geometry+motion+CNN embedding
        video_dst = Path(dst_dir) / video_name
        video_dst.mkdir(parents=True, exist_ok=True)

        track_ids = sorted(tracks_data.keys())
        print(f"    Found {len(track_ids)} valid tracks in {video_name}")
        for idx, tid in enumerate(track_ids, start=1):
            save_path = video_dst / f"track_{tid}.npz"
            if save_path.exists():
                # Resume: skip tracks already processed
                continue

            instances = tracks_data[tid]
            instances.sort(key=lambda x: x["global_frame_id"])

            feats_list: List[np.ndarray] = []
            labels_list: List[int] = []
            frames_list: List[int] = []

            prev_cx_norm = None
            prev_cy_norm = None

            batch_images: List[torch.Tensor] = []
            batch_bbox_feats: List[List[float]] = []
            batch_kp_feats: List[List[float]] = []
            batch_labels: List[int] = []
            batch_frames: List[int] = []

            for inst in instances:
                frame_id = inst["global_frame_id"]  # use global for saving to NPZ
                x, y, w, h = inst["bbox"]
                img_w, img_h = inst["img_size"]
                file_name = inst["file_name"]

                # Resolve image path robustly. annotations may store file_name as
                # "clip/frame.png" or just "frame.png". Try several sensible
                # locations before skipping.
                img_path = frames_root_path / video_name / file_name
                if not img_path.exists():
                    img_path_alt = frames_root_path / file_name
                    img_path_alt2 = frames_root_path / video_name / os.path.basename(file_name)
                    if img_path_alt.exists():
                        img_path = img_path_alt
                    elif img_path_alt2.exists():
                        img_path = img_path_alt2
                    else:
                        print(f"!!! Missing frame for track {tid}: tried {img_path}, {img_path_alt}, {img_path_alt2}")
                        continue

                bbox_feats, prev_cx_norm, prev_cy_norm = compute_bbox_features(
                    x,
                    y,
                    w,
                    h,
                    img_w,
                    img_h,
                    prev_cx_norm=prev_cx_norm,
                    prev_cy_norm=prev_cy_norm,
                )

                x1, y1, x2, y2 = pad_and_clip_bbox(
                    x,
                    y,
                    w,
                    h,
                    img_w,
                    img_h,
                    padding_factor=bbox_padding_factor,
                )

                with Image.open(img_path).convert("RGB") as img:
                    crop = img.crop((x1, y1, x2, y2))
                    tensor = transform(crop)

                batch_images.append(tensor)
                batch_bbox_feats.append(bbox_feats)

                if use_keypoints:
                    keypoints = inst.get("keypoints", [])
                    kp_feat = []
                    if keypoints and len(keypoints) >= 51:
                        for i in range(0, 51, 3):
                            kx, ky, kv = keypoints[i], keypoints[i+1], keypoints[i+2]
                            kp_feat.extend([kx / img_w, ky / img_h, kv])
                    else:
                        kp_feat = [0.0] * 51
                    batch_kp_feats.append(kp_feat)

                batch_labels.append(inst["label"])
                batch_frames.append(frame_id)

            # end for inst
            # Optional: report how many frames were gathered for this track so user can detect skips
            # (will appear when saving or skipping)

                if len(batch_images) >= batch_size:
                    with torch.no_grad():
                        batch_tensor = torch.stack(batch_images, dim=0).to(device)
                        emb_batch = backbone(batch_tensor)
                    emb_np_batch = emb_batch.cpu().numpy().astype(np.float32)

                    for j in range(len(batch_images)):
                        if use_keypoints:
                            full_vector = np.concatenate(
                                [emb_np_batch[j], np.array(batch_bbox_feats[j], dtype=np.float32), np.array(batch_kp_feats[j], dtype=np.float32)],
                                axis=0,
                            )
                        else:
                            full_vector = np.concatenate(
                                [emb_np_batch[j], np.array(batch_bbox_feats[j], dtype=np.float32)],
                                axis=0,
                            )
                        feats_list.append(full_vector)
                        labels_list.append(batch_labels[j])
                        frames_list.append(batch_frames[j])

                    batch_images.clear()
                    batch_bbox_feats.clear()
                    if use_keypoints:
                        batch_kp_feats.clear()
                    batch_labels.clear()
                    batch_frames.clear()

            # Process remaining batch
            if batch_images:
                with torch.no_grad():
                    batch_tensor = torch.stack(batch_images, dim=0).to(device)
                    emb_batch = backbone(batch_tensor)
                emb_np_batch = emb_batch.cpu().numpy().astype(np.float32)

                for j in range(len(batch_images)):
                    if use_keypoints:
                        full_vector = np.concatenate(
                            [emb_np_batch[j], np.array(batch_bbox_feats[j], dtype=np.float32), np.array(batch_kp_feats[j], dtype=np.float32)],
                            axis=0,
                        )
                    else:
                        full_vector = np.concatenate(
                            [emb_np_batch[j], np.array(batch_bbox_feats[j], dtype=np.float32)],
                            axis=0,
                        )
                    feats_list.append(full_vector)
                    labels_list.append(batch_labels[j])
                    frames_list.append(batch_frames[j])

            if not feats_list:
                continue

            feats_array = np.stack(feats_list, axis=0)
            labels_array = np.array(labels_list, dtype=np.int32)
            frames_array = np.array(frames_list, dtype=np.int32)

            np.savez(save_path, features=feats_array, labels=labels_array, frames=frames_array)

            total_tracks += 1
            if idx % 5 == 0 or idx == len(track_ids):
                print(f"    {video_name}: processed {idx}/{len(track_ids)} tracks")

    print(f"\n>>> FINISHED. Features saved in: {dst_dir}")


if __name__ == "__main__":
    # Load main config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    action_ids = config.get("behavior_classes", {"Lying": 0})
    frames_root = config.get("frames_folder", "data/images/frames")
    padding_factor = config.get("bbox_padding_factor", 1.10)

    use_keypoints = config.get("use_keypoints", False)
    SRC = "data/annotations/behavior"
    DST = "data/features_kp" if use_keypoints else "data/features"

    extract_features(
        SRC,
        DST,
        frames_root=frames_root,
        action_to_id=action_ids,
        cnn_name="resnet18",
        image_size=224,
        bbox_padding_factor=padding_factor,
        batch_size=16,
        use_keypoints=use_keypoints,
    )

