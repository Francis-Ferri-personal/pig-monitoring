import os
import json
import numpy as np
from pycocotools import mask as mask_utils

def new_coco_data(video_id, video_name, category_name="pig", super_category="animal"):
    """
    Creates an empty COCO dataset skeleton for a single video clip.

    Use alongside `append_frame_to_coco` for memory-friendly incremental
    conversion: each frame is converted to compact RLE and dropped from RAM
    immediately instead of buffering all raw masks for the whole clip.
    """
    return {
        "videos": [
            {
                "id": video_id,
                "file_name": video_name
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": category_name,
                "supercategory": super_category,
            }
        ]
    }

def append_frame_to_coco(coco_data, outputs, frame_idx, img_id, ann_id, prev_image_id=-1, next_image_id=-1):
    """
    Appends one SAM 3 frame's outputs to an existing COCO dict and returns the
    next annotation ID.

    The raw `out_binary_masks` (N, H, W) bool array is consumed here and
    converted to COCO Compressed RLE strings, so callers can drop the original
    arrays right after this call to keep peak host memory low.
    """
    masks = outputs.get("out_binary_masks", [])
    if len(masks) > 0:
        h, w = masks.shape[1], masks.shape[2]
    else:
        h, w = 0, 0

    coco_data["images"].append({
        "id": img_id,
        "video_id": coco_data["videos"][0]["id"],
        "frame_id": frame_idx,
        "prev_image_id": prev_image_id,
        "next_image_id": next_image_id,
        "height": int(h),
        "width": int(w)
    })

    obj_ids = outputs.get("out_obj_ids", [])
    probs = outputs.get("out_probs", [])
    boxes = outputs.get("out_boxes_xywh", [])  # [x, y, w, h] normalized

    for obj_idx in range(len(obj_ids)):
        mask = masks[obj_idx]

        # 1. Encode mask to RLE string (COCO compressed RLE, column-major)
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')  # byte to string

        # 2. Calculate area
        area = float(mask_utils.area(rle))

        # 3. Convert bbox to absolute pixel coordinates (using integers)
        box = boxes[obj_idx]
        abs_box = [
            int(round(box[0] * w)),
            int(round(box[1] * h)),
            int(round(box[2] * w)),
            int(round(box[3] * h))
        ]

        ann_id += 1
        coco_data["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "track_id": int(obj_ids[obj_idx]),
            "bbox": abs_box,
            "segmentation": rle,
            "area": area,
            "conf": float(probs[obj_idx]),
            "iscrowd": 0
        })

    return ann_id

def sam_to_coco(outputs_per_frame, video_id, video_name, frame_ids, category_name="pig", super_category="animal", global_img_id_offset=0, global_ann_id_offset=0):
    """
    Converts SAM 3 video predictor outputs to COCO format.
    
    Args:
        outputs_per_frame: Dict mapping frame_idx to dictionary of outputs (out_obj_ids, out_probs, out_boxes_xywh, out_binary_masks).
        video_id: Integer ID for the video.
        video_name: Logical name for the video.
        frame_ids: List of frame IDs (e.g., "00000", "00001") indexed by frame_idx, used for the COCO file_name.
        category_name: Name of the category (e.g., "pig").
        global_img_id_offset: For multi-clip datasets, offset for image IDs.
        global_ann_id_offset: For multi-clip datasets, offset for annotation IDs.
        
    Returns:
        coco_data: Dictionary containing the COCO data.
        next_ann_id: The ID to use for the next annotation in a sequence.

    Note on segmentation 'counts':
        This is COCO's Compressed RLE (Run-Length Encoding) format. 
        It represents the mask as a series of lengths of alternating 
        0 (background) and 1 (foreground) pixel runs in column-major order.
        It's much more memory-efficient than a list of polygons.

    For long clips, prefer the incremental path: `new_coco_data` +
    `append_frame_to_coco` per frame (releases raw masks immediately).
    """
    coco_data = new_coco_data(video_id, video_name, category_name, super_category)
    
    ann_id = global_ann_id_offset
    sorted_frame_indices = sorted(outputs_per_frame.keys())
    
    for i, frame_idx in enumerate(sorted_frame_indices):
        outputs = outputs_per_frame[frame_idx]
        img_id = global_img_id_offset + frame_idx

        prev_image_id = (global_img_id_offset + sorted_frame_indices[i-1]) if i > 0 else -1
        next_image_id = (global_img_id_offset + sorted_frame_indices[i+1]) if i < len(sorted_frame_indices)-1 else -1

        ann_id = append_frame_to_coco(
            coco_data, outputs, frame_idx, img_id, ann_id,
            prev_image_id=prev_image_id, next_image_id=next_image_id,
        )
            
    return coco_data, ann_id

def save_coco_to_json(coco_data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
