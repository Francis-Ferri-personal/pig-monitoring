import os
import json
import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

def visualize_coco_frame(video_id, clip_id, frame_id, annotations_dir="data/annotations/sam", frames_root="data/images/frames", figsize=(12, 8), output_path=None, show_pose=False):
    """
    Visualizes a specific frame from a clip using its COCO annotations. 
    Optimized version using OpenCV for significantly faster rendering.
    """
    video_dir = f"video{video_id}"
    json_path = os.path.join(annotations_dir, video_dir, f"{clip_id}.json")
    if not os.path.exists(json_path):
        print(f"Error: Annotation file not found at {json_path}")
        return None

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # 1. Find the image entry
    img_entry = None
    for img in coco_data.get('images', []):
        if img.get('frame_id') == frame_id:
            img_entry = img
            break
    
    if not img_entry:
        return None

    # 2. Load the image
    potential_paths = [
        os.path.join(frames_root, video_dir, img_entry['file_name']),
        os.path.join(frames_root, video_dir, os.path.basename(clip_id), os.path.basename(img_entry['file_name'])),
        os.path.join("data/images/frames_masked", os.path.basename(clip_id), os.path.basename(img_entry['file_name'])),
    ]
    
    actual_path = None
    for p in potential_paths:
        if os.path.exists(p):
            actual_path = p
            break
            
    if not actual_path:
        return None

    # Load with OpenCV directly (BGR)
    image = cv2.imread(actual_path)
    if image is None:
        return None
        
    h, w = image.shape[:2]
    vis_image = image.copy()

    # 3. Get annotations and Categories
    img_id = img_entry['id']
    anns = [ann for ann in coco_data.get('annotations', []) if ann['image_id'] == img_id]
    categories = {cat['id']: cat for cat in coco_data.get('categories', [])}

    # Consistent colors for tracks
    np.random.seed(42)
    colors = (np.random.random((100, 3)) * 255).astype(np.uint8)

    for ann in anns:
        track_id = ann.get('track_id', 0)
        # Convert to BGR for OpenCV
        color = colors[track_id % 100].tolist() # [B, G, R]
        
        # Draw Mask
        seg = ann.get('segmentation')
        if seg and isinstance(seg, dict) and 'counts' in seg:
            mask = mask_utils.decode(seg)
            # Create a colored mask
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask > 0] = color
            # Alpha blending for the mask
            idx = mask > 0
            vis_image[idx] = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)[idx]
        
        # Draw BBox
        bbox = ann.get('bbox') # [x, y, w, h]
        if bbox:
            x, y, bw, bh = map(int, bbox)
            cv2.rectangle(vis_image, (x, y), (x + bw, y + bh), color, 2)
            
            # Label
            label = f"ID: {track_id}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_image, (x, y - th - 10), (x + tw + 10, y), color, -1)
            cv2.putText(vis_image, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw Keypoints and Skeleton if requested and available
        if show_pose and 'keypoints' in ann and ann['keypoints']:
            kpts = np.array(ann['keypoints']).reshape(-1, 3) # [x, y, v]
            
            # Draw Skeleton
            cat = categories.get(ann['category_id'], {})
            skeleton = cat.get('skeleton', [])
            for pair in skeleton:
                p1_idx, p2_idx = pair[0] - 1, pair[1] - 1
                if p1_idx < len(kpts) and p2_idx < len(kpts):
                    p1 = kpts[p1_idx]
                    p2 = kpts[p2_idx]
                    if p1[2] > 0 and p2[2] > 0: # Check visibility
                        cv2.line(vis_image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)
            
            # Draw Joints
            for kp in kpts:
                if kp[2] > 1: # Visible
                    cv2.circle(vis_image, (int(kp[0]), int(kp[1])), 4, (0, 255, 255), -1) # Yellow
                    cv2.circle(vis_image, (int(kp[0]), int(kp[1])), 4, (0, 0, 0), 1)
                elif kp[2] == 1: # Labeled but obscured
                    cv2.circle(vis_image, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1) # Red

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to: {output_path}")

    return vis_image

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize COCO annotations (SAM or Pose) for a specific video frame.")
    parser.add_argument("--video", type=int, default=1, help="Video ID (default: 1)")
    parser.add_argument("--clip", type=str, required=True, help="Clip ID (e.g., '1' or '01')")
    parser.add_argument("--frame", type=int, required=True, help="Frame index")
    parser.add_argument("--pose", action="store_true", help="Visualize Pose annotations (uses data/annotations/pose)")
    parser.add_argument("--ann_dir", type=str, default=None, help="Custom path to annotations folder")
    parser.add_argument("--frame_root", type=str, default="data/images/frames", help="Path to frames root folder")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the output image")
    
    args = parser.parse_args()
    
    # Logic to select the correct annotations directory
    if args.ann_dir:
        final_ann_dir = args.ann_dir
    elif args.pose:
        final_ann_dir = "data/annotations/pose"
    else:
        final_ann_dir = "data/annotations/sam"
        
    # Format clip_id to match file naming (e.g., '1' -> '01')
    clip_id = args.clip
    if clip_id.isdigit():
        clip_id = f"{int(clip_id):02d}"
        
    res = visualize_coco_frame(
        video_id=args.video,
        clip_id=clip_id, 
        frame_id=args.frame, 
        annotations_dir=final_ann_dir, 
        frames_root=args.frame_root,
        output_path=args.output,
        show_pose=args.pose
    )
    
    if res is not None and not args.output:
        cv2.imshow("Frame Visualization", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
