import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as mask_utils

def visualize_coco_frame(video_id, clip_id, frame_id, annotations_dir="data/annotations/sam", frames_root="data/images/frames", figsize=(12, 8), output_path=None):
    """
    Visualizes a specific frame from a clip using its COCO annotations.
    
    Args:
        video_id: Integer ID for the video (e.g. 1).
        clip_id: The ID/name of the clip (e.g., "01").
        frame_id: The index of the frame within the video.
        annotations_dir: Path to the directory containing COCO JSON files.
        frames_root: Root directory where frames are stored (default: data/images/frames).
        figsize: Size of the matplotlib figure.
        output_path: If provided, save the visualization to this path.
    """
    video_dir = f"video{video_id}"
    json_path = os.path.join(annotations_dir, video_dir, f"{clip_id}.json")
    if not os.path.exists(json_path):
        print(f"Error: Annotation file not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # 1. Find the image entry
    img_entry = None
    for img in coco_data['images']:
        if img['frame_id'] == frame_id:
            img_entry = img
            break
    
    if not img_entry:
        print(f"Error: Frame {frame_id} not found in {json_path}")
        return

    # 2. Load the image
    # file_name in JSON is usually "01/00028.png"
    # The structure in frames_root (e.g. data/images/frames) is videoX/01/00028.png
    
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
        print(f"Error: Image not found for {img_entry['file_name']}. Checked: {potential_paths}")
        return

    image = np.array(Image.open(actual_path))
    h, w = image.shape[:2]

    # 3. Get annotations
    img_id = img_entry['id']
    anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

    # 4. Visualize
    plt.figure(figsize=figsize)
    plt.imshow(image)
    ax = plt.gca()

    # Colors for different tracks
    np.random.seed(42)
    colors = np.random.random((100, 3)) # Up to 100 tracks

    for ann in anns:
        track_id = ann.get('track_id', 0)
        color = colors[track_id % 100]
        
        # Draw Mask
        seg = ann['segmentation']
        if isinstance(seg, dict) and 'counts' in seg:
            # Decode RLE
            mask = mask_utils.decode(seg)
            
            # Create semi-transparent overlay
            mask_overlay = np.zeros((*mask.shape, 4))
            mask_overlay[mask > 0] = [*color, 0.4]
            ax.imshow(mask_overlay)
        
        # Draw BBox
        bbox = ann['bbox'] # [x, y, w, h]
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                             fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        # Label
        ax.text(bbox[0], bbox[1] - 5, f"ID: {track_id}", 
                color='white', fontsize=12, weight='bold',
                bbox=dict(facecolor=color, alpha=0.8, edgecolor='none'))

    plt.title(f"Video: {video_id} | Clip: {clip_id} | Frame Index: {frame_id} | File: {os.path.basename(actual_path)}")
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path)
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Example usage:
    # python utils/viz_utils.py --video 1 --clip 1 --frame 0 --output output.png
    import argparse
    parser = argparse.ArgumentParser(description="Visualize COCO annotations for a specific video frame.")
    parser.add_argument("--video", type=int, default=1, help="Video ID (default: 1)")
    parser.add_argument("--clip", type=str, required=True, help="Clip ID (e.g., '1' or '01')")
    parser.add_argument("--frame", type=int, required=True, help="Frame index")
    parser.add_argument("--ann_dir", type=str, default="data/annotations/sam", help="Path to annotations folder")
    parser.add_argument("--frame_root", type=str, default="data/images/frames", help="Path to frames root folder")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the output image")
    
    args = parser.parse_args()
    
    # Format clip_id to match file naming (e.g., '1' -> '01')
    clip_id = args.clip
    if clip_id.isdigit():
        clip_id = f"{int(clip_id):02d}"
        
    visualize_coco_frame(
        video_id=args.video,
        clip_id=clip_id, 
        frame_id=args.frame, 
        annotations_dir=args.ann_dir, 
        frames_root=args.frame_root,
        output_path=args.output
    )
