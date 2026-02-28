import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as mask_utils

def visualize_coco_frame(video_id, clip_id, frame_id, annotations_dir="data/annotations/sam", frames_root="data/images/frames", figsize=(12, 8), output_path=None, show_pose=False):
    """
    Visualizes a specific frame from a clip using its COCO annotations. 
    Can optionally show keypoints if they exist in the annotation.
    
    Args:
        video_id: Integer ID for the video (e.g. 1).
        clip_id: The ID/name of the clip (e.g., "01").
        frame_id: The index of the frame within the video.
        annotations_dir: Path to the directory containing COCO JSON files.
        frames_root: Root directory where frames are stored (default: data/images/frames).
        figsize: Size of the matplotlib figure.
        output_path: If provided, save the visualization to this path.
        show_pose: If True, attempt to draw keypoints/skeleton from the annotations.
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
    for img in coco_data.get('images', []):
        if img.get('frame_id') == frame_id:
            img_entry = img
            break
    
    if not img_entry:
        print(f"Error: Frame {frame_id} not found in {json_path}")
        return

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
        print(f"Error: Image not found for {img_entry['file_name']}. Checked: {potential_paths}")
        return

    image = np.array(Image.open(actual_path))
    h, w = image.shape[:2]

    # 3. Get annotations
    img_id = img_entry['id']
    anns = [ann for ann in coco_data.get('annotations', []) if ann['image_id'] == img_id]

    # 4. Get Skeleton/Categories info for pose
    categories = {cat['id']: cat for cat in coco_data.get('categories', [])}

    # 5. Visualize
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
        seg = ann.get('segmentation')
        if seg and isinstance(seg, dict) and 'counts' in seg:
            mask = mask_utils.decode(seg)
            mask_overlay = np.zeros((*mask.shape, 4))
            mask_overlay[mask > 0] = [*color, 0.4]
            ax.imshow(mask_overlay)
        
        # Draw BBox
        bbox = ann.get('bbox') # [x, y, w, h]
        if bbox:
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                 fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            
            # Label
            ax.text(bbox[0], bbox[1] - 5, f"ID: {track_id}", 
                    color='white', fontsize=12, weight='bold',
                    bbox=dict(facecolor=color, alpha=0.8, edgecolor='none'))
        
        # Draw Keypoints and Skeleton if requested and available
        if show_pose and 'keypoints' in ann and ann['keypoints']:
            kpts = np.array(ann['keypoints']).reshape(-1, 3) # [x, y, v]
            
            # Draw Skeleton
            cat = categories.get(ann['category_id'], {})
            skeleton = cat.get('skeleton', [])
            for pair in skeleton:
                p1_idx, p2_idx = pair[0] - 1, pair[1] - 1 # 1-indexed to 0-indexed
                if p1_idx < len(kpts) and p2_idx < len(kpts):
                    p1 = kpts[p1_idx]
                    p2 = kpts[p2_idx]
                    if p1[2] > 0 and p2[2] > 0: # Check visibility
                        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2, alpha=0.8)
            
            # Draw Joints
            for kp in kpts:
                if kp[2] > 1: # Visible
                    plt.scatter(kp[0], kp[1], color='yellow', s=20, edgecolors='black', zorder=10)
                elif kp[2] == 1: # Labeled but obscured
                    plt.scatter(kp[0], kp[1], color='red', s=10, alpha=0.5, zorder=10)

    plt.title(f"Video: {video_id} | Clip: {clip_id} | Frame Index: {frame_id} | Pose: {'ON' if show_pose else 'OFF'}")
    plt.axis('off')
    plt.tight_layout()
    
    # NEW: Capture the plot into a numpy array for video generation
    fig = plt.gcf()
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    # We use buffer_rgba() which is available in Agg backend
    vis_image = np.array(fig.canvas.buffer_rgba())
    # Convert RGBA to RGB (drop alpha) and ignore matplotlib's default color channel order (already RGB)
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGBA2BGR)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path)
        print(f"Visualization saved to: {output_path}")
    
    # If not saving or showing, we still want the image returned
    if not output_path and not plt.get_fignums():
        # This case is usually for batch processing/video gen
        pass
    else:
        if not output_path:
             plt.show()

    plt.close() # Always close to free memory
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
        
    visualize_coco_frame(
        video_id=args.video,
        clip_id=clip_id, 
        frame_id=args.frame, 
        annotations_dir=final_ann_dir, 
        frames_root=args.frame_root,
        output_path=args.output,
        show_pose=args.pose
    )
