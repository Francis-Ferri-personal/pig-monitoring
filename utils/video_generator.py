import os
import cv2
import yaml
import argparse
from tqdm import tqdm
import json
import re

# Import the visualization function
from viz_utils import visualize_coco_frame

def process_single_clip(video_id, clip_id, ann_dir, frames_root, config, args):
    """Processes a single clip and generates a video."""
    video_dir = f"video{video_id}"
    json_path = os.path.join(ann_dir, video_dir, f"{clip_id}.json")
    
    if not os.path.exists(json_path):
        print(f"  ! Error: Annotation file not found at {json_path}")
        return

    # Determine output path and check for resume
    out_base = config.get('video_out_folder', "out/videos")
    os.makedirs(out_base, exist_ok=True)
    type_str = "pose" if args.pose else "sam"
    
    if args.output and not args.all:
        video_out_path = args.output
    else:
        video_out_path = os.path.join(out_base, f"video{video_id}_{clip_id}_{type_str}.mp4")

    if os.path.exists(video_out_path):
        print(f"  >>> Skipping: {video_out_path} (Already exists)")
        return

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Sort images by frame_id to ensure chronological order
    images = sorted(coco_data.get('images', []), key=lambda x: x['frame_id'])
    
    if not images:
        print(f"  ! Error: No images found in {json_path}")
        return

    print(f"\n>>> Generating video for {video_dir}/{clip_id}")
    print(f"  - Total annotated frames: {len(images)}")
    print(f"  - Target FPS: {args.fps}")

    video_writer = None
    
    for img_entry in tqdm(images, desc=f"Clip {clip_id}", leave=False):
        frame_idx = img_entry['frame_id']
        
        # Call viz_utils to get the rendered image (as a BGR numpy array)
        vis_frame = visualize_coco_frame(
            video_id=video_id,
            clip_id=clip_id,
            frame_id=frame_idx,
            annotations_dir=ann_dir,
            frames_root=frames_root,
            show_pose=args.pose
        )
        
        if vis_frame is None:
            continue
            
        h, w = vis_frame.shape[:2]
        
        # Initialize video writer
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_out_path, fourcc, args.fps, (w, h))
        
        video_writer.write(vis_frame)

    if video_writer:
        video_writer.release()
        print(f"  ✓ Video saved: {video_out_path}")
    else:
        print(f"  ! No frames processed for {clip_id}")

def main():
    parser = argparse.ArgumentParser(description="Generate videos from COCO annotations.")
    parser.add_argument("--video", type=int, default=None, help="Video ID (e.g., 1 for video1)")
    parser.add_argument("--clip", type=str, default=None, help="Clip ID (e.g., '01')")
    parser.add_argument("--all", action="store_true", help="Process all available annotations")
    parser.add_argument("--pose", action="store_true", help="Use Pose annotations instead of SAM")
    parser.add_argument("--output", type=str, default=None, help="Output video path (for single clip)")
    parser.add_argument("--fps", type=int, default=1, help="Output video FPS.")
    
    args = parser.parse_args()

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    ann_dir = "data/annotations/pose" if args.pose else "data/annotations/sam"
    frames_root = config.get('frames_folder', "data/images/frames")

    if args.all:
        # Scan ann_dir for all videoX folders and JSON clips
        if not os.path.exists(ann_dir):
            print(f"Error: Annotation directory {ann_dir} does not exist.")
            return

        video_folders = sorted([d for d in os.listdir(ann_dir) if os.path.isdir(os.path.join(ann_dir, d))])
        
        for v_folder in video_folders:
            # Extract video digits
            match = re.search(r'video(\d+)', v_folder)
            if not match: continue
            v_id = int(match.group(1))
            
            v_path = os.path.join(ann_dir, v_folder)
            clip_files = sorted([f for f in os.listdir(v_path) if f.endswith('.json')])
            
            for c_file in clip_files:
                c_id = c_file.replace('.json', '')
                process_single_clip(v_id, c_id, ann_dir, frames_root, config, args)
    else:
        # Single clip mode
        if args.video is None or args.clip is None:
            print("Error: You must provide --video and --clip OR use --all")
            return
        process_single_clip(args.video, args.clip, ann_dir, frames_root, config, args)

    print("\n>>> FINISHED.")

if __name__ == "__main__":
    main()
