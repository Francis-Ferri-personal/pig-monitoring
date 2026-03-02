import os
import cv2
import yaml
import argparse
from tqdm import tqdm
import json
import re

# Import the visualization function
from viz_utils import visualize_coco_frame

def process_single_clip(video_id, clip_id, ann_dir, frames_root, config, args, mode):
    """Processes a single clip and generates a video (raw or annotated)."""
    video_dir = f"video{video_id}"
    json_path = os.path.join(ann_dir, video_dir, f"{clip_id}.json")
    
    if not os.path.exists(json_path):
        print(f"  ! Error: Annotation file not found at {json_path}")
        return

    # Determine output path and check for resume
    out_base = config.get('video_out_folder', "out/videos")
    final_out_dir = os.path.join(out_base, video_dir)
    os.makedirs(final_out_dir, exist_ok=True)
    
    if args.output and not args.all:
        video_out_path = args.output
    else:
        video_out_path = os.path.join(final_out_dir, f"{clip_id}.mp4")

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

    print(f"\n>>> Generating {mode} video for {video_dir}/{clip_id}")
    video_writer = None
    
    for img_entry in tqdm(images, desc=f"Clip {clip_id}", leave=False):
        frame_idx = img_entry['frame_id']
        
        if mode == "raw":
            # Just load the original frame directly (fastest)
            # viz_utils logic for finding actual path (check multiple locations)
            potential_paths = [
                os.path.join(frames_root, video_dir, img_entry['file_name']),
                os.path.join(frames_root, video_dir, os.path.basename(clip_id), os.path.basename(img_entry['file_name'])),
                os.path.join("data/images/frames_masked", os.path.basename(clip_id), os.path.basename(img_entry['file_name'])),
            ]
            vis_frame = None
            for p in potential_paths:
                if os.path.exists(p):
                    vis_frame = cv2.imread(p)
                    break
        else:
            # Call viz_utils to render annotations (SAM or Pose)
            vis_frame = visualize_coco_frame(
                video_id=video_id,
                clip_id=clip_id,
                frame_id=frame_idx,
                annotations_dir=ann_dir,
                frames_root=frames_root,
                show_pose=(mode == "pose")
            )
        
        if vis_frame is None:
            continue
            
        h, w = vis_frame.shape[:2]
        
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
    parser = argparse.ArgumentParser(description="Generate videos from frames (raw or annotated).")
    parser.add_argument("--video", type=int, default=None, help="Video ID (e.g., 1 for video1)")
    parser.add_argument("--clip", type=str, default=None, help="Clip ID (e.g., '01')")
    parser.add_argument("--all", action="store_true", help="Process all available clips")
    parser.add_argument("--pose", action="store_true", help="Use Pose annotations (skeleton)")
    parser.add_argument("--sam", action="store_true", help="Use SAM annotations (masks)")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--fps", type=int, default=1, help="Output video FPS.")
    
    args = parser.parse_args()

    # Determine mode and directories
    if args.pose:
        mode = "pose"
        ann_dir = "data/annotations/pose"
        default_out = "out/videos/pose"
    elif args.sam:
        mode = "sam"
        ann_dir = "data/annotations/sam"
        default_out = "out/videos/sam"
    else:
        # Default mode is now raw clips (no annotations)
        mode = "raw"
        ann_dir = "data/annotations/sam" # We still need any JSON to get the frame list for the clip
        default_out = "out/videos/clip"

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override output folder from config if present, otherwise use logic above
    config['video_out_folder'] = default_out
    frames_root = config.get('frames_folder', "data/images/frames")

    if args.all:
        if not os.path.exists(ann_dir):
            print(f"Error: Annotation directory {ann_dir} does not exist.")
            return

        video_folders = sorted([d for d in os.listdir(ann_dir) if os.path.isdir(os.path.join(ann_dir, d))])
        
        for v_folder in video_folders:
            match = re.search(r'video(\d+)', v_folder)
            if not match: continue
            v_id = int(match.group(1))
            
            v_path = os.path.join(ann_dir, v_folder)
            clip_files = sorted([f for f in os.listdir(v_path) if f.endswith('.json')])
            
            for c_file in clip_files:
                c_id = c_file.replace('.json', '')
                process_single_clip(v_id, c_id, ann_dir, frames_root, config, args, mode)
    else:
        if args.video is None or args.clip is None:
            print("Error: Provide --video and --clip OR use --all")
            return
        process_single_clip(args.video, args.clip, ann_dir, frames_root, config, args, mode)

    print("\n>>> FINISHED.")

if __name__ == "__main__":
    main()
