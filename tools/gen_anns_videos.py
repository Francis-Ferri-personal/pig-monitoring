import os
import sys
import glob
import json
import torch
import re
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# Ensure imports work regardless of where the script is called from
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.append(project_root)

# Add SAM 3 repo to path for internal package resolution
sam3_repo_path = os.path.join(project_root, 'sam3')
if sam3_repo_path not in sys.path:
    sys.path.insert(0, sam3_repo_path)

from sam3.model_builder import build_sam3_video_predictor
from utils.coco_utils import sam_to_coco, save_coco_to_json

def natural_key(name):
    """Sort key that orders embedded numbers numerically: video1, video2, video10."""
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]

def load_mask(mask_path):
    """Load a static mask as a binary (0/255) grayscale image."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask at {mask_path}")
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

def read_masked_frames(video_path, mask):
    """
    Reads all frames from a clip video and applies the static mask in memory
    (
    Returns the list of masked PIL frames and their frame IDs (0, 1, 2, ...).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! Warning: Could not open video {video_path}")
        return [], []

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Resize mask if necessary
    if mask.shape[:2] != (height, width):
        mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask

    # Convert grayscale mask to 3-channel
    if len(mask_resized.shape) == 2:
        mask_3d = cv2.merge([mask_resized, mask_resized, mask_resized])
    else:
        mask_3d = mask_resized

    pil_frames = []
    pbar = tqdm(total=total_frames, desc=f"  Reading frames {os.path.basename(os.path.dirname(video_path))}/{os.path.basename(video_path)}", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        masked_frame = cv2.bitwise_and(frame, mask_3d)
        frame_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(frame_rgb))
        pbar.update(1)

    pbar.close()
    cap.release()

    frame_ids = [str(i) for i in range(len(pil_frames))]
    return pil_frames, frame_ids

def generate_annotations(prompt_text="pig", video_filter=None, mask_path="data/images/mask.png", clip_filter=None, gpus=None):
    """
    Processes all videos and their clips found in data/videos/clips, applying the
    static mask in memory (no need for pre-masked videos or extracted frames).
    """
    # 1. Load mask once
    print(f">>> Loading mask from {mask_path}...")
    mask = load_mask(mask_path)

    # 2. Initialize predictor once (default: single GPU to avoid multi-GPU worker noise/hangs)
    print(">>> Initializing SAM 3 Predictor...")
    if gpus is None:
        gpus = [torch.cuda.current_device()]
    predictor = build_sam3_video_predictor(gpus_to_use=gpus)
    print(f">>> Using GPU(s): {gpus}")

    try:
        _process_videos(predictor, mask, prompt_text=prompt_text,
                        video_filter=video_filter, clip_filter=clip_filter)
    except KeyboardInterrupt:
        print("\n>>> Interrupted by user. Shutting down predictor cleanly...")
    finally:
        predictor.shutdown()
        print(">>> SAM 3 predictor shutdown complete.\n")

def _process_videos(predictor, mask, prompt_text="pig", video_filter=None, clip_filter=None):
    """Run the annotation pipeline across videos/clips using the given predictor."""
    clips_base_root = "data/videos/clips"
    if not os.path.exists(clips_base_root):
        print(f"Error: Base directory {clips_base_root} not found.")
        return

    # 3. Find all video directories
    all_video_dirs = sorted([d for d in os.listdir(clips_base_root) 
                            if os.path.isdir(os.path.join(clips_base_root, d))], key=natural_key)
    
    if video_filter:
        video_dirs = [d for d in all_video_dirs if d == video_filter]
    else:
        if clip_filter:
            print("Error: --clip requires --video to be specified.")
            return
        video_dirs = all_video_dirs
    
    if not video_dirs:
        print("No video directories found matching the filter in data/videos/clips.")
        return

    print(f">>> Found {len(video_dirs)} videos: {', '.join(video_dirs)}")

    # 4. Process each video (video_id is auto-incremental, starting at 0)
    video_counter = 0
    for video_dir_name in video_dirs:
        video_id = video_counter
        video_counter += 1

        clips_root = os.path.join(clips_base_root, video_dir_name)
        output_base_dir = os.path.join("data/annotations/sam", video_dir_name)
        os.makedirs(output_base_dir, exist_ok=True)

        # Get and sort clips (01, 02, etc.)
        clips = sorted([os.path.splitext(f)[0] for f in os.listdir(clips_root) 
                       if f.lower().endswith(".mp4")], key=natural_key)

        # Filter to a single clip if requested
        if clip_filter:
            if clip_filter in clips:
                clips = [clip_filter]
            else:
                print(f"  ! Clip {clip_filter} not found in video {video_dir_name}. "
                      f"Available clips: {', '.join(clips)}")
                continue
        
        # Count clips already generated (for resume progress)
        clips_done = sum(1 for c in clips
                        if os.path.exists(os.path.join(output_base_dir, f"{c}.json")))
        
        # Reset counters for each NEW video
        global_img_id_offset = 0
        global_ann_id_offset = 0

        print(f"\n" + "="*50)
        print(f" STARTED VIDEO: {video_dir_name} (ID: {video_id})")
        print(f" Clips already done: {clips_done}/{len(clips)} (pending: {len(clips)-clips_done})")
        print("="*50 + "\n")
        
        for i, clip_id in enumerate(clips):
            output_json = os.path.join(output_base_dir, f"{clip_id}.json")
            
            # --- RESUME LOGIC ---
            if os.path.exists(output_json):
                try:
                    with open(output_json, 'r') as f:
                        existing_data = json.load(f)
                    
                    num_frames = len(existing_data.get('images', []))
                    existing_anns = existing_data.get('annotations', [])
                    max_ann_ptr = global_ann_id_offset
                    if existing_anns:
                        max_ann_ptr = max(ann['id'] for ann in existing_anns)
                    
                    # Update offsets as if we had processed it
                    global_img_id_offset += num_frames
                    global_ann_id_offset = max_ann_ptr
                    
                    print(f"[{i+1}/{len(clips)}] Skipping Clip: {clip_id} (Already exists. Resuming offsets...)")
                    continue
                except Exception as e:
                    print(f"[{i+1}/{len(clips)}] Error reading existing {output_json}: {e}. Will re-process.")

            # --- STANDARD PROCESSING ---
            print(f"[{i+1}/{len(clips)}] Processing Clip: {clip_id} ...", flush=True)
            
            clip_video_path = os.path.join(clips_root, f"{clip_id}.mp4")
            video_frames_masked, frame_ids = read_masked_frames(clip_video_path, mask)
            
            if not video_frames_masked:
                print(f"  ! Warning: Could not read frames from {clip_id}. Skipping.")
                continue

            # SAM 3 session logic (pass frames in memory, no temp files)
            session_id = None
            try:
                print(f"  Starting SAM 3 session with {len(frame_ids)} frames...", flush=True)
                response = predictor.handle_request(dict(type="start_session", resource_path=video_frames_masked))
                session_id = response["session_id"]
                predictor.handle_request(dict(type="reset_session", session_id=session_id))

                # Add prompt on frame 0
                predictor.handle_request(dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=prompt_text
                ))

                # Propagate and collect outputs (yields once per frame; "both" direction = ~2x frames)
                print(f"  Propagating prompt across frames...", flush=True)
                outputs_per_frame = {}
                pbar = tqdm(total=2 * len(frame_ids),
                            desc=f"  SAM3 {video_dir_name}/{clip_id}",
                            unit="frame")
                for resp in predictor.handle_stream_request(dict(
                    type="propagate_in_video",
                    session_id=session_id
                )):
                    outputs_per_frame[resp["frame_index"]] = resp["outputs"]
                    pbar.update(1)
                pbar.close()

                # Convert to COCO
                coco_data, last_ann_id = sam_to_coco(
                    outputs_per_frame=outputs_per_frame,
                    video_id=video_id,
                    video_name=clip_id,
                    frame_ids=frame_ids,
                    global_img_id_offset=global_img_id_offset,
                    global_ann_id_offset=global_ann_id_offset
                )

                # Save JSON
                save_coco_to_json(coco_data, output_json)

                # Update offsets
                global_img_id_offset += len(frame_ids)
                global_ann_id_offset = last_ann_id
                
                print(f"  ✓ Clip {clip_id} Done. ({len(frame_ids)} frames. Total frames: {global_img_id_offset})\n")

            except Exception as e:
                print(f"  ✗ Error processing clip {clip_id}: {e}")
            
            finally:
                if session_id is not None:
                    # Close the inference session to free its GPU resources
                    predictor.handle_request(dict(type="close_session", session_id=session_id))

        print(f"\n>>> COMPLETED Video {video_dir_name}. Final Annotation ID: {global_ann_id_offset}")

    print("\n>>> ALL VIDEOS PROCESSED SUCCESSFULLY.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate COCO annotations for all available videos and clips.")
    parser.add_argument("--prompt", type=str, default="pig", help="Text prompt for SAM 3 (default: 'pig')")
    parser.add_argument("--video", type=str, default=None, help="Process only the specified video folder name, e.g., May_25_01.")
    parser.add_argument("--clip", type=str, default=None, help="Process only the specified clip number, e.g., 01 (requires --video).")
    parser.add_argument("--mask", type=str, default="data/images/mask.png", help="Path to mask PNG (default: data/images/mask.png)")
    parser.add_argument("--gpus", type=int, nargs="+", default=None, help="GPU ID(s) to use (default: single GPU)")
    
    args = parser.parse_args()
    generate_annotations(prompt_text=args.prompt, video_filter=args.video, mask_path=args.mask, clip_filter=args.clip, gpus=args.gpus)