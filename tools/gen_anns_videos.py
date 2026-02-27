import os
import sys
import glob
import json
import torch
import re
from tqdm import tqdm

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

def get_video_id(dir_name):
    """Extracts numeric ID from folder name like 'video1' -> 1"""
    match = re.search(r'video(\d+)', dir_name)
    return int(match.group(1)) if match else None

def generate_annotations(prompt_text="pig"):
    """
    Processes all videos and their clips found in data/images/frames_masked.
    """
    # 1. Initialize predictor once
    print(">>> Initializing SAM 3 Predictor...")
    gpus_to_use = range(torch.cuda.device_count())
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    masked_base_root = "data/images/frames_masked"
    if not os.path.exists(masked_base_root):
        print(f"Error: Base directory {masked_base_root} not found.")
        return

    # 2. Find all video directories
    video_dirs = sorted([d for d in os.listdir(masked_base_root) 
                        if os.path.isdir(os.path.join(masked_base_root, d)) and d.startswith("video")])
    
    if not video_dirs:
        print("No video directories found starting with 'video' in data/images/frames_masked.")
        return

    print(f">>> Found {len(video_dirs)} videos: {', '.join(video_dirs)}")

    # 3. Process each video
    for video_dir_name in video_dirs:
        video_id = get_video_id(video_dir_name)
        if video_id is None:
            continue

        masked_root = os.path.join(masked_base_root, video_dir_name)
        output_base_dir = os.path.join("data/annotations/sam", video_dir_name)
        os.makedirs(output_base_dir, exist_ok=True)

        # Get and sort clips (01, 02, etc.)
        clips = sorted([d for d in os.listdir(masked_root) if os.path.isdir(os.path.join(masked_root, d))])
        
        # Reset counters for each NEW video
        global_img_id_offset = 0
        global_ann_id_offset = 0

        print(f"\n" + "="*50)
        print(f" STARTED VIDEO: {video_dir_name} (ID: {video_id})")
        print(f" Total Clips to process: {len(clips)}")
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
            
            clip_masked_path = os.path.join(masked_root, clip_id)
            video_frames_masked = sorted(glob.glob(os.path.join(clip_masked_path, "*.png")), 
                                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
            
            if not video_frames_masked:
                print(f"  ! Warning: No frames found in {clip_id}. Skipping.")
                continue

            # SAM 3 session logic
            session_id = None
            try:
                response = predictor.handle_request(dict(type="start_session", resource_path=clip_masked_path))
                session_id = response["session_id"]
                predictor.handle_request(dict(type="reset_session", session_id=session_id))

                # Add prompt on frame 0
                predictor.handle_request(dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=prompt_text
                ))

                # Propagate and collect outputs
                outputs_per_frame = {}
                for resp in predictor.handle_stream_request(dict(
                    type="propagate_in_video",
                    session_id=session_id
                )):
                    outputs_per_frame[resp["frame_index"]] = resp["outputs"]

                # Convert to COCO
                coco_data, last_ann_id = sam_to_coco(
                    outputs_per_frame=outputs_per_frame,
                    video_id=video_id,
                    video_name=clip_id,
                    frame_paths=video_frames_masked,
                    global_img_id_offset=global_img_id_offset,
                    global_ann_id_offset=global_ann_id_offset
                )

                # Save JSON
                save_coco_to_json(coco_data, output_json)

                # Update offsets
                global_img_id_offset += len(video_frames_masked)
                global_ann_id_offset = last_ann_id
                
                print(f"  ✓ Clip {clip_id} Done. (Total frames: {global_img_id_offset})\n")

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
    
    args = parser.parse_args()
    generate_annotations(prompt_text=args.prompt)
