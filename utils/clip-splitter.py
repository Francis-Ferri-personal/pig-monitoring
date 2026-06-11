import argparse
import cv2
import os
import yaml

def extract_frames():
    # 1. Parse arguments
    parser = argparse.ArgumentParser(description="Extract frames from video clips.")
    parser.add_argument(
        "--resume", "--skip-existing",
        dest="resume",
        action="store_true",
        help="Skip clips that have already had frames extracted (output directory contains image files)."
    )
    args = parser.parse_args()

    # 2. Setup Paths
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(utils_dir)
    config_path = os.path.join(root_dir, "config.yaml")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return

    clips_root_dir = os.path.join(root_dir, config['clips_folder'])
    frames_root_dir = os.path.join(root_dir, config['frames_folder']) # Asegúrate que coincida con tu yaml
    fps_target = config['frames_per_second']
    img_ext = config.get('image_extension', 'png')


    # 3. Walk through subfolders (Video_A, Video_B, etc.)
    for subdir, dirs, files in sorted(os.walk(clips_root_dir)):
        # Filter only mp4 files
        clip_files = [f for f in files if f.lower().endswith(".mp4")]
        
        # Sort clip_files by name
        clip_files.sort()
        
        if not clip_files:
            continue

        # Determine relative path to recreate folder structure in frames_folder
        relative_path = os.path.relpath(subdir, clips_root_dir)
        
        for clip_name in clip_files:
            clip_path = os.path.join(subdir, clip_name)
            
            # Create output path: frames_folder / Video_A / clip_01 /
            clip_base_name = os.path.splitext(clip_name)[0]
            clip_output_dir = os.path.join(frames_root_dir, relative_path, clip_base_name)
            
            # Check if we should skip this clip
            if args.resume and os.path.exists(clip_output_dir) and any(f.lower().endswith(f".{img_ext}") for f in os.listdir(clip_output_dir)):
                print(f"Skipping {relative_path}/{clip_name}: already extracted (output folder contains images)")
                continue

            os.makedirs(clip_output_dir, exist_ok=True)

            cap = cv2.VideoCapture(clip_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {clip_path}")
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            
            if video_fps == 0:
                print(f"Error: FPS is 0 for {clip_name}. The file might be corrupted or in an unsupported format.")
                cap.release()
                continue

            hop_interval = max(1, round(video_fps / fps_target))
            frame_count = 0
            saved_count = 0

            print(f"--- Extracting {img_ext.upper()} from: {relative_path}/{clip_name} ---")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % hop_interval == 0:
                    frame_filename = f"{saved_count:05d}.{img_ext}"
                    save_path = os.path.join(clip_output_dir, frame_filename)
                    
                    if img_ext == 'png':
                        cv2.imwrite(save_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    else:
                        cv2.imwrite(save_path, frame)
                    saved_count += 1
                
                frame_count += 1
            
            cap.release()
            print(f"Success: {saved_count} frames stored in {clip_output_dir}")

if __name__ == "__main__":
    extract_frames()