import argparse
import os
import yaml
from moviepy import VideoFileClip

def parse_resolution(value):
    """Parse a resolution spec into a (width, height) tuple, or None if invalid/absent.
    Accepts: [W, H] (or (W, H)) or "WxH"."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        try:
            w, h = (int(x) for x in value.lower().split("x"))
            return (w, h)
        except ValueError:
            return None
    return None

def split_videos():
    # 1. Parse arguments
    parser = argparse.ArgumentParser(description="Split raw videos into shorter clips.")
    parser.add_argument(
        "--resume", "--skip-existing",
        dest="resume",
        action="store_true",
        help="Skip videos that have already been split (output directory exists and contains mp4 clips)."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Custom input directory with raw .mp4 videos (default: config['videos_folder'])."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Custom output directory for the split clips (default: config['clips_folder'])."
    )
    parser.add_argument(
        "--resolution", type=str, default=None,
        help="Override clips resolution to WxH, e.g. 1920x1080. Default: config['clip_resolution']."
    )
    args = parser.parse_args()

    # 2. Determine root directory
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(utils_dir)
    
    # 3. Load the YAML config
    config_path = os.path.join(root_dir, "config.yaml")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return

    # 4. Setup Paths (custom --input/--output override the config defaults)
    def resolve(path, default):
        # Custom paths may be absolute or relative to the project root.
        return path if path is not None else default

    input_dir = resolve(args.input, os.path.join(root_dir, config['videos_folder']))
    output_base_dir = resolve(args.output, os.path.join(root_dir, config['clips_folder']))
    if not os.path.isabs(input_dir):
        input_dir = os.path.join(root_dir, input_dir)
    if not os.path.isabs(output_base_dir):
        output_base_dir = os.path.join(root_dir, output_base_dir)
    clip_len = config['clip_duration_minutes'] * 60

    # Determine clip resolution: --resolution overrides config['clip_resolution']
    if args.resolution:
        resize_size = parse_resolution(args.resolution)
        if resize_size is None:
            print(f"Error: invalid --resolution '{args.resolution}'. Expected WxH, e.g. 1920x1080.")
            return
    else:
        resize_size = parse_resolution(config.get("clip_resolution"))
        if resize_size is None and "clip_resolution" in config:
            print("Error: config['clip_resolution'] must be {width: int, height: int}.")
            return
    if resize_size:
        print(f">>> Clips will be resized to {resize_size[0]}x{resize_size[1]}.")

    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # 5. Process only .mp4 files
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")]

    for filename in video_files:
        input_path = os.path.join(input_dir, filename)
        
        # Create a folder for each raw video
        raw_video_name = os.path.splitext(filename)[0]
        video_output_dir = os.path.join(output_base_dir, raw_video_name)
        
        # Check if we should skip this video
        if args.resume and os.path.exists(video_output_dir) and any(f.lower().endswith(".mp4") for f in os.listdir(video_output_dir)):
            print(f"Skipping {filename}: already split (output folder contains .mp4 files)")
            continue

        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        print(f"\n--- Processing Raw Video: {filename} ---")
        
        video = None
        try:
            video = VideoFileClip(input_path, audio=False)
            duration = video.duration
            full_clips = int(duration // clip_len)

            for clip_num in range(full_clips):
                start_time = clip_num * clip_len
                end_time = start_time + clip_len
                
                # Naming clips as 01.mp4, 02.mp4, etc.
                clip_index_str = str(clip_num + 1).zfill(2)
                output_filename = f"{clip_index_str}.mp4"
                output_path = os.path.join(video_output_dir, output_filename)
                
                print(f"Exporting to {raw_video_name}/: {output_filename}")
                
                new_clip = video.subclipped(start_time, end_time)
                if resize_size is not None:
                    new_clip = new_clip.resized(new_size=resize_size)
                new_clip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio=False,
                    logger=None
                )
            
        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")
        finally:
            if video is not None:
                video.close()

    print("\nBatch processing complete.")

if __name__ == "__main__":
    split_videos()