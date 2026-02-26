import os
import yaml
from moviepy import VideoFileClip

def split_videos():
    # 1. Determine root directory
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(utils_dir)
    
    # 2. Load the YAML config
    config_path = os.path.join(root_dir, "config.yaml")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return

    # 3. Setup Paths
    input_dir = os.path.join(root_dir, config['videos_folder'])
    output_base_dir = os.path.join(root_dir, config['clips_folder'])
    clip_len = config['clip_duration_minutes'] * 60

    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # 4. Process only .mp4 files
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")]

    for filename in video_files:
        input_path = os.path.join(input_dir, filename)
        
        # Create a folder for each raw video
        raw_video_name = os.path.splitext(filename)[0]
        video_output_dir = os.path.join(output_base_dir, raw_video_name)
        
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