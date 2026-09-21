import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def apply_mask_to_video(video_path, mask, output_path):
    """Apply mask to a video file and save as new video (preserving all frames and original FPS)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return False

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

    # Ensure output directory exists
    os.makedirs(output_path.parent, exist_ok=True)

    # Create video writer using the same FPS and frame size as the input
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (width, height))

    if not out.isOpened():
        print(f"Warning: Could not create output video {output_path}")
        cap.release()
        return False

    saved_count = 0
    pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preserve ALL frames: write every frame at the original FPS
        masked_frame = cv2.bitwise_and(frame, mask_3d)
        out.write(masked_frame)
        saved_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    print(f"  Saved {saved_count} frames to {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Apply image mask to video clips.")
    parser.add_argument("--mask", type=str, default="data/images/mask.png", help="Path to mask PNG (default: data/images/mask.png)")
    parser.add_argument("--input", type=str, default="data/videos/clips", help="Input directory containing video clips (default: data/videos/clips)")
    parser.add_argument("--output", type=str, default="data/videos/clips_masked", help="Output directory for masked videos (default: data/videos/clips_masked)")
    parser.add_argument("--video", type=str, default=None, help="Video folder name for a single clip, e.g., May_25_01 (overwrites output by default)")
    parser.add_argument("--clip", type=str, default=None, help="Clip number for a single clip, e.g., 1 or 01 (requires --video)")
    parser.add_argument("--resume", action='store_true', help="Skip videos that already exist in output directory")
    args = parser.parse_args()

    mask_path = Path(args.mask)
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Load mask as grayscale
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not read mask at {mask_path}")
        return

    # Normalize mask to binary (0 or 255)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Walk through input directory
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}

    single_clip_overwrite = False
    tasks = []

    if args.video:
        if args.clip is None:
            print("Error: --clip <number> is required when using --video.")
            return

        clip_number = str(args.clip).zfill(2)
        video_path = input_dir / args.video / f"{clip_number}.mp4"
        if not os.path.exists(video_path):
            print(f"Error: Video not found at {video_path}")
            return
        out_path = output_dir / args.video / f"{clip_number}.mp4"
        tasks.append((video_path, out_path))
        single_clip_overwrite = True
        print(f"Single clip mode: {video_path} -> {out_path} (overwrite enabled)")
    else:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if Path(file).suffix.lower() in video_extensions:
                    video_path = Path(root) / file
                    rel_path = video_path.relative_to(input_dir)
                    out_path = output_dir / rel_path
                    tasks.append((video_path, out_path))

    # Sort tasks alphabetically by video_path
    tasks.sort(key=lambda x: str(x[0]))

    if not tasks:
        print("No videos found to process.")
        return

    print(f"Found {len(tasks)} videos. Applying mask...")
    for video_path, out_path in tasks:
        if not single_clip_overwrite and args.resume and os.path.exists(out_path):
            print(f"Skipping {video_path.relative_to(input_dir)}: already exists")
            continue
        apply_mask_to_video(video_path, mask, out_path)

    print(f"Done! Results saved in {output_dir}")

if __name__ == "__main__":
    main()
