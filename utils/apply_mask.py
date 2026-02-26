import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def apply_mask(frame_path, mask, output_path):
    # Load frame
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"Warning: Could not read frame {frame_path}")
        return

    # Resize mask if necessary
    if mask.shape[:2] != frame.shape[:2]:
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask

    # Apply mask (black area in mask becomes black in frame)
    # Binary mask: 0 where black, 255 where white
    # We can use bitwise_and if the mask is 3-channel or broadcast it
    if len(mask_resized.shape) == 2:
        mask_3d = cv2.merge([mask_resized, mask_resized, mask_resized])
    else:
        mask_3d = mask_resized

    masked_frame = cv2.bitwise_and(frame, mask_3d)

    # Save frame
    os.makedirs(output_path.parent, exist_ok=True)
    cv2.imwrite(str(output_path), masked_frame)

def main():
    parser = argparse.ArgumentParser(description="Apply image mask to video frames.")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask PNG")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing frames")
    parser.add_argument("--output", type=str, required=True, help="Output directory for masked frames")
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
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    tasks = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                frame_path = Path(root) / file
                rel_path = frame_path.relative_to(input_dir)
                out_path = output_dir / rel_path
                tasks.append((frame_path, out_path))

    # Sort tasks alphabetically by frame_path to ensure consistent order
    tasks.sort(key=lambda x: str(x[0]))

    if not tasks:
        print("No images found to process.")
        return

    print(f"Found {len(tasks)} images. Applying mask...")
    for frame_path, out_path in tqdm(tasks):
        apply_mask(frame_path, mask, out_path)

    print(f"Done! Results saved in {output_dir}")

if __name__ == "__main__":
    main()
