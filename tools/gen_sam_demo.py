import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

def process_and_save_video(input_path, output_path, prompt_text):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load the SAM 3 model and processor from Hugging Face
    print("Loading facebook/sam3 model...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    # 2. Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video file: {input_path}")

    # Retrieve video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate total frame limit for 60 seconds
    max_frames = fps * 60
    print(f"FPS detected: {fps}. Processing up to {max_frames} frames (1 minute).")

    # 3. Configure video output (VideoWriter)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Processing frames...")
    frame_count = 0

    while cap.isOpened():
        # Stop processing once the 1-minute limit is reached
        if frame_count >= max_frames:
            print("Reached 1-minute video limit.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from BGR (OpenCV) to RGB and then to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Preprocess frame and prompt text with SAM 3
        inputs = processor(images=pil_image, text=prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process segmentation masks to match original image size
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        masks = results["masks"] # Tensor shape: [N, Height, Width]

        # 4. Overlay masks onto the current frame
        if len(masks) > 0:
            # Create a color overlay (e.g., Green in BGR format)
            color_overlay = np.zeros_like(frame, dtype=np.uint8)
            
            # Combine all detected masks into a single boolean mask
            combined_mask = torch.any(masks, dim=0).cpu().numpy().astype(bool)
            
            # Set mask color (BGR: [0, 255, 0] for Green)
            color_overlay[combined_mask] = [0, 255, 0]

            # Blend mask layer with the original frame (40% mask opacity, 60% original)
            frame = cv2.addWeighted(frame, 0.6, color_overlay, 0.4, 0)

        # 5. Write the processed frame to the output video file
        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed frames: {frame_count}/{max_frames}")

    # Release resources
    cap.release()
    out.release()
    print(f"Video saved successfully at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Segmentation using SAM 3")
    parser.add_argument("--video_path", "-v", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_path", "-o", type=str, default="output_segmented.mp4", help="Path to output video file")
    parser.add_argument("--prompt", "-p", type=str, default="pig", help="Text prompt for target object")

    args = parser.parse_args()

    process_and_save_video(args.video_path, args.output_path, args.prompt)