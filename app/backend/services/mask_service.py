# services/mask_service.py
import logging
from pathlib import Path
import cv2
import numpy as np

# Configure logging at the top of your script if you haven't already
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MaskService:

    def __init__(self, mask_path: str):
        self.mask = cv2.imread(mask_path)

        if self.mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

    def apply(self, video_path: str) -> str:
        logging.info(f"Starting masking process for: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video file: {video_path}")
            raise FileNotFoundError(f"Could not open video: {video_path}")

        # 1. Dynamically generate the new path with '_masked'
        original_path = Path(video_path)
        new_filename = f"{original_path.stem}_masked{original_path.suffix}"
        output_path = str(original_path.with_name(new_filename))

        # Properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logging.info(f"Video metadata loaded. Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total Frames: {total_frames}")

        resized_mask = cv2.resize(self.mask, (frame_width, frame_height))
        resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2GRAY)

        # Video writer using the dynamic output path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break 

            # Note: Ensure 'mascara' is accessible here (e.g., self.mascara)
            frame_masked = cv2.bitwise_and(frame, frame, mask=resized_mask)

            # Write the processed frame to the new video file
            out.write(frame_masked)
            
            frame_count += 1
            # Optional: Logs progress every 100 frames so you know the server hasn't frozen
            if frame_count % 100 == 0:
                logging.info(f"Processing progress: {frame_count}/{total_frames} frames completed.")

        # Release resources
        cap.release()
        out.release()

        # Success Logs
        logging.info(f"Process successfully finished. Total frames processed: {frame_count}")
        logging.info(f"Saved masked video to: {output_path}")

        return output_path