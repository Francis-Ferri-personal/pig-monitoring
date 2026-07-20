import os
import logging
from pathlib import Path
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoSamplingService:
    """Service to downsample any video source down to exactly 1 Frame Per Second (1 FPS) 
    and extract individual frame images for downstream AI processing."""
    
    def __init__(self):
        logging.info("Video Sampling Service initialized successfully.")

    def downsample_to_1fps(self, video_path: str, session_id: str) -> tuple[str, str]:
        """
        Reads a video and creates a new downsampled version where 1 second = 1 frame.
        Additionally, dumps every sampled frame as a JPEG image into 'data/frames/{session_id}/'.
        
        Returns:
            tuple: (path_to_1fps_video, path_to_extracted_frames_dir)
        """
        logging.info(f"Downsampling video to 1 FPS for session {session_id}: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open source video for sampling: {video_path}")
            raise FileNotFoundError(f"Video not found: {video_path}")

        # 1. Retrieve source video metadata
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if original_fps <= 0:
            logging.warning("Could not detect original FPS. Defaulting frame step gap calculation to 30.")
            frame_step = 30
        else:
            # frame_step dictates how many frames to skip before extracting the next one.
            # e.g., If the video runs at 30 FPS, frame_step = 30 (capturing frames 0, 30, 60, etc.)
            frame_step = round(original_fps)

        logging.info(f"Original properties -> Resolution: {frame_width}x{frame_height}, FPS: {original_fps}, Total Frames: {total_frames}")
        logging.info(f"Sampling mathematical step gap calculated: Take 1 frame every {frame_step} frames.")

        # 2. Configure target frames directory using the unique session_id
        frames_dir = Path("data") / "frames" / session_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Frames target extraction directory prepared: {frames_dir}")

        # 3. Generate the dynamic output path appending the '_1fps' suffix
        original_path = Path(video_path)
        new_filename = f"{original_path.with_suffix('').name}_1fps{original_path.suffix}"
        output_path = str(original_path.with_name(new_filename))

        # 4. Configure the VideoWriter instance forcing an output rate of exactly 1.0 FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 1.0, (frame_width, frame_height))

        input_frame_idx = 0
        saved_frame_count = 0

        # 5. Process the video timeline extracting only the targeted intervals
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame only if it lands precisely on the calculated second boundary
            if input_frame_idx % frame_step == 0:
                # A. Write to the consolidated 1 FPS video file
                out.write(frame)
                
                # B. Write the individual image file to disk for SAM and MMPose pipelines
                # Sequential zero-padded naming convention: frame_000000.jpg, frame_000001.jpg, etc.
                frame_filename = f"frame_{saved_frame_count:05d}.jpg"
                frame_output_path = frames_dir / frame_filename
                cv2.imwrite(str(frame_output_path), frame)
                
                saved_frame_count += 1

            input_frame_idx += 1

        cap.release()
        out.release()

        logging.info(f"Downsampling and extraction completed. Saved {saved_frame_count} frames to {frames_dir}")
        logging.info(f"New 1 FPS video temporary path: {output_path}")
        
        # Return both the video path string and the isolated frames location folder
        return output_path, str(frames_dir)