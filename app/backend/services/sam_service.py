import os
import shutil
import logging
from pathlib import Path
import cv2
import torch
import json

# from transformers import Sam3Processor, Sam3Model
from sam3.model_builder import build_sam3_video_predictor

from utils.format import sam_to_coco

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# IMPORTANT: To make this work you have to login and create a token in huggingface.
class SamService:
    def __init__(self):
        logging.info("Initializing SAM Video Predictor Service...")
        # Automatically detect and use available GPUs
        self.gpus_to_use = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        if not self.gpus_to_use:
            logging.warning("No CUDA GPUs detected. Running on CPU might be extremely slow.")
        
        # Build the predictor instance globally for the service
        self.predictor = build_sam3_video_predictor(gpus_to_use=self.gpus_to_use)
        logging.info(f"SAM Predictor successfully loaded on GPUs: {self.gpus_to_use}")

    def process_video(
        self, 
        video_path: str, 
        prompt_text: str = "pig",
    ) -> tuple[str, int]:
        video_path_obj = Path(video_path)
        # output_json = str(video_path_obj.with_name(f"{clip_id}_coco.json"))

        # Generate virtual path names to maintain compatibility with sam_to_coco
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        virtual_frame_paths = [f"frame_{i:05d}.jpg" for i in range(total_frames)]

        try:
            logging.info(f"Starting SAM tracking session directly from video file: {video_path}")
            response = self.predictor.handle_request(dict(
                type="start_session", 
                resource_path=str(video_path)
            ))

            session_id = response.get("session_id", "default_session")
            self.predictor.handle_request(dict(type="reset_session", session_id=session_id))

            logging.info(f"Applying text grounding prompt: '{prompt_text}' on frame index 0")
            self.predictor.handle_request(dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=prompt_text
            ))

            logging.info("Propagating tracking masks through video sequence...")
            outputs_per_frame = {}
            for resp in self.predictor.handle_stream_request(dict(
                type="propagate_in_video",
                session_id=session_id
            )):
                outputs_per_frame[resp["frame_index"]] = resp["outputs"]
                if resp["frame_index"] % 50 == 0:
                    logging.info(f"SAM tracking running... Frame {resp['frame_index']}/{total_frames} computed.")

            logging.info("Converting output masks into COCO format data...")
            coco_data, _ = sam_to_coco(
                outputs_per_frame=outputs_per_frame,
                video_id='N/A',
                video_name='N/A',
                frame_paths=virtual_frame_paths,
            )

            logging.info(f"SAM Pipeline finished. Returning COCO data directly from memory.")

            return coco_data

        except Exception as e:
            logging.error(f"Critical error during SAM tracking execution: {str(e)}", exc_info=True)
            raise e



if __name__ == "__main__":
    print("\n--- STARTING DEBUGRUN FOR SAMSERVICE ---")
