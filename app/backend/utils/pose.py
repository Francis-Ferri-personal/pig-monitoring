import os
import json
import logging
import subprocess

POSE_ENV_PYTHON = "/workspace/pig-monitoring/.venv-pose/bin/python"

def trigger_isolated_pose_inference(sam_coco_data: dict, video_id: int, frames_directory: str) -> dict:
    """
    Bridges the main backend environment with the isolated MMPose environment (.venv-pose)
    via an automated terminal subprocess.
    """
    logging.info(f"Preparing serialization exchange channels for video ID: {video_id}")
    
    os.makedirs("data/temp", exist_ok=True)
    temp_input_path = f"data/temp/sam_raw_{video_id}.json"
    temp_output_path = f"data/temp/pose_res_{video_id}.json"
    
    # 1. Write datat temporary
    with open(temp_input_path, 'w') as f:
        json.dump(sam_coco_data, f)
    
    # 2. Build CLI command pointing to the file we just saved above
    command = [
        POSE_ENV_PYTHON,
        "services/pose_service.py",
        "--input_json", temp_input_path,
        "--output_json", temp_output_path,
        "--frames_dir", frames_directory,
        "--batch_size", "32"
    ]
    
    logging.info("Launching isolated MMPose environment subprocess block...")
    
    # 3. Launch subprocess synchronously/blocking and capture outputs
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Validate internal segmentation failures or missing dependencies in .venv-pose
    if result.returncode != 0:
        logging.error(f"Isolated Pose environment crashed! Stderr logs:\n{result.stderr}")
        raise RuntimeError(f"MMPose Subprocess failed: {result.stderr}")
        
    # 4. Load the processed JSON with keypoints directly into memory
    logging.info("MMPose subprocess finished cleanly. Loading data back to memory...")
    with open(temp_output_path, 'r') as f:
        final_coco_data = json.load(f)
        
    # Remove tmp files
    if os.path.exists(temp_input_path): os.remove(temp_input_path)
    if os.path.exists(temp_output_path): os.remove(temp_output_path)
        
    return final_coco_data