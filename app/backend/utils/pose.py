import os
import json
import logging
import subprocess

import sys
from pathlib import Path

# Absolute path to the isolated MMPose environment.
POSE_ENV_PYTHON = "/workspace/pig-monitoring/.venv-pose/bin/python"

# SAM holds cuda:0 with ~23GB reserved (see sam3 logs), gpu1 is free.
# Keep pose on the free GPU to avoid CUDA contention/OOM.
POSE_DEVICE = os.environ.get("POSE_DEVICE", "cuda:1")

# Absolute path to the app/backend project root. The bridge and the subprocess
# both run from this directory so `services/...` and `utils/...` resolve.
APP_BACKEND = Path(__file__).resolve().parent.parent


def _run_from_backend(cmd: list, cwd: Path) -> subprocess.CompletedProcess:
    """Run a command with CWD set to the app/backend root."""
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))


def trigger_isolated_pose_inference(sam_coco_data: dict, video_id, video_path: str) -> dict:
    """
    Bridges the main backend environment with the isolated MMPose environment (.venv-pose)
    via an automated terminal subprocess. The prepared video path is forwarded so the
    pose worker can read its frames directly from disk (no per-frame JPEGs needed).

    Pose runs frame-chunked with a checkpoint at data/videos/{video_id}/pose_checkpoint.json,
    so a killed run resumes instead of starting from zero.
    """
    logging.info(f"Preparing serialization exchange channels for video ID: {video_id}")

    temp_dir = APP_BACKEND / "data" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    video_dir = APP_BACKEND / "data" / "videos" / str(video_id)
    video_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(video_dir / "pose_checkpoint.json")

    if not os.path.isabs(video_path):
        video_path_abs = str(APP_BACKEND / video_path)
    else:
        video_path_abs = video_path

    temp_input_path = str(temp_dir / f"sam_raw_{video_id}.json")
    temp_output_path = str(temp_dir / f"pose_res_{video_id}.json")

    # 1. Serialize the SAM annotations to a temporary JSON.
    with open(temp_input_path, 'w') as f:
        json.dump(sam_coco_data, f)

    # 2. Build the CLI command (resume from checkpoint when one exists).
    resume_flag = ["--resume"] if os.path.exists(checkpoint_path) else []
    command = [
        POSE_ENV_PYTHON,
        "services/pose_service.py",
        "--input_json", temp_input_path,
        "--output_json", temp_output_path,
        "--video_path", video_path_abs,
        "--batch_size", "32",
        "--device", POSE_DEVICE,
        "--checkpoint", checkpoint_path,
        "--chunk_frames", "25",
        *resume_flag,
    ]

    logging.info(f"Launching isolated MMPose environment subprocess on {POSE_DEVICE}...")
    result = _run_from_backend(command, cwd=APP_BACKEND)

    # 3. Validate the subprocess (log returncode + tails, not full dumps).
    if result.returncode != 0:
        stdout_tail = (result.stdout or "")[-3000:]
        stderr_tail = (result.stderr or "")[-3000:]
        logging.error(
            f"MMPose subprocess failed. returncode={result.returncode} "
            f"device={POSE_DEVICE}\n--- stdout tail ---\n{stdout_tail}\n--- stderr tail ---\n{stderr_tail}"
        )
        raise RuntimeError(f"MMPose subprocess failed (rc={result.returncode}): {stderr_tail}")

    # 4. Load the processed JSON with keypoints back into memory.
    logging.info("MMPose subprocess finished. Loading data back to memory...")
    with open(temp_output_path, 'r') as f:
        final_coco_data = json.load(f)

    # 5. Clean up temp files.
    if os.path.exists(temp_input_path):
        os.remove(temp_input_path)
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    return final_coco_data
