import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid
from pathlib import Path
import subprocess

from services.sampling_service import VideoSamplingService
from services.mask_service import MaskService
from services.sam_service import SamService
from services.feat_extract_service import FeatureExtractionService
from services.behavior_service import  BehaviorPredictionService

from utils.pose import trigger_isolated_pose_inference

from services.video_service import VideoRenderService


sampling_service = VideoSamplingService()

mask_service = MaskService(
    "media/mask.png"
)

sam_service = SamService()

feature_extractor = FeatureExtractionService(model_name="resnet18", batch_size=16)

behavior_service = BehaviorPredictionService()

video_service  = VideoRenderService()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define base directories
BASE_DIR = Path("/workspace/pig-monitoring/app/backend")
UPLOAD_DIR = BASE_DIR / "data" / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"UPLOAD_DIR = {UPLOAD_DIR}")
logger.info(f"Exists = {UPLOAD_DIR.exists()}")

app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads"
)

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files to serve the uploaded and processed videos
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Uploads a video and returns its URL immediately for playback.
    """
    logger.info(f"Received upload request for file: {file.filename}")
    try:
        # 1. Create a unique ID for this session
        session_id = str(uuid.uuid4())
        video_ext = os.path.splitext(file.filename)[1]
        video_name_stem = Path(file.filename).stem
        
        # 2. Save the uploaded file to the uploads directory
        uploaded_path = UPLOAD_DIR / f"{session_id}{video_ext}"
        

        logger.info(f"Saving uploaded file to: {uploaded_path}")
        with uploaded_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        processed_path = UPLOAD_DIR / f"{session_id}_web.mp4"
        subprocess.run([
            "ffmpeg",
            "-i", str(uploaded_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart",
            str(processed_path)
        ])
        logger.info(f"Saving web compatible version to: {processed_path}")

        # 3. Apply 1fps sampling AND extract individual frames
        video_1fps_path, frames_directory = sampling_service.downsample_to_1fps(processed_path, session_id)

        # 4. Apply mask
        masked_path = mask_service.apply(video_1fps_path)

        # 5. Apply SAM
        coco_anns = sam_service.process_video(masked_path)
        
        # 6. Pose estimation (Aislado vía Subproceso)
        logger.info(f"Initiating isolated MMPose pipeline for session: {session_id}")
        try:
            frames_directory = os.path.join("data", "frames", session_id) 
            
            final_coco_with_pose = trigger_isolated_pose_inference(
                sam_coco_data=coco_anns,
                video_id=session_id,
                frames_directory=frames_directory
            )
            
            logger.info("MMPose pipeline integrated successfully into current request context.")
            
        except Exception as e:
            logger.error(f"Pose Estimation stage failed for session {session_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Pose Estimation error: {str(e)}")


        # 7. Visual and keypoint feature extraction
        logger.info(f"Initiating multi-modal feature extraction batching for session: {session_id}")
        try:
            output_npz_filename = f"{session_id}_features.npz"
            output_npz_path = os.path.join("data", "features", output_npz_filename)
            os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)

            # Process the raw predictions dictionary straight in memory
            saved_features_path = feature_extractor.extract_features_from_coco(
                coco_data=final_coco_with_pose,
                frames_directory=frames_directory,
                output_npz_path=output_npz_path,
                padding_factor=1.1
            )
            logger.info(f"Features extracted successfully and cached at: {saved_features_path}")

        except Exception as e:
            logger.error(f"Feature Extraction stage failed for session {session_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Feature Extraction error: {str(e)}")


        # 8. Getting behaviors
        predictions_output_dir = behavior_service.predict_and_count(
            session_id=session_id,
            coco_data=final_coco_with_pose
        )

        # 9. Rendering Annotated Videos
        logger.info(f"Rendering output videos for session: {session_id}")
        keypoints_video_path = video_service.generate_pose_video(
            session_id=session_id,
            coco_data=final_coco_with_pose,
            frames_dir=frames_directory,
            output_dir=UPLOAD_DIR
        )

        behavior_video_path = video_service.generate_behavior_video(
            session_id=session_id,
            coco_data=final_coco_with_pose,
            predictions_dir=predictions_output_dir,
            frames_dir=frames_directory,
            output_dir=UPLOAD_DIR
        )
        
        return {
            "session_id": session_id,
            "video_name": video_name_stem,
            "original_url": f"/uploads/{session_id}_web.mp4",
            "keypoints_url": f"/uploads/{session_id}_pose.mp4",
            "behavior_url": f"/uploads/{session_id}_behavior.mp4",
            "message": "Upload successful. Video is ready for playback."
        }

    except Exception as e:
        logger.error(f"Error during upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

@app.get("/status/{session_id}")
async def get_status(session_id: str):
    logger.info(f"Checking status for session: {session_id}")
    return {"status": "completed", "session_id": session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)