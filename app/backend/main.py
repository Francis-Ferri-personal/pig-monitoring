import asyncio
import json
import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path

import yaml

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from services.video_prep_service import VideoPrepService
from services.sam_service import SamService
from services.feat_extract_service import FeatureExtractionService
from services.behavior_service import BehaviorPredictionService
from services.video_service import VideoRenderService

from utils.pose import trigger_isolated_pose_inference

# Base path for this deployment: services resolve `services.*` / `utils.*` from here.
BASE_DIR = Path(__file__).resolve().parent

# Video storage: one folder per upload, named after the video (slug).
# Legacy numeric folders ("6", ...) are still read as id "6".
VIDEOS_DIR = BASE_DIR / "data" / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = BASE_DIR / "data" / "temp"
FEATURES_DIR = BASE_DIR / "data" / "features"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "video").lower()).strip("-")
    return slug or "video"

video_prep_service = VideoPrepService(resolution=(1920, 1080))
# Smaller chunks than the 240-frame default: keeps SAM's GPU-memory peak
# (attention state per chunk) low enough to coexist with other GPU tenants.
sam_service = SamService(default_chunk_frames=120, chunk_overlap=30)
feature_extractor = FeatureExtractionService(model_name="resnet18", batch_size=16)
behavior_service = BehaviorPredictionService()
video_service = VideoRenderService(fps=5)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NOTE: the /videos static mount is registered at the END of this file (after
# all API routes) so POST /videos/{id}/retry is not swallowed by StaticFiles
# (which answers 405 for non-GET). Route order matters in Starlette.


# ---------------------------------------------------------------------------
# WebSocket hub: pushes {type: "video", video: {...}} events to all connected
# frontends (upload received / processing / completed / failed).
# ---------------------------------------------------------------------------
class VideoHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.clients: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def register(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        with self.lock:
            self._loop = asyncio.get_running_loop()
            self.clients.add(queue)
        return queue

    def unregister(self, queue: asyncio.Queue) -> None:
        with self.lock:
            self.clients.discard(queue)

    def notify(self, video: dict) -> None:
        """Thread-safe: may be called from pipeline worker threads."""
        with self.lock:
            loop = self._loop
            queues = list(self.clients)
        if loop is None:
            return
        message = json.dumps({"type": "video", "video": video}, default=str)

        def _put(queue: asyncio.Queue) -> None:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass  # Drop the event if the client is too slow.

        for queue in queues:
            loop.call_soon_threadsafe(_put, queue)


hub = VideoHub()
STATE_LOCK = threading.Lock()
PIPELINE_LOCK = threading.Lock()

# Video registry: id (slug string, e.g. "01"; legacy numeric "6") -> state dict.
_videostates: dict = {}
_reserved_ids: set = set()


def _video_dir(video_id) -> Path:
    return VIDEOS_DIR / str(video_id)


def _video_state_path(video_id) -> Path:
    return _video_dir(video_id) / "metadata.json"


def _load_states() -> dict:
    states = {}
    for folder in VIDEOS_DIR.iterdir():
        if not folder.is_dir():
            continue
        meta_path = folder / "metadata.json"
        if not meta_path.exists():
            continue
        data = json.loads(meta_path.read_text())
        data["id"] = folder.name  # folder name is the source of truth
        states[folder.name] = data
    _videostates.clear()
    _videostates.update(states)
    return states


def _save_state(video_id, video: dict) -> None:
    video_id = str(video_id)
    video["id"] = video_id
    state_path = _video_state_path(video_id)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(video, indent=2, default=str))
    tmp_path.replace(state_path)
    with STATE_LOCK:
        _videostates[video_id] = video


def _get_state(video_id) -> dict:
    return _videostates.get(str(video_id)) or {}


def _emit(video: dict) -> None:
    state = _get_state(video["id"]) or video
    hub.notify(dict(state))


def _reserve_video_id(video_name: str) -> str:
    """Reserve a slug dir named after the video (e.g. "01", "01-2" on collision)."""
    base = _slugify(Path(video_name).stem)
    with STATE_LOCK:
        candidate = base
        suffix = 2
        while (candidate in _videostates or candidate in _reserved_ids
               or _video_dir(candidate).exists()):
            candidate = f"{base}-{suffix}"
            suffix += 1
        _reserved_ids.add(candidate)
        _video_dir(candidate).mkdir(parents=True, exist_ok=True)
        return candidate


# Stage checkpoints live in the video dir so a retry resumes instead of
# starting from zero. Order matters: clearing a stage clears everything after it.
STAGE_CHECKPOINTS = ["sam_coco.json", "pose_coco.json", "features_npz", "sam_mp4", "behavior_mp4"]

STAGE_ORDER = ["resizing", "detection", "pose", "features", "rendering_sam",
               "behavior", "rendering_behavior"]


def _checkpoint_paths(video_id) -> dict:
    video_dir = _video_dir(video_id)
    return {
        "sam_coco.json": video_dir / "sam_coco.json",
        "pose_coco.json": video_dir / "pose_coco.json",
        "features_npz": FEATURES_DIR / f"{video_id}_features.npz",
        "sam_mp4": video_dir / "sam.mp4",
        "behavior_mp4": video_dir / "behavior.mp4",
    }


def _clear_checkpoints_from(video_id, stage: str) -> None:
    """Delete the checkpoint of `stage` and every later stage (for retry-from)."""
    stage_to_ckpt = {
        "resizing": [],
        "detection": ["sam_coco.json", "pose_coco.json", "features_npz", "sam_mp4", "behavior_mp4"],
        "pose": ["pose_coco.json", "features_npz", "sam_mp4", "behavior_mp4"],
        "features": ["features_npz", "sam_mp4", "behavior_mp4"],
        "rendering_sam": ["sam_mp4", "behavior_mp4"],
        "behavior": ["behavior_mp4"],
        "rendering_behavior": ["behavior_mp4"],
    }
    for key in stage_to_ckpt.get(stage, []):
        p = _checkpoint_paths(video_id)[key]
        if p.exists():
            p.unlink()
    # Pose internal progress checkpoint is also stale when redoing pose or earlier.
    if stage in ("detection", "pose"):
        internal = _video_dir(video_id) / "pose_checkpoint.json"
        if internal.exists():
            internal.unlink()


def _video_public(video: dict) -> dict:
    return dict(video)


def _behavior_classes_from_config():
    """Read behavior class names from the backend config (order by id)."""
    backend_config_path = BASE_DIR / "config.yaml"
    if not backend_config_path.exists():
        logger.error("Backend config.yaml not found! Cannot determine behavior classes.")
        raise FileNotFoundError("Backend config.yaml missing essential class mapping.")
    with open(backend_config_path, "r") as f:
        config = yaml.safe_load(f)
    mapping = config["behavior_classes"]
    max_id = max(mapping.values())
    behavior_names = [None] * (max_id + 1)
    for name, idx in mapping.items():
        behavior_names[idx] = name
    return [name for name in behavior_names if name is not None]


# ---------------------------------------------------------------------------
# Pipeline: runs in a background thread, one video at a time.
# ---------------------------------------------------------------------------
def run_pipeline(video_id, video_name: str) -> None:
    video_id = str(video_id)
    video_dir = _video_dir(video_id)
    raw_path = video_dir / "raw.mp4"
    resized_path = video_dir / "resized.mp4"
    sam_video_path = video_dir / "sam.mp4"
    behavior_video_path = video_dir / "behavior.mp4"
    ckpts = _checkpoint_paths(video_id)

    def _update(**fields):
        with STATE_LOCK:
            state = dict(_get_state(video_id))
            state.update(fields)
            state["updated_at"] = time.time()
            _videostates[video_id] = state
        _save_state(video_id, state)
        _emit(state)

    try:
        logger.info(f"[video {video_id}] Pipeline started")

        # Stage metadata only when a stage finishes so an interrupted run never
        # exposes a missing stage video as "completed".
        _update(status="processing", stage="resizing")

        # 1. Normalize resolution to 1920x1080 @ 5fps (skip if already prepared).
        if resized_path.exists() and resized_path.stat().st_size > 0:
            prepared_path = str(resized_path)
            logger.info(f"[video {video_id}] Reusing prepared video at: {resized_path}")
        else:
            prepared_path = video_prep_service.prepare(str(raw_path), output_path=str(resized_path))
            if Path(prepared_path).resolve() != resized_path.resolve():
                shutil.copyfile(prepared_path, resized_path)
            logger.info(f"[video {video_id}] Prepared video at: {resized_path}")

        # 2. SAM 3 tracking (resume from sam_coco.json when present; legacy
        #    temp sam_raw_{id}.json from an interrupted run also counts).
        _update(stage="detection")
        coco_anns = None
        if ckpts["sam_coco.json"].exists():
            logger.info(f"[video {video_id}] Resuming from checkpoint sam_coco.json")
            coco_anns = json.loads(ckpts["sam_coco.json"].read_text())
        else:
            legacy_sam = TEMP_DIR / f"sam_raw_{video_id}.json"
            if legacy_sam.exists():
                logger.info(f"[video {video_id}] Resuming from legacy {legacy_sam.name}")
                coco_anns = json.loads(legacy_sam.read_text())
                ckpts["sam_coco.json"].write_text(json.dumps(coco_anns))
        if coco_anns is None:
            coco_anns = sam_service.process_video(prepared_path, str(video_id))
            ckpts["sam_coco.json"].write_text(json.dumps(coco_anns))
        logger.info(f"[video {video_id}] SAM tracking finished")

        # 3. Pose estimation (resume from pose_coco.json when present; the
        #    isolated worker itself resumes frame-chunks via pose_checkpoint.json).
        _update(stage="pose")
        logger.info(f"Initiating isolated MMPose pipeline for video: {video_id}")
        try:
            if ckpts["pose_coco.json"].exists():
                logger.info(f"[video {video_id}] Resuming from checkpoint pose_coco.json")
                final_coco_with_pose = json.loads(ckpts["pose_coco.json"].read_text())
            else:
                final_coco_with_pose = trigger_isolated_pose_inference(
                    sam_coco_data=coco_anns,
                    video_id=video_id,
                    video_path=prepared_path,
                )
                ckpts["pose_coco.json"].write_text(json.dumps(final_coco_with_pose))
            logger.info("MMPose pipeline integrated successfully into current request context.")
        except Exception as e:
            logger.error(f"Pose Estimation stage failed for video {video_id}: {str(e)}")
            raise

        # 4. Feature extraction (resume: the .npz cache is the checkpoint).
        _update(stage="features")
        logger.info(f"Initiating multimodal feature extraction for video: {video_id}")
        try:
            output_npz_path = str(ckpts["features_npz"])
            os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)

            if ckpts["features_npz"].exists():
                saved_features_path = output_npz_path
                logger.info(f"[video {video_id}] Reusing features cache at: {saved_features_path}")
            else:
                saved_features_path = feature_extractor.extract_features_from_coco(
                    coco_data=final_coco_with_pose,
                    video_path=prepared_path,
                    output_npz_path=output_npz_path,
                    padding_factor=1.1
                )
            logger.info(f"Features extracted successfully and cached at: {saved_features_path}")
        except Exception as e:
            logger.error(f"Feature Extraction stage failed for video {video_id}: {str(e)}")
            raise

        # 5. SAM/keypoints video (resume: skip when sam.mp4 already rendered).
        _update(stage="rendering_sam")
        if sam_video_path.exists() and sam_video_path.stat().st_size > 0:
            logger.info(f"[video {video_id}] Reusing SAM video at: {sam_video_path}")
        else:
            video_service.generate_pose_video(
                session_id=str(video_id),
                coco_data=final_coco_with_pose,
                video_path=prepared_path,
                output_dir=video_dir,
                fps=5,
            )
            rendered_sam = video_dir / f"{video_id}_pose.mp4"
            if rendered_sam.resolve() != sam_video_path.resolve():
                shutil.move(rendered_sam, sam_video_path)
            logger.info(f"[video {video_id}] SAM annotated video saved at: {sam_video_path}")

        # 6-7. Behavior inference + video (resume: skip both when behavior.mp4 exists).
        if behavior_video_path.exists() and behavior_video_path.stat().st_size > 0:
            logger.info(f"[video {video_id}] Reusing behavior video at: {behavior_video_path}")
        else:
            _update(stage="behavior")
            csv_path, pigs_predictions = behavior_service.predict_and_count(
                session_id=str(video_id),
                coco_data=final_coco_with_pose
            )

            _update(stage="rendering_behavior")
            behavior_classes = _behavior_classes_from_config()
            video_service.generate_behavior_video(
                session_id=str(video_id),
                coco_data=final_coco_with_pose,
                pigs_predictions=pigs_predictions,
                video_path=prepared_path,
                output_dir=video_dir,
                behavior_classes=behavior_classes,
                fps=5,
            )
            rendered_behavior = video_dir / f"{video_id}_behavior.mp4"
            if rendered_behavior.resolve() != behavior_video_path.resolve():
                shutil.move(rendered_behavior, behavior_video_path)
            logger.info(f"[video {video_id}] Behavior annotated video saved at: {behavior_video_path}")

        _update(
            status="completed",
            stage=None,
            original_url=f"/videos/{video_id}/raw.mp4",
            keypoints_url=f"/videos/{video_id}/sam.mp4",
            behavior_url=f"/videos/{video_id}/behavior.mp4",
            error=None,
        )
        logger.info(f"[video {video_id}] Pipeline completed successfully")

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[video {video_id}] Pipeline failed: {error_msg}", exc_info=True)
        _update(status="failed", stage=None, error=error_msg)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Saves the uploaded video and starts the full pipeline in the background:
      resize (1920x1080 @ 5fps) -> SAM 3 -> Pose -> Features -> Annotated videos.
    Returns the new video id immediately; progress is streamed over the /ws socket.
    """
    logger.info(f"Received upload request for file: {file.filename}")
    video_ext = os.path.splitext(file.filename)[1].lower()
    if video_ext not in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {video_ext or 'unknown'}")

    video_name = Path(file.filename).stem
    # The id IS the video name (slug). "01.mp4" -> id "01".
    video_id = _reserve_video_id(video_name)

    raw_path = video_dir = None
    try:
        video_dir = _video_dir(video_id)
        raw_path = video_dir / "raw.mp4"
        with raw_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving upload: {str(e)}")
        with STATE_LOCK:
            _reserved_ids.discard(video_id)
        if video_dir is not None:
            shutil.rmtree(video_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error saving upload: {str(e)}")

    state = {
        "id": video_id,
        "name": video_name,
        "status": "received",
        "stage": None,
        "original_url": None,
        "keypoints_url": None,
        "behavior_url": None,
        "error": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    _save_state(video_id, state)
    with STATE_LOCK:
        _reserved_ids.discard(video_id)
    _emit(state)

    def _run_sequentially():
        with PIPELINE_LOCK:
            run_pipeline(video_id, video_name)

    worker = threading.Thread(target=_run_sequentially, name=f"pipeline-{video_id}", daemon=True)
    worker.start()

    return {
        "id": video_id,
        "name": video_name,
        "status": "processing",
        "message": "Upload received. Processing started; watch the websocket for completion.",
    }


@app.get("/videos")
async def list_videos():
    """Returns every processed/completed video, newest first."""
    with STATE_LOCK:
        all_videos = [_video_public(v) for v in _videostates.values()]
    completed = [v for v in all_videos if v["status"] == "completed"]
    return sorted(completed, key=lambda v: v.get("created_at", 0), reverse=True)


@app.post("/videos/{video_id}/retry")
async def retry_video(video_id: str, from_stage: str = None):
    """Re-run a failed (or any) video, resuming from checkpoints.

    Query `from_stage` (resizing|detection|pose|features|rendering_sam|behavior|
    rendering_behavior) clears that stage's checkpoint and everything after it,
    so only the last part is redone. Default: resume from the first missing
    checkpoint without deleting anything.
    """
    video_id = str(video_id)
    state = _get_state(video_id)
    if not state and not _video_dir(video_id).exists():
        raise HTTPException(status_code=404, detail=f"Unknown video: {video_id}")
    if from_stage:
        if from_stage not in STAGE_ORDER:
            raise HTTPException(status_code=400,
                                detail=f"Unknown stage: {from_stage}. Use one of {STAGE_ORDER}")
        _clear_checkpoints_from(video_id, from_stage)
    video_name = (state.get("name") if state else None) or video_id

    def _run_sequentially():
        with PIPELINE_LOCK:
            run_pipeline(video_id, video_name)

    worker = threading.Thread(target=_run_sequentially, name=f"pipeline-retry-{video_id}", daemon=True)
    worker.start()
    return {"id": video_id, "status": "processing",
            "message": f"Retry started for {video_id}" + (f" from {from_stage}." if from_stage else " (resuming).")}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    queue = await hub.register()
    try:
        # Replay current states so a freshly opened socket immediately knows
        # about all in-flight / completed uploads.
        with STATE_LOCK:
            snapshot = [_video_public(v) for v in _videostates.values()]
        for video in snapshot:
            try:
                await websocket.send_text(json.dumps({"type": "video", "video": video}, default=str))
            except (WebSocketDisconnect, RuntimeError):
                return
        while True:
            message = await queue.get()
            try:
                await websocket.send_text(message)
            except (WebSocketDisconnect, RuntimeError):
                break
    except WebSocketDisconnect:
        pass
    finally:
        hub.unregister(queue)


_load_states()

# Static files LAST: API routes above take precedence (see note at the top).
app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
