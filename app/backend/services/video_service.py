import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2

from services.video_style import (
    SKELETON_CONNECTIONS,
    convert_to_web_mp4,
    draw_pose_annotations,
)
from utils.frame_cache import FrameReader

logger = logging.getLogger(__name__)


class VideoRenderService:
    """
    Service responsible for rendering annotated videos (Keypoints, Tracking IDs,
    and Behavior Predictions) by reading frames directly from the prepared video
    file (no per-frame JPEGs on disk).
    """

    SKELETON_CONNECTIONS = SKELETON_CONNECTIONS

    def __init__(self, fps: Optional[float] = None):
        self.fps = fps

    def generate_pose_video(
        self,
        session_id: str,
        coco_data: Dict[str, Any],
        video_path: str,
        output_dir: Path,
        fps: Optional[float] = None,
    ) -> Path:
        """Generate a video overlay showing BBoxes, Track IDs, and Keypoint Skeletons."""
        raw_output_path = output_dir / f"{session_id}_pose_raw.mp4"
        final_output_path = output_dir / f"{session_id}_pose.mp4"

        logger.info(f"Starting Pose video rendering for session: {session_id} (from {video_path})")

        self._build_video(
            video_path=video_path,
            output_video_path=raw_output_path,
            fps=fps or self.fps or 5.0,
            draw_callback=lambda frame, frame_idx, _src: self._draw_pose_overlay(
                frame, frame_idx, coco_data
            ),
        )

        self._convert_to_web_mp4(raw_output_path, final_output_path)

        if raw_output_path.exists():
            os.remove(raw_output_path)

        logger.info(f"Pose video successfully saved at: {final_output_path}")
        return final_output_path

    def generate_behavior_video(
        self,
        session_id: str,
        coco_data: Dict[str, Any],
        pigs_predictions: Dict[int, Dict[int, int]],
        video_path: str,
        output_dir: Path,
        behavior_classes: List[str],
        fps: Optional[float] = None,
    ) -> Path:
        """Generate a video overlay showing BBoxes and behavior prediction labels per frame."""
        raw_output_path = output_dir / f"{session_id}_behavior_raw.mp4"
        final_output_path = output_dir / f"{session_id}_behavior.mp4"
        log_file_path = output_dir / f"{session_id}_behavior_log.csv"

        with open(log_file_path, "w") as f:
            f.write("pig_id,frame,behavior\n")

        logger.info(f"Starting Behavior video rendering for session: {session_id} (from {video_path})")

        self._build_video(
            video_path=video_path,
            output_video_path=raw_output_path,
            fps=fps or self.fps or 5.0,
            draw_callback=lambda frame, frame_idx, _src: self._draw_behavior_overlay(
                frame, frame_idx, coco_data, pigs_predictions, behavior_classes, log_file_path
            ),
        )

        self._convert_to_web_mp4(raw_output_path, final_output_path)

        if raw_output_path.exists():
            os.remove(raw_output_path)

        logger.info(f"Behavior video successfully saved at: {final_output_path}")
        return final_output_path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_video(
        self,
        video_path: str,
        output_video_path: Path,
        fps: float,
        draw_callback,
    ) -> None:
        """
        Iterate over every frame of `video_path` (decoded via FrameReader),
        apply `draw_callback`, and write the resulting frames to an MP4 at `fps`.
        """
        frame_reader = FrameReader(video_path)
        try:
            total_frames = frame_reader.total_frames
            if total_frames == 0:
                raise FileNotFoundError(f"Video has no frames: {video_path}")

            first = frame_reader.read(0)
            if first is None:
                raise FileNotFoundError(f"Could not read first frame of: {video_path}")
            height, width, _ = first.shape

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
            if not out.isOpened():
                raise RuntimeError(f"Could not open VideoWriter: {output_video_path}")

            for idx in range(total_frames):
                frame = frame_reader.read(idx)
                if frame is None:
                    continue
                annotated = draw_callback(frame, idx, video_path)
                out.write(annotated)

            out.release()
        finally:
            frame_reader.close()

    def _annotations_for_frame(self, coco_data: Dict[str, Any], frame_idx: int):
        """Find COCO annotations whose image.frame_id equals `frame_idx`."""
        for image in coco_data.get("images", []):
            if image.get("frame_id") == frame_idx:
                image_id = image.get("id")
                return [a for a in coco_data.get("annotations", []) if a.get("image_id") == image_id]
        return []

    def _draw_pose_overlay(self, frame, frame_idx: int, coco_data: Dict[str, Any]):
        return draw_pose_annotations(frame, self._annotations_for_frame(coco_data, frame_idx))

    def _draw_behavior_overlay(
        self,
        frame,
        frame_idx: int,
        coco_data: Dict[str, Any],
        pigs_predictions: Dict[int, Dict[int, int]],
        behavior_classes: List[str],
        log_file_path: Path,
    ):
        """Draw behavior labels per frame alongside BBoxes (no keypoints)."""
        annotations = self._annotations_for_frame(coco_data, frame_idx)
        frame = draw_pose_annotations(frame, annotations, draw_keypoints=False)

        for ann in annotations:
            track_id = ann.get("track_id", 0)
            if track_id is None:
                continue

            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue

            x, y, w, h = map(int, bbox)

            pred_label = "Waiting"
            if track_id in pigs_predictions and frame_idx in pigs_predictions[track_id]:
                pred_idx = pigs_predictions[track_id][frame_idx]
                if 0 <= pred_idx < len(behavior_classes):
                    pred_label = behavior_classes[pred_idx]

            log_entry = f"{track_id},{frame_idx},{pred_label}"
            print(log_entry)
            with open(log_file_path, "a") as f:
                f.write(log_entry + "\n")

            from services.video_style import _track_color
            color = _track_color(track_id)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            text_pred = f"Pred: {pred_label}"
            (tw, th), _ = cv2.getTextSize(text_pred, font, font_scale, thickness)
            bg_height = th + 10

            cv2.rectangle(
                frame,
                (x, max(0, y - bg_height)),
                (x + tw + 10, y),
                color,
                -1,
            )

            cv2.putText(
                frame,
                text_pred,
                (x + 5, max(th + 5, y - 5)),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

        return frame

    def _convert_to_web_mp4(self, input_path: Path, output_path: Path) -> None:
        try:
            convert_to_web_mp4(input_path, output_path)
        except RuntimeError as error:
            logger.error("FFmpeg conversion failed: %s", error)
            raise


video_render_service = VideoRenderService()
