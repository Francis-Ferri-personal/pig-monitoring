import os
import cv2
import glob
import logging
from pathlib import Path
from typing import Dict, Any

from services.video_style import (
    SKELETON_CONNECTIONS,
    convert_to_web_mp4,
    draw_pose_annotations,
    draw_prediction_label,
)

logger = logging.getLogger(__name__)


class VideoRenderService:
    """
    Service responsible for rendering annotated videos (Keypoints, Tracking IDs,
    and Behavior Predictions) from frame sequences and COCO annotations.
    """

    # Kept as a class attribute for backwards compatibility.
    SKELETON_CONNECTIONS = SKELETON_CONNECTIONS

    def __init__(self, fps: int = 1):
        """
        Initialize the video render service.

        :param fps: Frames per second for the output video (default: 1fps).
        """
        self.fps = fps

    def generate_pose_video(
        self,
        session_id: str,
        coco_data: Dict[str, Any],
        frames_dir: str,
        output_dir: Path
    ) -> Path:
        """
        Generates a video overlay showing Bounding Boxes, Track IDs, and Keypoint Skeletons.

        :param session_id: Unique session identifier.
        :param coco_data: Dictionary containing COCO-formatted annotations with pose keypoints.
        :param frames_dir: Directory containing individual frame images (.jpg/.png).
        :param output_dir: Destination path where the final video will be saved.
        :return: Path to the web-compatible rendered video.
        """
        raw_output_path = output_dir / f"{session_id}_pose_raw.mp4"
        final_output_path = output_dir / f"{session_id}_pose.mp4"

        logger.info(f"Starting Pose video rendering for session: {session_id}")

        # Process frames and write raw MP4 using OpenCV
        self._build_video_from_frames(
            frames_dir=frames_dir,
            output_video_path=raw_output_path,
            draw_callback=lambda frame, frame_idx, frame_path: self._draw_pose_overlay(frame, frame_idx, coco_data, frame_path)
        )

        # Transcode raw video to standard H.264 format for browser web compatibility
        self._convert_to_web_mp4(raw_output_path, final_output_path)

        # Clean up temporary raw video file
        if raw_output_path.exists():
            os.remove(raw_output_path)

        logger.info(f"Pose video successfully saved at: {final_output_path}")
        return final_output_path

    def generate_behavior_video(
        self,
        session_id: str,
        coco_data: Dict[str, Any],
        predictions_dir: str,
        frames_dir: str,
        output_dir: Path
    ) -> Path:
        """
        Generates a video overlay showing BBoxes, Keypoints, Ground Truth, and Behavior Predictions.

        :param session_id: Unique session identifier.
        :param coco_data: Dictionary containing COCO-formatted annotations.
        :param predictions_dir: Directory containing prediction results (e.g., JSON/CSV output).
        :param frames_dir: Directory containing individual frame images.
        :param output_dir: Destination path where the final video will be saved.
        :return: Path to the web-compatible rendered video.
        """
        raw_output_path = output_dir / f"{session_id}_behavior_raw.mp4"
        final_output_path = output_dir / f"{session_id}_behavior.mp4"

        logger.info(f"Starting Behavior video rendering for session: {session_id}")

        self._build_video_from_frames(
            frames_dir=frames_dir,
            output_video_path=raw_output_path,
            draw_callback=lambda frame, frame_idx, frame_path: self._draw_behavior_overlay(
                frame, frame_idx, coco_data, predictions_dir, frame_path
            )
        )

        # Transcode raw video to standard H.264 format for browser web compatibility
        self._convert_to_web_mp4(raw_output_path, final_output_path)

        # Clean up temporary raw video file
        if raw_output_path.exists():
            os.remove(raw_output_path)

        logger.info(f"Behavior video successfully saved at: {final_output_path}")
        return final_output_path

    def _build_video_from_frames(self, frames_dir: str, output_video_path: Path, draw_callback) -> None:
        """
        Helper method to iterate through frames, apply custom drawing callbacks, and write an MP4.
        """
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg"))) + \
                      sorted(glob.glob(os.path.join(frames_dir, "*.png")))

        if not frame_files:
            raise FileNotFoundError(f"No frame images found in directory: {frames_dir}")

        # Read the first frame to get video dimensions
        sample_frame = cv2.imread(frame_files[0])
        height, width, _ = sample_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, self.fps, (width, height))

        for frame_idx, frame_path in enumerate(frame_files):
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            # Apply annotations on the current frame
            annotated_frame = draw_callback(frame, frame_idx, frame_path)
            out.write(annotated_frame)

        out.release()

    def _annotations_for_frame(self, coco_data: Dict[str, Any], frame_idx: int, frame_path: str):
        """Find COCO annotations for the exact source image, without index offsets."""
        images = coco_data.get("images", [])
        image_name = Path(frame_path).name
        image = next((item for item in images if Path(item.get("file_name", "")).name == image_name), None)
        if image is None:
            image = next((item for item in images if item.get("frame_id") == frame_idx), None)
        if image is None:
            ordered_images = sorted(images, key=lambda item: item.get("frame_id", item.get("id", 0)))
            image = ordered_images[frame_idx] if frame_idx < len(ordered_images) else None
        if image is None:
            return []
        image_id = image.get("id")
        return [annotation for annotation in coco_data.get("annotations", []) if annotation.get("image_id") == image_id]

    def _draw_pose_overlay(self, frame: cv2.Mat, frame_idx: int, coco_data: Dict[str, Any], frame_path: str) -> cv2.Mat:
        """Draw pose annotations matched to the actual frame file."""
        return draw_pose_annotations(frame, self._annotations_for_frame(coco_data, frame_idx, frame_path))


    def _draw_behavior_overlay(
        self,
        frame: cv2.Mat,
        frame_idx: int,
        coco_data: Dict[str, Any],
        predictions_dir: str,
        frame_path: str
    ) -> cv2.Mat:
        """
        Draws behavior labels (Ground Truth & Predicted Behaviors) alongside bounding boxes.
        """
        # First draw base pose annotations
        frame = draw_pose_annotations(
            frame, self._annotations_for_frame(coco_data, frame_idx, frame_path), draw_keypoints=False
        )

        # Overlay behavior prediction labels over the same source image.
        annotations = self._annotations_for_frame(coco_data, frame_idx, frame_path)

        for ann in annotations:
            draw_prediction_label(
                frame,
                ann.get("bbox", []),
                ann.get("predicted_behavior", "Feeding"),
            )

        return frame

    def _convert_to_web_mp4(self, input_path: Path, output_path: Path) -> None:
        """
        Converts a raw MP4 file into an H.264 / AAC web-compatible MP4 using FFmpeg.
        """
        try:
            convert_to_web_mp4(input_path, output_path)
        except RuntimeError as error:
            logger.error("FFmpeg conversion failed: %s", error)
            raise


# Singleton instance ready for importation
video_render_service = VideoRenderService(fps=1)