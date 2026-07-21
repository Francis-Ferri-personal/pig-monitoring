import os
import cv2
import glob
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class VideoRenderService:
    """
    Service responsible for rendering annotated videos (Keypoints, Tracking IDs,
    and Behavior Predictions) from frame sequences and COCO annotations.
    """

    # COCO Keypoint Skeletons (Pairs of connected keypoint indices)
    # Adjust this skeleton topology if your keypoint format differs
    SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
        (0, 1), (0, 2), (1, 3), (2, 4),        # Facial keypoints / Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),# Upper body / Limbs
        (5, 11), (6, 12), (11, 12),             # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Lower body / Legs
    ]

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
            draw_callback=lambda frame, frame_idx: self._draw_pose_overlay(frame, frame_idx, coco_data)
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
            draw_callback=lambda frame, frame_idx: self._draw_behavior_overlay(
                frame, frame_idx, coco_data, predictions_dir
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
            annotated_frame = draw_callback(frame, frame_idx)
            out.write(annotated_frame)

        out.release()

    def _draw_pose_overlay(self, frame: cv2.Mat, frame_idx: int, coco_data: Dict[str, Any]) -> cv2.Mat:
        """
        Draws bounding boxes, track IDs, and keypoint skeletons for a specific frame index.
        """
        # Search annotations corresponding to the current frame index
        image_id = frame_idx + 1  # Standard COCO 1-based indexing
        annotations = [ann for ann in coco_data.get("annotations", []) if ann.get("image_id") == image_id]

        for ann in annotations:
            track_id = ann.get("track_id", ann.get("id", "N/A"))
            bbox = ann.get("bbox", [])  # Format: [x, y, width, height]
            keypoints = ann.get("keypoints", [])  # Format: [x1, y1, v1, x2, y2, v2, ...]

            # 1. Draw Bounding Box
            if len(bbox) == 4:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw ID Tag
                label = f"ID: {track_id}"
                cv2.putText(frame, label, (x, max(y - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 2. Draw Keypoints and Connections
            if keypoints and len(keypoints) >= 3:
                parsed_kpts = []
                for i in range(0, len(keypoints), 3):
                    kx, ky, vis = keypoints[i], keypoints[i+1], keypoints[i+2]
                    parsed_kpts.append((int(kx), int(ky), vis))

                    # Draw keypoint joint if visible
                    if vis > 0:
                        cv2.circle(frame, (int(kx), int(ky)), 4, (0, 0, 255), -1)

                # Draw skeleton lines
                for p1_idx, p2_idx in self.SKELETON_CONNECTIONS:
                    if p1_idx < len(parsed_kpts) and p2_idx < len(parsed_kpts):
                        pt1, pt2 = parsed_kpts[p1_idx], parsed_kpts[p2_idx]
                        if pt1[2] > 0 and pt2[2] > 0:  # Both keypoints visible
                            cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (255, 255, 0), 2)

        return frame

    def _draw_behavior_overlay(
        self,
        frame: cv2.Mat,
        frame_idx: int,
        coco_data: Dict[str, Any],
        predictions_dir: str
    ) -> cv2.Mat:
        """
        Draws behavior labels (Ground Truth & Predicted Behaviors) alongside bounding boxes.
        """
        # First draw base pose annotations
        frame = self._draw_pose_overlay(frame, frame_idx, coco_data)

        # Overlay behavior prediction labels over the bounding boxes
        image_id = frame_idx + 1
        annotations = [ann for ann in coco_data.get("annotations", []) if ann.get("image_id") == image_id]

        for ann in annotations:
            bbox = ann.get("bbox", [])
            pred_label = ann.get("predicted_behavior", "Feeding")  # Fetch predicted behavior label
            gt_label = ann.get("gt_behavior", "N/A")

            if len(bbox) == 4:
                x, y, w, h = map(int, bbox)
                
                # Display Prediction & Ground Truth badges
                behavior_text = f"Pred: {pred_label}"
                cv2.putText(frame, behavior_text, (x, y + h + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        return frame

    def _convert_to_web_mp4(self, input_path: Path, output_path: Path) -> None:
        """
        Converts a raw MP4 file into an H.264 / AAC web-compatible MP4 using FFmpeg.
        """
        command = [
            "ffmpeg",
            "-y",  # Overwrite output file if exists
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart",  # Optimizes file for fast web streaming
            str(output_path)
        ]
        
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e}")
            raise RuntimeError(f"Failed to transcode video to web-compatible format: {e}")


# Singleton instance ready for importation
video_render_service = VideoRenderService(fps=1)