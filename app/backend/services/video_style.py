"""Shared drawing and encoding helpers for annotated pig videos."""

import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple

import cv2
import numpy as np
from pycocotools import mask as mask_utils


# Pig pose topology from data/pig_pose.yaml, converted from 1-based COCO indices.
SKELETON_CONNECTIONS: Sequence[Tuple[int, int]] = (
    (0, 1), (0, 2), (1, 2), (2, 3), (3, 4),
    (3, 5), (5, 6), (6, 7), (3, 8), (8, 9), (9, 10),
    (4, 11), (11, 12), (12, 13), (4, 14), (14, 15), (15, 16),
)

# High-contrast BGR colours, deterministic by tracking ID.
TRACK_COLORS: Sequence[Tuple[int, int, int]] = (
    (0, 255, 0), (0, 170, 255), (255, 80, 0), (255, 0, 255),
    (0, 255, 255), (180, 80, 255), (255, 255, 0), (80, 255, 120),
)
PREDICTION_COLOR = (255, 255, 255)


def _track_color(track_id: Any) -> Tuple[int, int, int]:
    try:
        return TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]
    except (TypeError, ValueError):
        return TRACK_COLORS[0]


def _draw_mask(frame: cv2.Mat, segmentation: Any, color: Tuple[int, int, int]) -> None:
    """Blend a COCO RLE mask over a frame; invalid masks do not break rendering."""
    if not isinstance(segmentation, Mapping) or "counts" not in segmentation:
        return
    try:
        mask = mask_utils.decode(segmentation)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2)
        mask = mask.astype(bool)
        if mask.shape != frame.shape[:2]:
            return
        overlay = np.empty_like(frame)
        overlay[:] = color
        frame[mask] = cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)[mask]
    except (TypeError, ValueError):
        return


def draw_pose_annotations(
    frame: cv2.Mat, annotations: Iterable[Mapping[str, Any]], draw_keypoints: bool = True
) -> cv2.Mat:
    """Draw coloured SAM masks, tracking IDs, pig skeletons and keypoints in-place."""
    for annotation in annotations:
        track_id = annotation.get("track_id", annotation.get("id", "N/A"))
        color = _track_color(track_id)
        _draw_mask(frame, annotation.get("segmentation"), color)

        bbox = annotation.get("bbox", [])
        if len(bbox) == 4:
            x, y, width, height = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)
            label = f"ID: {track_id}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            label_top = max(0, y - text_height - 10)
            cv2.rectangle(frame, (x, label_top), (x + text_width + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, max(text_height + 2, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        if draw_keypoints:
            raw_keypoints = annotation.get("keypoints", [])
            keypoints = []
            for index in range(0, len(raw_keypoints) - 2, 3):
                keypoints.append(tuple(raw_keypoints[index : index + 3]))

            for first_index, second_index in SKELETON_CONNECTIONS:
                if first_index >= len(keypoints) or second_index >= len(keypoints):
                    continue
                first_point, second_point = keypoints[first_index], keypoints[second_index]
                if first_point[2] > 0 and second_point[2] > 0:
                    cv2.line(frame, (int(first_point[0]), int(first_point[1])), (int(second_point[0]), int(second_point[1])), color, 3)

            for keypoint_x, keypoint_y, visibility in keypoints:
                if visibility <= 0:
                    continue
                point = (int(keypoint_x), int(keypoint_y))
                cv2.circle(frame, point, 6, (0, 0, 0), -1)
                cv2.circle(frame, point, 4, (0, 255, 255) if visibility > 1 else (0, 0, 255), -1)

    return frame


def draw_prediction_label(frame: cv2.Mat, bbox: Sequence[float], prediction: str) -> cv2.Mat:
    """Draw a readable prediction badge below a bounding box."""
    if len(bbox) == 4:
        x, y, width, height = map(int, bbox)
        label = f"Pred: {prediction}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = min(frame.shape[0] - 4, y + height + text_height + 12)
        cv2.rectangle(frame, (x, label_y - text_height - 8), (x + text_width + 10, label_y + 4), (30, 30, 30), -1)
        cv2.putText(frame, label, (x + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PREDICTION_COLOR, 2)
    return frame


def convert_to_web_mp4(input_path: Path, output_path: Path) -> None:
    """Encode an OpenCV MP4 as browser-compatible H.264 with fast start."""
    command = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-movflags", "+faststart", str(output_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Failed to transcode video to web-compatible format: {error}") from error
