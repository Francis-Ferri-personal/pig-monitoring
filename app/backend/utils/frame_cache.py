"""Frame reader shared by every downstream stage (pose, features, rendering).

Frames are read straight out of the prepared video (1920x1080) by index and
cached in a small LRU so that consecutive requests for the same frame (features
and video rendering both need it) do not pay the decode cost twice.

Nothing is ever written to disk: the single source of truth for pixel data is
the prepared MP4 itself.
"""

import threading
from collections import OrderedDict
from typing import Optional, Tuple

import cv2

# (width, height) the prepared video is always generated at.
PREPARED_SIZE = (1920, 1080)

# LRU size: 8 frames is enough to cover the worst case where pose, features and
# rendering traverse the same video concurrently without thrashing memory.
_LRU_MAX = 8


class FrameReader:
    """Thread-safe reader for a single video file, by frame index."""

    def __init__(self, video_path: str, target_fps: float = None):
        self.video_path = str(video_path)
        self.target_fps = target_fps
        self._lock = threading.Lock()
        self._cap = None
        self._total_frames: Optional[int] = None
        self._cache: "OrderedDict[int, cv2.Mat]" = OrderedDict()

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._cache.clear()

    # -- metadata ------------------------------------------------------------
    def _ensure_cap(self):
        if self._cap is None:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video: {self.video_path}")
            self._cap = cap
            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self._cap

    @property
    def total_frames(self) -> int:
        self._ensure_cap()
        return self._total_frames

    @property
    def fps(self) -> float:
        cap = self._ensure_cap()
        fps = cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else float(self.target_fps or 30.0)

    # -- frame access --------------------------------------------------------
    def read(self, frame_index: int) -> Optional["cv2.Mat"]:
        """Return the BGR frame at `frame_index` (0-based) or None if out of range."""
        cap = self._ensure_cap()
        if frame_index < 0 or (self._total_frames is not None and frame_index >= self._total_frames):
            return None

        # Cache hit fast path.
        with self._lock:
            if frame_index in self._cache:
                self._cache.move_to_end(frame_index)
                return self._cache[frame_index]

        # Seek then read (held under the same lock: decode is a single blocking
        # call and this reader instance is used by one pipeline at a time).
        with self._lock:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            frame = frame.copy()
            self._cache[frame_index] = frame
            while len(self._cache) > _LRU_MAX:
                self._cache.popitem(last=False)
            return frame
