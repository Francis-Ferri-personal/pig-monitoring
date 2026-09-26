import os
import logging
import shutil
from pathlib import Path
import cv2
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Normalization targets for the SAM pipeline (clip is normalized to this
# resolution and fps before tracking, matching the offline CLI behaviour).
DEFAULT_RESOLUTION = (1920, 1080)
DEFAULT_FPS = 5


class VideoPrepService:
    """Normalizes an uploaded video (resolution + frame rate) before SAM.

    Unlike the old sampling service it does NOT extract per-frame JPEGs: the
    prepared MP4 is the single source of pixel truth and every downstream stage
    (SAM, pose, features, rendering) reads frames from it on demand.

    By default the video is re-encoded to 1920x1080 @ 5 fps (same as the CLI
    pipeline). The resolution / fps can be overridden per-instance.
    """

    def __init__(self, resolution: tuple = DEFAULT_RESOLUTION, fps: float = DEFAULT_FPS):
        self.resolution = tuple(resolution)
        self.fps = float(fps)
        logging.info(
            f"Video Prep Service initialized (target {self.resolution[0]}x{self.resolution[1]} @ {self.fps} fps)."
        )

    def prepare(self, video_path: str, output_path: str = None) -> str:
        """Resize + re-time `video_path` to the configured resolution and fps.

        Returns the path of the prepared MP4: either `output_path` when given
        (copying it there if the source already matches the target), or the
        same folder with a `_prepared` suffix.
        """
        logging.info(f"Preparing video for SAM: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open source video: {video_path}")

        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        logging.info(f"Source -> {src_w}x{src_h}, fps={src_fps:.2f}, frames={total_frames}")

        original_path = Path(video_path)
        if (src_w, src_h) == self.resolution and abs(src_fps - self.fps) < 0.5:
            logging.info("Source already matches target size and fps; skipping re-encode.")
            if output_path is not None:
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    shutil.copyfile(video_path, str(out))
                return str(out)
            return video_path

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            output_path = str(out)
        else:
            output_path = str(original_path.with_name(f"{original_path.stem}_prepared{original_path.suffix}"))
        tw, th = self.resolution

        command = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"scale={tw}:{th}:flags=lanczos",
            "-r", str(self.fps),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-an", "-movflags", "+faststart",
            str(output_path),
        ]
        logging.info(f"Re-encoding to {tw}x{th} @ {self.fps} fps with ffmpeg ...")
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        out_cap = cv2.VideoCapture(output_path)
        out_w = int(out_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(out_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_fps = out_cap.get(cv2.CAP_PROP_FPS)
        out_frames = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_cap.release()
        logging.info(
            f"Prepared -> {out_w}x{out_h}, fps={out_fps:.2f}, frames={out_frames}: {output_path}"
        )
        return output_path
