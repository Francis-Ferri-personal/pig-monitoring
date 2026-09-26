import contextlib
import gc
import logging
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask_utils

# sam3 repo (submodule) must be resolvable.
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_sam3_path = os.path.join(_project_root, "sam3")
if _sam3_path not in sys.path:
    sys.path.insert(0, _sam3_path)

from sam3.model_builder import build_sam3_video_predictor  # noqa: E402

from utils.format import sam_to_coco  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# Chunking helpers (kept behaviourally identical to the offline CLI, tools/gen_anns.py)
# ---------------------------------------------------------------------------
def make_chunks(total, chunk_frames, chunk_overlap):
    """Split `total` frames into (start, end) ranges of up to `chunk_frames`,
    with `chunk_overlap` shared frames between consecutive chunks."""
    if not chunk_frames or chunk_frames <= 0 or chunk_frames >= total:
        return [(0, total)]
    step = max(chunk_frames - chunk_overlap, 1)
    chunks = []
    start = 0
    while start < total:
        end = min(start + chunk_frames, total)
        chunks.append((start, end))
        if end >= total:
            break
        start = end - chunk_overlap
    return chunks


def iou_xywh_norm(b1, b2):
    """IoU between two normalized [x, y, w, h] boxes."""
    xa = max(b1[0], b2[0]); ya = max(b1[1], b2[1])
    xb = min(b1[0] + b1[2], b2[0] + b2[2])
    yb = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    ua = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / ua if ua > 0 else 0.0


def compute_track_mapping(lead_by_frame, trail_by_frame, raw_ids_in_chunk, track_counter):
    """Map a chunk's raw SAM track IDs onto the global track-ID space using the
    overlap region shared with the previous chunk (matched by box IoU)."""
    scores = {}
    for gf, lead_objs in lead_by_frame.items():
        trail_objs = trail_by_frame.get(gf, {})
        for raw_id, box in lead_objs.items():
            for gid, gbox in trail_objs.items():
                key = (raw_id, gid)
                s = iou_xywh_norm(box, gbox)
                cur = scores.get(key)
                if cur is None:
                    scores[key] = [s, 1]
                else:
                    cur[0] += s
                    cur[1] += 1

    lead_order = []
    seen = set()
    for gf in sorted(lead_by_frame):
        for raw_id in lead_by_frame[gf]:
            if raw_id not in seen:
                seen.add(raw_id)
                lead_order.append(raw_id)

    mapping = {}
    used_gids = set()
    for raw_id in lead_order:
        best_gid, best_mean = None, 0.0
        for (rid, gid), (tot, n) in scores.items():
            if rid == raw_id and gid not in used_gids:
                mean = tot / n
                if mean > best_mean:
                    best_mean, best_gid = mean, gid
        if best_gid is not None and best_mean >= 0.5:
            mapping[raw_id] = best_gid
            used_gids.add(best_gid)
        else:
            mapping[raw_id] = track_counter
            track_counter += 1

    for raw_id in raw_ids_in_chunk:
        if raw_id not in mapping:
            mapping[raw_id] = track_counter
            track_counter += 1

    return mapping, track_counter


def _remap_frame_outputs(outputs, mapping):
    """Return a copy of `outputs` with raw obj ids remapped to global ids (array order kept)."""
    new = dict(outputs)
    ids = outputs.get("out_obj_ids")
    if ids is not None and len(ids) > 0:
        new["out_obj_ids"] = [mapping.get(int(i), int(i)) for i in ids]
    return new


def _frame_count(video_path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def _read_frames_range(video_path, start: int, count: int):
    """Read `count` frames starting at `start` into RGB PIL images.

    Bounded per-chunk reading: peak memory stays at ~chunk_size frames instead
    of the whole video (the old eager full-list load hit GB-scale RAM usage).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    pil_frames = []
    for _ in range(count):
        ret, frame = cap.read()
        if not ret:
            break
        pil_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return pil_frames


def _encode_mask_rle(mask) -> dict:
    """Encode a binary mask (torch tensor or array) into a small COCO RLE dict."""
    if hasattr(mask, "detach"):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    rle = coco_mask_utils.encode(np.asfortranarray(arr.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _slim_frame_outputs(out: dict) -> dict:
    """Return `out` with full-resolution masks replaced by (tiny) RLE encodings.

    Keeping binary (N, H, W) masks per frame across the whole video uses
    several GB of RAM; the RLE encoding is a few KB per object and is exactly
    what the COCO export (and the mask consumers) need.
    """
    slim = {k: v for k, v in out.items() if k != "out_binary_masks"}
    masks = out.get("out_binary_masks")
    n_obj = len(out.get("out_obj_ids", []))
    rles = []
    if masks is not None:
        try:
            n_mask = len(masks)
        except TypeError:
            n_mask = 0
        for i in range(min(n_obj, n_mask)):
            try:
                rles.append(_encode_mask_rle(masks[i]))
            except Exception as e:
                logging.warning(f"  Failed to RLE-encode mask {i}: {e}")
                rles.append(None)
    slim["out_rles"] = rles
    return slim


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class SamService:
    """SAM 3 video grounding service.

    Processes a (prepared) video in bounded chunks so peak GPU memory grows
    with the chunk size rather than the full clip length, while keeping track
    IDs continuous across chunk boundaries (IoU matching in the overlap region).
    """

    def __init__(self, default_chunk_frames: int = 240, chunk_overlap: int = 60):
        logging.info("Initializing SAM Video Predictor Service...")
        # Use a single GPU: the offline pipeline deliberately avoids multi-GPU workers
        # (NCCL worker/heartbeat hangs when sessions are opened and closed repeatedly).
        self.gpus_to_use = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        if not self.gpus_to_use:
            logging.warning("No CUDA GPUs detected. Running on CPU might be extremely slow.")

        self.default_chunk_frames = default_chunk_frames
        self.default_chunk_overlap = chunk_overlap
        self.predictor = build_sam3_video_predictor(gpus_to_use=self.gpus_to_use)
        logging.info(f"SAM Predictor successfully loaded on GPUs: {self.gpus_to_use}")
        logging.info(f"Chunking: {self.default_chunk_frames} frames/chunk, overlap {self.default_chunk_overlap}.")

        # SAM3's tracker enables torch.autocast(bf16) thread-locally at model
        # construction (sam3_tracking_predictor.py). autocast state lives on the
        # thread stack, so workers running on other threads must (re)enter the
        # same context or conv layers receive bf16 inputs with fp32 weights.
        self.autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else contextlib.nullcontext()
        )

    def process_video(
        self,
        video_path: str,
        video_name: str,
        prompt_text: str = "pig",
        chunk_frames: int = None,
        chunk_overlap: int = None,
    ) -> dict:
        """Run SAM tracking over `video_path` and return a COCO dict (in memory)."""
        if chunk_frames is None:
            chunk_frames = self.default_chunk_frames
        if chunk_overlap is None:
            chunk_overlap = self.default_chunk_overlap

        total = _frame_count(video_path)
        if total == 0:
            raise ValueError(f"No frames could be read from {video_path}")

        if chunk_overlap >= chunk_frames:
            chunk_overlap = max(1, chunk_frames // 2)
        chunks = make_chunks(total, chunk_frames, chunk_overlap)

        logging.info(f"SAM processing {total} frames in {len(chunks)} chunk(s) "
                     f"({', '.join(f'{s}-{e-1}' for s, e in chunks)}).")

        outputs_per_frame = {}
        track_counter = 0
        trail_by_frame = {}

        with self.autocast_ctx:
            for ci, (start, end) in enumerate(chunks):
                logging.info(f"  [SAM chunk {ci+1}/{len(chunks)}] frames {start}-{end-1} ...")
                chunk_pils = _read_frames_range(video_path, start, end - start)
                is_first = (ci == 0)
                is_last = (ci == len(chunks) - 1)
                overlap = chunk_overlap if (chunk_frames < total) else 0
                chunk_len = end - start

                sid = None
                try:
                    response = self.predictor.handle_request(dict(type="start_session", resource_path=chunk_pils))
                    sid = response["session_id"]
                    del chunk_pils
                    self.predictor.handle_request(dict(type="reset_session", session_id=sid))
                    self.predictor.handle_request(dict(
                        type="add_prompt", session_id=sid, frame_index=0, text=prompt_text
                    ))

                    chunk_outputs = {}
                    lead_by_frame = {}
                    trail_cap = {}
                    raw_ids_in_chunk = set()

                    for resp in self.predictor.handle_stream_request(dict(
                        type="propagate_in_video", session_id=sid
                    )):
                        rel = resp["frame_index"]
                        gf = start + rel
                        out = resp["outputs"]
                        chunk_outputs[rel] = _slim_frame_outputs(out)

                        obj_ids = out.get("out_obj_ids", [])
                        boxes = out.get("out_boxes_xywh", [])
                        for oid, bx in zip(obj_ids, boxes):
                            oid = int(oid)
                            raw_ids_in_chunk.add(oid)
                            if not is_first and rel < overlap:
                                lead_by_frame.setdefault(gf, {})[oid] = [float(v) for v in bx]
                            if not is_last and rel >= chunk_len - overlap:
                                trail_cap.setdefault(gf, {})[oid] = [float(v) for v in bx]

                        if rel % 50 == 0 or rel == chunk_len - 1:
                            logging.info(f"    SAM tracking... frame {rel+1}/{chunk_len} computed.")

                    # Merge this chunk's raw IDs into the global space.
                    mapping, track_counter = compute_track_mapping(
                        lead_by_frame, trail_by_frame, raw_ids_in_chunk, track_counter
                    )

                    # Store remapped outputs (skip lead overlap already written by prev chunk).
                    for rel, out in chunk_outputs.items():
                        gf = start + rel
                        if not is_first and rel < overlap:
                            continue
                        outputs_per_frame[gf] = _remap_frame_outputs(out, mapping)

                    if not is_last:
                        new_trail = {}
                        for gf, objs in trail_cap.items():
                            new_trail[gf] = {mapping.get(rid, rid): b for rid, b in objs.items()}
                        trail_by_frame = new_trail

                finally:
                    if sid is not None:
                        self.predictor.handle_request(dict(type="close_session", session_id=sid))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        # Build the COCO dict from the globally consistent outputs.
        virtual_frame_paths = [f"frame_{i:05d}.jpg" for i in range(total)]
        logging.info("Converting output masks into COCO format data...")
        coco_data, _ = sam_to_coco(
            outputs_per_frame=outputs_per_frame,
            video_id=video_name,
            video_name=video_name,
            frame_paths=virtual_frame_paths,
        )
        logging.info(f"SAM pipeline finished. {len(coco_data['annotations'])} annotations, "
                     f"{len(coco_data['images'])} frames.")
        return coco_data


if __name__ == "__main__":
    print("\n--- STARTING DEBUG RUN FOR SamService ---")
