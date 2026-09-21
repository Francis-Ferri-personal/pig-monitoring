import os
import sys
import glob
import json
import gc
import torch
import re
import time

# OpenCV 4.10's ffmpeg backend aborts a stream after a fixed 32s decode stall
# (e.g. when the machine is under heavy CPU load). Bump the retry budget so
# transient stalls do not abort the whole clip. Must be set before cv2 is used.
os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "8192")

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# Ensure imports work regardless of where the script is called from
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.append(project_root)

# Add SAM 3 repo to path for internal package resolution
sam3_repo_path = os.path.join(project_root, 'sam3')
if sam3_repo_path not in sys.path:
    sys.path.insert(0, sam3_repo_path)

from sam3.model_builder import build_sam3_video_predictor
from utils.coco_utils import new_coco_data, append_frame_to_coco, save_coco_to_json

def natural_key(name):
    """Sort key that orders embedded numbers numerically: video1, video2, video10."""
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]

def load_mask(mask_path):
    """Load a static mask as a binary (0/255) grayscale image."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask at {mask_path}")
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

def log_memory(stage, clip_id):
    """Print current host RAM + GPU memory usage (helps catch OOM early)."""
    rss_mb = 0
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS'):
                rss_mb = int(line.split()[1]) // 1024
                break
    line = f"  [mem] {stage} clip {clip_id}: host RSS={rss_mb} MB"
    if torch.cuda.is_available():
        line += (f" | GPU used={torch.cuda.memory_allocated() // 1024**2} MB "
                 f"reserved={torch.cuda.memory_reserved() // 1024**2} MB")
    print(line, flush=True)

def make_chunks(total, chunk_frames, chunk_overlap):
    """
    Split `total` frames into list of (start, end) ranges of up to `chunk_frames`
    frames each, with `chunk_overlap` frames shared between consecutive chunks.

    Overlap is used to keep track identity continuous across chunks: objects are
    matched in the shared frames (by box IoU) before merging the COCO outputs.
    Disable chunking by passing chunk_frames=None or >= total.
    """
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
    """
    Map the current chunk's raw SAM3 track IDs onto the global track IDs.

    `lead_by_frame`:  {global_frame: {raw_id: [x,y,w,h]}} for the overlap frames that
                      open the current chunk (start of the chunk).
    `trail_by_frame`: {global_frame: {global_id: [x,y,w,h]}} for the same frames as
                      seen in the PREVIOUS chunk (its trailing overlap).
    Both dicts are keyed by the SAME global video frames, so per-frame IoU matching
    is meaningful. Track IDs present in the overlap that do not match any previous
    object, and IDs that only appear outside the overlap, get fresh global IDs.

    Returns:
        (mapping dict raw_id -> global_id, updated track_counter)
    """
    mapping = {}
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

    # raw IDs in the order they first appear (inside the overlap region)
    lead_order = []
    seen = set()
    for gf in sorted(lead_by_frame):
        for raw_id in lead_by_frame[gf]:
            if raw_id not in seen:
                seen.add(raw_id)
                lead_order.append(raw_id)

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

    # IDs that never appear in the overlap region get new global IDs
    for raw_id in raw_ids_in_chunk:
        if raw_id not in mapping:
            mapping[raw_id] = track_counter
            track_counter += 1

    return mapping, track_counter

def run_chunk_session(predictor, chunk_pils, chunk_start, chunk_end, overlap,
                      is_first, is_last, trail_by_frame, track_counter, prompt_text,
                      coco_data, global_img_id_offset, total_frames, ann_id, pbar):
    """
    Runs one SAM3 session over a chunk of PIL frames (chunk_start..chunk_end-1),
    streams the outputs into `coco_data` incrementally, and merges the chunk's raw
    track IDs into the global ID space using the overlap region shared with the
    previous chunk.

    Overlap frames that were already annotated by the previous chunk are skipped
    here (they are only used for track-ID matching).

    Returns (ann_id, track_counter, trail_by_frame_for_next_chunk).
    Raises on any failure; the caller is responsible for checkpoint cleanup.
    """
    session_id = None
    chunk_len = chunk_end - chunk_start
    lead_by_frame = {}      # opening overlap of THIS chunk -> {raw_id: box}
    trail_capture = {}      # closing overlap of THIS chunk -> {raw_id: box}
    raw_ids_in_chunk = set()
    ann_start = len(coco_data["annotations"])

    try:
        response = predictor.handle_request(dict(type="start_session", resource_path=chunk_pils))
        session_id = response["session_id"]

        # The masked frames have been copied to the model tensor; free this
        # chunk's PIL list before propagation to keep host RAM low.
        del chunk_pils

        predictor.handle_request(dict(type="reset_session", session_id=session_id))

        # Add prompt on frame 0 of this chunk
        predictor.handle_request(dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt_text
        ))

        for resp in predictor.handle_stream_request(dict(
            type="propagate_in_video",
            session_id=session_id
        )):
            rel_idx = resp["frame_index"]
            gf = chunk_start + rel_idx          # global frame index (within clip)
            outputs = resp["outputs"]

            # Remember boxes for the overlap regions (cheap; used only to keep
            # track IDs consistent across chunk borders).
            obj_ids = outputs.get("out_obj_ids", [])
            boxes = outputs.get("out_boxes_xywh", [])
            for oid, bx in zip(obj_ids, boxes):
                oid = int(oid)
                b = [float(x) for x in bx]
                raw_ids_in_chunk.add(oid)
                if not is_first and rel_idx < overlap:
                    lead_by_frame.setdefault(gf, {})[oid] = b
                if not is_last and rel_idx >= chunk_len - overlap:
                    trail_capture.setdefault(gf, {})[oid] = b

            # Overlap frames were already annotated by the previous chunk: skip
            # them here (they are only used for ID matching).
            is_dup_overlap = (not is_first and rel_idx < overlap)
            if not is_dup_overlap:
                img_id = global_img_id_offset + gf
                prev_image_id = img_id - 1 if gf > 0 else -1
                next_image_id = img_id + 1 if gf < total_frames - 1 else -1
                ann_id = append_frame_to_coco(
                    coco_data, outputs, gf, img_id, ann_id,
                    prev_image_id=prev_image_id, next_image_id=next_image_id,
                )

            # Release the raw masks before moving on (they are the main host-RAM
            # consumer in this loop).
            del outputs
            del resp
            pbar.update(1)

        # Merge this chunk's raw track IDs into the global ID space and fix the
        # annotations that were just written by this chunk.
        mapping, track_counter = compute_track_mapping(
            lead_by_frame, trail_by_frame, raw_ids_in_chunk, track_counter
        )
        for a in coco_data["annotations"][ann_start:]:
            a["track_id"] = mapping[a["track_id"]]

        # Prepare overlap reference for the next chunk (using global IDs).
        new_trail = {}
        if not is_last:
            for gf, objs in trail_capture.items():
                new_trail[gf] = {mapping[rid]: b for rid, b in objs.items()}

        # Close this chunk's session to free its GPU resources.
        predictor.handle_request(dict(type="close_session", session_id=session_id))
        session_id = None

        return ann_id, track_counter, new_trail
    finally:
        if session_id is not None:
            predictor.handle_request(dict(type="close_session", session_id=session_id))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def read_masked_frames(video_path, mask):
    """
    Reads all frames from a clip video and applies the static mask in memory
    (
    Returns the list of masked PIL frames and their frame IDs (0, 1, 2, ...).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! Warning: Could not open video {video_path}")
        return [], []

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Resize mask if necessary
    if mask.shape[:2] != (height, width):
        mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask

    # Convert grayscale mask to 3-channel
    if len(mask_resized.shape) == 2:
        mask_3d = cv2.merge([mask_resized, mask_resized, mask_resized])
    else:
        mask_3d = mask_resized

    pil_frames = []
    pbar = tqdm(total=total_frames, desc=f"  Reading frames {os.path.basename(os.path.dirname(video_path))}/{os.path.basename(video_path)}", unit="frame")

    last_good = -1  # last successfully read frame index
    fail_count = 0
    MAX_FAILS = 50   # number of consecutive retries before giving up
    fail_sleep = 0.3 # seconds to wait before retrying a stalled decode

    while True:
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count > MAX_FAILS:
                break
            # Rewind to the next expected frame and give the decoder some time
            target = last_good + 1
            if target >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            time.sleep(fail_sleep)
            continue

        # Successful read – reset the failure counter
        fail_count = 0
        last_good += 1

        masked_frame = cv2.bitwise_and(frame, mask_3d)
        frame_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(frame_rgb))
        pbar.update(1)

    pbar.close()
    cap.release()

    frame_ids = [str(i) for i in range(len(pil_frames))]
    return pil_frames, frame_ids

def generate_annotations(prompt_text="pig", video_filter=None, mask_path="data/images/mask.png", clip_filter=None, gpus=None, chunk_frames=None, chunk_overlap=60, clips_root=None, out_root=None):
    """
    Processes all videos and their clips found in the clips directory, applying the
    static mask in memory (no need for pre-masked videos or extracted frames).

    Args:
        clips_root: Optional custom input directory containing <video>/<clip>.mp4
                    hierarchies (default: data/videos/clips).
        out_root:   Optional custom output directory for the COCO JSONs
                    (default: data/annotations/sam).
    """
    # 1. Load mask once
    print(f">>> Loading mask from {mask_path}...")
    mask = load_mask(mask_path)

    # 2. Initialize predictor once (default: single GPU to avoid multi-GPU worker noise/hangs)
    print(">>> Initializing SAM 3 Predictor...")
    if gpus is None:
        gpus = [torch.cuda.current_device()]
    predictor = build_sam3_video_predictor(gpus_to_use=gpus)
    print(f">>> Using GPU(s): {gpus}")

    try:
        _process_videos(predictor, mask, prompt_text=prompt_text,
                        video_filter=video_filter, clip_filter=clip_filter,
                        chunk_frames=chunk_frames, chunk_overlap=chunk_overlap,
                        clips_root=clips_root, out_root=out_root)
    except KeyboardInterrupt:
        print("\n>>> Interrupted by user. Shutting down predictor cleanly...")
    finally:
        predictor.shutdown()
        print(">>> SAM 3 predictor shutdown complete.\n")

def _process_videos(predictor, mask, prompt_text="pig", video_filter=None, clip_filter=None, chunk_frames=None, chunk_overlap=60, clips_root=None, out_root=None):
    """Run the annotation pipeline across videos/clips using the given predictor."""
    clips_base_root = clips_root or "data/videos/clips"
    output_base_root = out_root or "data/annotations/sam"
    if not os.path.exists(clips_base_root):
        print(f"Error: Base directory {clips_base_root} not found.")
        return

    # 3. Find all video directories
    all_video_dirs = sorted([d for d in os.listdir(clips_base_root) 
                            if os.path.isdir(os.path.join(clips_base_root, d))], key=natural_key)
    
    if video_filter:
        video_dirs = [d for d in all_video_dirs if d == video_filter]
    else:
        if clip_filter:
            print("Error: --clip requires --video to be specified.")
            return
        video_dirs = all_video_dirs
    
    if not video_dirs:
        print("No video directories found matching the filter in data/videos/clips.")
        return

    print(f">>> Found {len(video_dirs)} videos: {', '.join(video_dirs)}")

    # 4. Process each video (video_id is auto-incremental, starting at 0)
    video_counter = 0
    for video_dir_name in video_dirs:
        video_id = video_counter
        video_counter += 1

        clips_root = os.path.join(clips_base_root, video_dir_name)
        output_base_dir = os.path.join(output_base_root, video_dir_name)
        os.makedirs(output_base_dir, exist_ok=True)

        # Get and sort clips (01, 02, etc.)
        clips = sorted([os.path.splitext(f)[0] for f in os.listdir(clips_root) 
                       if f.lower().endswith(".mp4")], key=natural_key)

        # Filter to a single clip if requested
        if clip_filter:
            if clip_filter in clips:
                clips = [clip_filter]
            else:
                print(f"  ! Clip {clip_filter} not found in video {video_dir_name}. "
                      f"Available clips: {', '.join(clips)}")
                continue
        
        # Count clips already generated (for resume progress)
        clips_done = sum(1 for c in clips
                        if os.path.exists(os.path.join(output_base_dir, f"{c}.json")))
        
        # Reset counters for each NEW video
        global_img_id_offset = 0
        global_ann_id_offset = 0

        print(f"\n" + "="*50)
        print(f" STARTED VIDEO: {video_dir_name} (ID: {video_id})")
        print(f" Clips already done: {clips_done}/{len(clips)} (pending: {len(clips)-clips_done})")
        print("="*50 + "\n")
        
        for i, clip_id in enumerate(clips):
            output_json = os.path.join(output_base_dir, f"{clip_id}.json")
            
            # --- RESUME LOGIC ---
            if os.path.exists(output_json):
                try:
                    with open(output_json, 'r') as f:
                        existing_data = json.load(f)
                    
                    num_frames = len(existing_data.get('images', []))
                    existing_anns = existing_data.get('annotations', [])
                    max_ann_ptr = global_ann_id_offset
                    if existing_anns:
                        max_ann_ptr = max(ann['id'] for ann in existing_anns)
                    
                    # Update offsets as if we had processed it
                    global_img_id_offset += num_frames
                    global_ann_id_offset = max_ann_ptr
                    
                    print(f"[{i+1}/{len(clips)}] Skipping Clip: {clip_id} (Already exists. Resuming offsets...)")
                    continue
                except Exception as e:
                    print(f"[{i+1}/{len(clips)}] Error reading existing {output_json}: {e}. Will re-process.")

            # --- STANDARD PROCESSING ---
            print(f"[{i+1}/{len(clips)}] Processing Clip: {clip_id} ...", flush=True)
            
            clip_video_path = os.path.join(clips_root, f"{clip_id}.mp4")
            video_frames_masked, frame_ids = read_masked_frames(clip_video_path, mask)
            
            if not video_frames_masked:
                print(f"  ! Warning: Could not read frames from {clip_id}. Skipping.")
                continue

            # SAM 3 session logic (pass frames in memory, no temp files)
            chunks = None
            chunk_slices = None
            chunk_pils = None
            coco_data = None
            pbar = None
            try:
                # Split the clip into smaller chunks so each SAM3 session only keeps
                # a fraction of the frames on GPU (peak GPU memory scales with the
                # number of frames per session). Consecutive chunks share an overlap
                # region used to keep track IDs continuous.
                if chunk_frames and chunk_overlap >= chunk_frames:
                    chunk_overlap = chunk_frames // 2
                    print(f"  ! Shrinking chunk overlap to {chunk_overlap} (must be < chunk-frames).", flush=True)
                chunks = make_chunks(len(video_frames_masked), chunk_frames, chunk_overlap)
                chunk_slices = [video_frames_masked[s:e] for s, e in chunks]
                del video_frames_masked
                video_frames_masked = None

                num_chunks = len(chunks)
                print(f"  Processing {len(frame_ids)} frames in {num_chunks} chunk(s) "
                      f"({', '.join(f'{s}-{e-1}' for s, e in chunks)})...", flush=True)

                coco_data = new_coco_data(
                    video_id=video_id,
                    video_name=clip_id,
                    category_name="pig",
                    super_category="animal",
                )
                ann_id = global_ann_id_offset
                total_frames = len(frame_ids)
                checkpoint_path = output_json + ".part"
                track_counter = 0
                trail_by_frame = {}   # global frame -> {global_id: box} of previous chunk
                pbar = tqdm(desc=f"  SAM3 {video_dir_name}/{clip_id}", total=total_frames, unit="frame")

                for ci, ((chunk_start, chunk_end), chunk_pils) in enumerate(zip(chunks, chunk_slices)):
                    is_first = (ci == 0)
                    is_last = (ci == num_chunks - 1)
                    overlap = chunk_overlap if (chunk_frames and chunk_frames < total_frames) else 0
                    print(f"    Chunk {ci+1}/{num_chunks}: frames {chunk_start}-{chunk_end-1} ...", flush=True)
                    log_memory(f"chunk {ci+1} start", clip_id)

                    ann_id, track_counter, trail_by_frame = run_chunk_session(
                        predictor=predictor,
                        chunk_pils=chunk_pils,
                        chunk_start=chunk_start,
                        chunk_end=chunk_end,
                        overlap=overlap,
                        is_first=is_first,
                        is_last=is_last,
                        trail_by_frame=trail_by_frame,
                        track_counter=track_counter,
                        prompt_text=prompt_text,
                        coco_data=coco_data,
                        global_img_id_offset=global_img_id_offset,
                        total_frames=total_frames,
                        ann_id=ann_id,
                        pbar=pbar,
                    )
                    del chunk_pils

                    # Checkpoint after each chunk so a crash mid-clip doesn't lose
                    # everything; only the final JSON counts as "done" for resume.
                    save_coco_to_json(coco_data, checkpoint_path)
                    pbar.set_postfix_str(f"anns={len(coco_data['annotations'])}")
                    log_memory(f"chunk {ci+1} done", clip_id)
                    gc.collect()

                pbar.close()

                # Save final (atomic-ish: write to temp then rename).
                save_coco_to_json(coco_data, checkpoint_path)
                os.replace(checkpoint_path, output_json)
                last_ann_id = ann_id

                # Update offsets
                global_img_id_offset += len(frame_ids)
                global_ann_id_offset = last_ann_id

                log_memory("after save", clip_id)
                print(f"  ✓ Clip {clip_id} Done. ({len(frame_ids)} frames, {len(coco_data['annotations'])} anns. Total frames: {global_img_id_offset})\n")

            except Exception as e:
                print(f"  ✗ Error processing clip {clip_id}: {e}")
                # Remove the stale checkpoint: it is not a valid annotation file
                # and would only confuse (resume regenerates it on the next run).
                part_path = output_json + ".part"
                if os.path.exists(part_path):
                    os.remove(part_path)
                print(f"  → Clip {clip_id} left pending; it will be retried on the next run.", flush=True)
            
            finally:
                if pbar is not None:
                    pbar.close()
                # These containers retain every PIL frame until their references
                # are cleared; otherwise the next clip is read alongside this one.
                chunk_pils = None
                chunk_slices = None
                chunks = None
                video_frames_masked = None
                frame_ids = None
                coco_data = None
                trail_by_frame = None
                gc.collect()
                if torch.cuda.is_available():
                    # Return reserved-but-unused GPU memory to the pool so other
                    # clips / tenants can use it while the model reloads.
                    torch.cuda.empty_cache()

        print(f"\n>>> COMPLETED Video {video_dir_name}. Final Annotation ID: {global_ann_id_offset}")

    print("\n>>> ALL VIDEOS PROCESSED SUCCESSFULLY.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate COCO annotations for all available videos and clips.")
    parser.add_argument("--prompt", type=str, default="pig", help="Text prompt for SAM 3 (default: 'pig')")
    parser.add_argument("--video", type=str, default=None, help="Process only the specified video folder name, e.g., May_25_01.")
    parser.add_argument("--clip", type=str, default=None, help="Process only the specified clip number, e.g., 01 (requires --video).")
    parser.add_argument("--mask", type=str, default="data/images/mask.png", help="Path to mask PNG (default: data/images/mask.png)")
    parser.add_argument("--gpus", type=int, nargs="+", default=None, help="GPU ID(s) to use (default: single GPU)")
    parser.add_argument("--chunk-frames", type=int, default=None, help="Process clips in chunks of N frames to reduce peak GPU memory (e.g. 300 for a 900-frame clip). Default: no chunking.")
    parser.add_argument("--chunk-overlap", type=int, default=60, help="Overlap frames between consecutive chunks, used to keep track IDs continuous (default: 60).")
    parser.add_argument("--clips-root", type=str, default=None, help="Custom input directory with <video>/<clip>.mp4 (default: data/videos/clips).")
    parser.add_argument("--out-root", type=str, default=None, help="Custom output directory for the COCO JSONs (default: data/annotations/sam).")
    
    args = parser.parse_args()
    generate_annotations(prompt_text=args.prompt, video_filter=args.video, mask_path=args.mask, clip_filter=args.clip, gpus=args.gpus, chunk_frames=args.chunk_frames, chunk_overlap=args.chunk_overlap, clips_root=args.clips_root, out_root=args.out_root)