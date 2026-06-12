import os
import json
import numpy as np
import argparse
from glob import glob
from scipy.optimize import linear_sum_assignment

def get_persistent_tracks(annotations, image_ids, min_occupancy_ratio=0.15):
    """
    Identifies track_ids that are present in at least a minimum ratio of frames in the clip.
    This helps filter out short-lived false positive detections.
    
    Args:
        annotations (list): List of COCO annotations.
        image_ids (list): List of image IDs in the clip.
        min_occupancy_ratio (float): Minimum ratio of frames the track must appear in.
        
    Returns:
        set: Set of track_ids that are persistent.
    """
    total_frames = len(image_ids)
    if total_frames == 0:
        return set()
        
    # Count occurrences of each track_id in the clip
    track_counts = {}
    for ann in annotations:
        t_id = ann['track_id']
        track_counts[t_id] = track_counts.get(t_id, 0) + 1
        
    # Filter tracks by occupancy ratio
    persistent_tracks = {
        t_id for t_id, count in track_counts.items() 
        if (count / total_frames) >= min_occupancy_ratio
    }
    
    return persistent_tracks

def get_average_bboxes(annotations, image_ids, n_frames, persistent_tracks, from_start=True, decay_factor=0.3):
    """
    Computes the weighted average bounding box for each persistent track_id across N boundary frames.
    An exponential decay weight is applied, giving the boundary frame (closest to the split point)
    the highest weight.
    
    Args:
        annotations (list): List of COCO annotations.
        image_ids (list): List of image IDs sorted by frame order.
        n_frames (int): Number of boundary frames to consider.
        persistent_tracks (set): Set of valid persistent track IDs.
        from_start (bool): If True, takes first N frames (start of clip). Otherwise, takes last N frames (end of clip).
        decay_factor (float): Exponential decay factor (0.0 to 1.0). 
                              0.0 means only the boundary frame is used.
                              1.0 means all N frames are weighted equally.
                              
    Returns:
        dict: Mapping of track_id to its weighted average bbox [x, y, w, h].
    """
    # Select the target boundary subset of image IDs in correct chronological order
    target_list = image_ids[:n_frames] if from_start else image_ids[-n_frames:]
    target_image_ids = set(target_list)
    
    # Pre-calculate exponential decay weights for each image_id in target list
    # The boundary frame (index 0 if starting, index -1 if ending) should have weight 1.0 (decay^0)
    weights_dict = {}
    num_elements = len(target_list)
    for idx, img_id in enumerate(target_list):
        if from_start:
            # Boundary is at the start (index 0)
            distance = idx
        else:
            # Boundary is at the end (index num_elements - 1)
            distance = num_elements - 1 - idx
            
        weights_dict[img_id] = decay_factor ** distance
        
    # Group bounding boxes and their weights by track_id
    track_boxes = {}
    track_weights = {}
    for ann in annotations:
        t_id = ann['track_id']
        img_id = ann['image_id']
        if img_id in target_image_ids and t_id in persistent_tracks:
            bbox = ann['bbox'] # [x, y, w, h]
            w = weights_dict[img_id]
            
            if t_id not in track_boxes:
                track_boxes[t_id] = []
                track_weights[t_id] = []
                
            track_boxes[t_id].append(bbox)
            track_weights[t_id].append(w)
            
    # Calculate the weighted average [x, y, w, h] for each track_id
    averaged_tracks = {}
    for t_id in track_boxes.keys():
        bboxes = np.array(track_boxes[t_id])
        weights = np.array(track_weights[t_id])
        
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            # Compute weighted mean
            weighted_mean = np.sum(bboxes * weights[:, np.newaxis], axis=0) / sum_weights
            averaged_tracks[t_id] = weighted_mean.tolist()
        
    return averaged_tracks

def get_centroid(bbox):
    """
    Calculates the centroid coordinates of a bounding box.
    """
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)

def match_pigs_hungarian(prev_averaged_pigs, curr_averaged_pigs, max_distance_threshold=250.0):
    """
    Solves the optimal bipartite matching between previous canonical IDs and
    current tracker IDs using the Hungarian Algorithm, supporting rectangular matrices
    and centroid distance gating to reject matches that are too far apart.
    
    Args:
        prev_averaged_pigs (dict): master_id -> average bbox
        curr_averaged_pigs (dict): tracker_id -> average bbox
        max_distance_threshold (float): Maximum allowed distance in pixels to accept a match.
        
    Returns:
        dict: Mapping of master_id (str) to tracker_id (str).
    """
    if not prev_averaged_pigs or not curr_averaged_pigs:
        return {}

    # Extract stable lists of master IDs and tracker IDs to maintain matrix indexing
    master_ids = list(prev_averaged_pigs.keys())
    tracker_ids = list(curr_averaged_pigs.keys())

    # Build cost matrix based on Euclidean distance of bounding box centroids
    cost_matrix = np.zeros((len(master_ids), len(tracker_ids)))
    for i, m_id in enumerate(master_ids):
        prev_center = get_centroid(prev_averaged_pigs[m_id])
        for j, t_id in enumerate(tracker_ids):
            curr_center = get_centroid(curr_averaged_pigs[t_id])
            # Euclidean distance between centroids
            cost_matrix[i, j] = np.sqrt((prev_center[0] - curr_center[0])**2 + (prev_center[1] - curr_center[1])**2)

    # Solve matching problem using the Hungarian algorithm (supports rectangular matrices)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Compile the final mapping dict, applying the distance threshold check (gating)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        dist = cost_matrix[r, c]
        m_id = str(master_ids[r])
        t_id = str(tracker_ids[c])
        
        if dist <= max_distance_threshold:
            mapping[m_id] = t_id
        else:
            print(f"    [Gate Rejected] Match canonical ID {m_id} -> tracker ID {t_id} rejected (distance {dist:.1f}px > threshold {max_distance_threshold}px)")

    return mapping

def match_orphan_trackers_to_empty_masters(empty_master_ids, orphan_tracker_bboxes, prev_averaged_pigs, max_distance_threshold=375.0):
    """
    Second-pass matching for vacant canonical slots using leftover persistent trackers.
    Uses a more permissive distance gate than the primary boundary match.
    """
    if not empty_master_ids or not orphan_tracker_bboxes:
        return {}

    master_ids = sorted(empty_master_ids)
    tracker_ids = sorted(orphan_tracker_bboxes.keys())

    cost_matrix = np.zeros((len(master_ids), len(tracker_ids)))
    for i, m_id in enumerate(master_ids):
        prev_bbox = prev_averaged_pigs.get(m_id)
        for j, t_id in enumerate(tracker_ids):
            orphan_center = get_centroid(orphan_tracker_bboxes[t_id])
            if prev_bbox is not None:
                prev_center = get_centroid(prev_bbox)
                cost_matrix[i, j] = np.sqrt(
                    (prev_center[0] - orphan_center[0])**2 + (prev_center[1] - orphan_center[1])**2
                )
            else:
                cost_matrix[i, j] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        dist = cost_matrix[r, c]
        m_id = str(master_ids[r])
        t_id = str(tracker_ids[c])
        if prev_averaged_pigs.get(master_ids[r]) is None or dist <= max_distance_threshold:
            mapping[m_id] = t_id
            if dist > 0:
                print(f"    [Fallback Match] canonical ID {m_id} -> tracker ID {t_id} (distance {dist:.1f}px)")
        else:
            print(
                f"    [Fallback Rejected] canonical ID {m_id} -> tracker ID {t_id} "
                f"rejected (distance {dist:.1f}px > threshold {max_distance_threshold}px)"
            )
    return mapping

def apply_fallback_remap(current_remap, prev_averaged_pigs, averaged_curr_pigs, max_dist=250.0):
    """
    Fills empty remap slots by matching unused persistent trackers to vacant masters.
    """
    used_trackers = {int(v) for v in current_remap.values() if v != ""}
    empty_masters = [int(k) for k, v in current_remap.items() if v == ""]
    orphan_trackers = {
        t_id: bbox for t_id, bbox in averaged_curr_pigs.items()
        if t_id not in used_trackers
    }

    if not empty_masters or not orphan_trackers:
        return current_remap

    fallback = match_orphan_trackers_to_empty_masters(
        empty_masters,
        orphan_trackers,
        prev_averaged_pigs,
        max_distance_threshold=max_dist * 1.5,
    )
    for master_id, tracker_id in fallback.items():
        current_remap[master_id] = tracker_id
    return current_remap

def generate_video_mapping(video_dir, source_ann_dir="data/annotations/sam", n_frames=5, min_occupancy=0.15, max_dist=250.0, decay_factor=0.3):
    """
    Processes all clips of a video sequentially to match IDs between clip boundaries.
    """
    video_path = os.path.join(source_ann_dir, video_dir)
    if not os.path.exists(video_path):
        print(f"Error: Video path {video_path} does not exist.")
        return None

    clip_files = sorted(glob(os.path.join(video_path, "*.json")))
    if not clip_files:
        print(f"No JSON annotation files found in {video_path}")
        return None

    print(f"\n>>> Analyzing video '{video_dir}' with {len(clip_files)} clips using Robust Hungarian Algorithm...")

    clips_mapping = []
    
    # Store the last known average positions of canonical pig IDs (0 to 4)
    last_known_pigs = {}

    for idx, clip_file in enumerate(clip_files):
        clip_name = os.path.splitext(os.path.basename(clip_file))[0]
        
        with open(clip_file, 'r') as f:
            data = json.load(f)

        annotations = data.get('annotations', [])
        images = data.get('images', [])
        
        if not images:
            print(f"  Clip {clip_name} has no images. Skipping.")
            continue

        # Sort images to ensure temporal order
        images_sorted = sorted(images, key=lambda img: img.get('frame_id', 0))
        sorted_img_ids = [img['id'] for img in images_sorted]
        first_frame_id = images_sorted[0].get('frame_id', 0)
        last_frame_id = images_sorted[-1].get('frame_id', 0)

        # 1. Identify persistent tracks within this clip
        persistent_tracks = get_persistent_tracks(annotations, sorted_img_ids, min_occupancy_ratio=min_occupancy)
        print(f"  Clip {clip_name}: Found {len(persistent_tracks)} persistent tracks out of {len(set(ann['track_id'] for ann in annotations))} total track IDs.")

        current_remap = {}

        if idx == 0:
            # First clip initialization: Map existing persistent tracker IDs directly to canonical IDs 0-4
            print(f"  Clip {clip_name} (First clip): Initializing canonical IDs from persistent tracks.")
            averaged_first_pigs = get_average_bboxes(annotations, sorted_img_ids, n_frames, persistent_tracks, from_start=True, decay_factor=decay_factor)
            tracker_ids_found = sorted(list(averaged_first_pigs.keys()))
            
            for m_id, t_id in enumerate(tracker_ids_found[:5]):
                current_remap[str(m_id)] = str(t_id)
            
            # Pad missing canonical IDs
            for m_id in range(5):
                if str(m_id) not in current_remap:
                    current_remap[str(m_id)] = ""
        else:
            # Subsequent clips: Match first N frames of current clip against last N frames of previous clip
            averaged_curr_pigs = get_average_bboxes(annotations, sorted_img_ids, n_frames, persistent_tracks, from_start=True, decay_factor=decay_factor)
            
            # Map canonical master IDs to their averaged positions from previous clip
            prev_pigs_matched = {int(m_id): pos for m_id, pos in last_known_pigs.items() if pos is not None}
            
            current_remap = match_pigs_hungarian(prev_pigs_matched, averaged_curr_pigs, max_distance_threshold=max_dist)

            # Ensure all canonical IDs 0-4 are keys in the output remap dictionary
            for m_id in range(5):
                if str(m_id) not in current_remap:
                    current_remap[str(m_id)] = ""

            # Second pass: pair leftover persistent trackers with empty master slots
            current_remap = apply_fallback_remap(
                current_remap, prev_pigs_matched, averaged_curr_pigs, max_dist=max_dist
            )

            print(f"  Clip {clip_name}: Matched using Hungarian Algorithm. Map: {current_remap}")

        # Compute average positions of persistent pigs in the last N frames of this clip
        averaged_last_pigs = get_average_bboxes(annotations, sorted_img_ids, n_frames, persistent_tracks, from_start=False, decay_factor=decay_factor)

        # Store positions using canonical master IDs (remapping current tracker IDs to master IDs)
        last_known_pigs = {str(m_id): None for m_id in range(5)}
        inverse_remap = {int(v): int(k) for k, v in current_remap.items() if v != ""}

        for tracker_id, avg_bbox in averaged_last_pigs.items():
            if tracker_id in inverse_remap:
                master_id = str(inverse_remap[tracker_id])
                last_known_pigs[master_id] = avg_bbox

        # Propagate positions of unmapped surplus trackers to still-empty master slots
        used_trackers = {int(v) for v in current_remap.values() if v != ""}
        empty_masters = [int(k) for k, v in current_remap.items() if v == ""]
        surplus_trackers = {
            t_id: bbox for t_id, bbox in averaged_last_pigs.items()
            if t_id not in used_trackers
        }
        if empty_masters and surplus_trackers:
            prev_for_fallback = {int(k): v for k, v in last_known_pigs.items() if v is not None}
            fallback_positions = match_orphan_trackers_to_empty_masters(
                empty_masters,
                surplus_trackers,
                prev_for_fallback,
                max_distance_threshold=max_dist * 1.5,
            )
            for master_id, tracker_id in fallback_positions.items():
                last_known_pigs[master_id] = averaged_last_pigs[int(tracker_id)]

        clips_mapping.append({
            "clip": clip_name,
            "remaps": [
                {
                    "frame_start": int(first_frame_id),
                    "frame_end": int(last_frame_id),
                    "remap": current_remap
                }
            ]
        })

    return {
        "video": video_dir,
        "clips": clips_mapping
    }

def save_video_mapping(new_video_mapping, mapping_path):
    """
    Saves the video remapping as a single entry inside a list.
    This maintains compatibility with the existing annotation_manager.py.
    """
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    # Save as a list containing the single video dictionary
    with open(mapping_path, 'w') as f:
        json.dump([new_video_mapping], f, indent=4)
    print(f"\n✓ Saved video mappings to {mapping_path}")

def main():
    parser = argparse.ArgumentParser(description="Auto match pig tracking IDs between consecutive video clips.")
    parser.add_argument("--video", required=True, help="Name of the video directory (e.g., June_23_01)")
    parser.add_argument("--source-dir", default="data/annotations/sam", help="Source annotation directory")
    parser.add_argument("--output-map", default=None, help="Path to save mapping JSON. Defaults to data/annotations/remappings/{video}.json")
    parser.add_argument("--n-frames", type=int, default=5, help="Number of frames to average at clip boundaries")
    parser.add_argument("--min-occupancy", type=float, default=0.15, help="Minimum frame occupancy ratio to consider a track valid (filters false positives)")
    parser.add_argument("--max-dist", type=float, default=250.0, help="Maximum allowed centroid distance in pixels for a valid boundary match")
    parser.add_argument("--decay-factor", type=float, default=0.3, help="Exponential decay factor (0.0 to 1.0). 0.0 uses only the boundary frame, 1.0 averages equally.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mapping file if it exists")

    args = parser.parse_args()

    # Determine default output path if not provided
    if args.output_map is None:
        output_path = os.path.join("data/annotations/remappings", f"{args.video}.json")
    else:
        output_path = args.output_map

    if os.path.exists(output_path) and not args.overwrite:
        print(f"\n(!) Mapping file already exists at {output_path}. Use --overwrite to regenerate it.")
        return

    video_mapping = generate_video_mapping(
        args.video, 
        source_ann_dir=args.source_dir, 
        n_frames=args.n_frames,
        min_occupancy=args.min_occupancy,
        max_dist=args.max_dist,
        decay_factor=args.decay_factor
    )
    
    if video_mapping:
        save_video_mapping(video_mapping, mapping_path=output_path)

if __name__ == "__main__":
    main()
