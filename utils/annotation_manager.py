import os
import shutil
import argparse
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

class AnnotationManager:
    def __init__(self, pose_dir="data/annotations/pose", refined_dir="data/annotations/refined"):
        self.pose_dir = pose_dir
        self.refined_dir = refined_dir

    def initialize_refined(self, video_id=None, overwrite=False):
        """Copies files/folders from data/annotations/pose to data/annotations/refined."""
        print(f">>> Initializing refined annotations: {self.pose_dir} -> {self.refined_dir}")
        if not os.path.exists(self.pose_dir):
            print(f"Error: Source directory {self.pose_dir} does not exist.")
            return

        os.makedirs(self.refined_dir, exist_ok=True)
        
        items = [video_id] if video_id else os.listdir(self.pose_dir)

        for item in items:
            src_item = os.path.join(self.pose_dir, item)
            dst_item = os.path.join(self.refined_dir, item)
            
            if not os.path.exists(src_item):
                print(f"Error: Source {src_item} does not exist. Skipping.")
                continue

            if os.path.exists(dst_item):
                if not overwrite:
                    print(f"--- The subfolder/file {item} already exists. Skipping (leaving unmodified).")
                    continue
                else:
                    print(f"✓ Overwriting {item}...")
                    if os.path.isdir(dst_item):
                        shutil.rmtree(dst_item)
                    else:
                        os.remove(dst_item)
            else:
                print(f"✓ Copying {item} to {self.refined_dir}...")

            if os.path.isdir(src_item):
                shutil.copytree(src_item, dst_item)
            else:
                shutil.copy2(src_item, dst_item)
        print("✓ Initialization completed.")

    def _ensure_refined_exists(self):
        if not os.path.exists(self.refined_dir):
            self.initialize_refined()

    def delete_id(self, video_id, clip_id, target_id):
        self._ensure_refined_exists()
        video_key = f"video{video_id}" if not str(video_id).startswith("video") else video_id
        clip_key = str(clip_id).replace(".json", "")
        
        json_path = os.path.join(self.refined_dir, video_key, f"{clip_key}.json")
        if not os.path.exists(json_path):
            print(f"Error: Annotation file not found at {json_path}")
            return

        with open(json_path, 'r') as f:
            data = json.load(f)

        new_annotations = [ann for ann in data.get('annotations', []) if ann.get('track_id') != target_id]
        deleted_count = len(data.get('annotations', [])) - len(new_annotations)
        
        if deleted_count > 0:
            data['annotations'] = new_annotations
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"✓ Removed {deleted_count} instances of ID {target_id} from {video_key}/{clip_key}")

    def delete_frames(self, video_id, clip_id, frame_start, frame_end):
        self._ensure_refined_exists()
        video_key = f"video{video_id}" if not str(video_id).startswith("video") else video_id
        clip_key = str(clip_id).replace(".json", "")
        
        json_path = os.path.join(self.refined_dir, video_key, f"{clip_key}.json")
        if not os.path.exists(json_path):
            print(f"Error: Annotation file not found at {json_path}")
            return

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Identify image_ids that fall within the frame range
        target_image_ids = {img['id'] for img in data.get('images', []) 
                           if frame_start <= img.get('frame_id', -1) <= frame_end}
        
        if not target_image_ids:
            print(f"--- No images found in range {frame_start}-{frame_end} for {video_key}/{clip_key}")
            return

        original_count = len(data.get('annotations', []))
        data['annotations'] = [ann for ann in data.get('annotations', []) if ann.get('image_id') not in target_image_ids]
        deleted_count = original_count - len(data['annotations'])

        if deleted_count > 0:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"✓ Removed {deleted_count} annotations from frames {frame_start}-{frame_end} in {video_key}/{clip_key}")
        else:
            print(f"--- No annotations found in frames {frame_start}-{frame_end} for {video_key}/{clip_key}")

    def _get_track_avg_bbox(self, data, images_sorted, track_id, n_frames=5, frame_start=None, frame_end=None):
        """Weighted average bbox for a track on the first N frames within an optional frame range."""
        target_images = []
        for img in images_sorted:
            frame_id = img.get('frame_id', 0)
            if frame_start is not None and frame_id < frame_start:
                continue
            if frame_end is not None and frame_id > frame_end:
                continue
            target_images.append(img)
        target_images = target_images[:n_frames]
        target_img_ids = {img['id'] for img in target_images}

        bboxes = [ann['bbox'] for ann in data.get('annotations', [])
                  if ann.get('track_id') == track_id and ann.get('image_id') in target_img_ids]
        if not bboxes:
            return None
        return np.mean(np.array(bboxes), axis=0).tolist()

    def _assign_orphan_trackers(self, data, clip_mapping, images_sorted, get_dist, n_frames=5, max_dist=300.0):
        """
        Fills empty remap slots by pairing orphan tracker IDs (not used as remap sources)
        with vacant master IDs using Hungarian matching on boundary-frame positions.
        Must run before initial remapping to avoid ID collisions.
        """
        orphan_count = 0
        img_to_frame = {img['id']: img.get('frame_id') for img in data.get('images', [])}

        for r_range in clip_mapping:
            remap = r_range['remap']
            frame_start = r_range['frame_start']
            frame_end = r_range['frame_end']

            mapped_tracker_values = {str(v) for v in remap.values() if v != ""}
            empty_masters = sorted(int(k) for k, v in remap.items() if v == "")

            clip_track_ids = set()
            for ann in data.get('annotations', []):
                frame_id = img_to_frame.get(ann.get('image_id'))
                if frame_id is None or not (frame_start <= frame_id <= frame_end):
                    continue
                tid = ann.get('track_id')
                if tid is not None:
                    clip_track_ids.add(tid)

            orphan_ids = sorted(tid for tid in clip_track_ids if str(tid) not in mapped_tracker_values)
            if not orphan_ids or not empty_masters:
                continue

            mapped_master_avgs = []
            for master_id, tracker_id in remap.items():
                if tracker_id == "":
                    continue
                avg_bbox = self._get_track_avg_bbox(
                    data, images_sorted, int(tracker_id), n_frames, frame_start, frame_end
                )
                if avg_bbox is not None:
                    mapped_master_avgs.append(avg_bbox)

            orphan_avgs = []
            for orphan_id in orphan_ids:
                avg_bbox = self._get_track_avg_bbox(
                    data, images_sorted, orphan_id, n_frames, frame_start, frame_end
                )
                orphan_avgs.append(avg_bbox)

            if not any(orphan_avgs):
                continue

            mapped_centroid = None
            if mapped_master_avgs:
                mapped_centroid = np.mean(np.array(mapped_master_avgs), axis=0).tolist()

            cost_matrix = np.zeros((len(empty_masters), len(orphan_ids)))
            for i, empty_master in enumerate(empty_masters):
                for j, orphan_avg in enumerate(orphan_avgs):
                    if orphan_avg is None:
                        cost_matrix[i, j] = float('inf')
                    elif mapped_centroid is not None:
                        cost_matrix[i, j] = get_dist(orphan_avg, mapped_centroid)
                    else:
                        cost_matrix[i, j] = 0.0

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                dist = cost_matrix[r, c]
                if dist <= max_dist or len(empty_masters) == len(orphan_ids) == 1:
                    master_id = str(empty_masters[r])
                    orphan_id = str(orphan_ids[c])
                    remap[master_id] = orphan_id
                    affected = sum(
                        1 for ann in data.get('annotations', [])
                        if ann.get('track_id') == orphan_ids[c]
                        and frame_start <= img_to_frame.get(ann.get('image_id'), -1) <= frame_end
                    )
                    orphan_count += affected

        return orphan_count

    def _run_hungarian_rescue(self, data, images_sorted, get_dist, valid_ids=None, max_dist=300.0):
        """Rescues extra/non-canonical IDs by globally renaming them to missing canonical IDs."""
        if valid_ids is None:
            valid_ids = {0, 1, 2, 3, 4}

        rescue_count = 0
        img_anns_post = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            img_anns_post.setdefault(img_id, []).append(ann)

        canonical_last_bbox = {}
        for img_entry in images_sorted:
            img_id = img_entry['id']
            anns = img_anns_post.get(img_id, [])

            present_canonical = {ann.get('track_id') for ann in anns if ann.get('track_id') in valid_ids}
            extra_anns = [ann for ann in anns if ann.get('track_id') not in valid_ids and ann.get('track_id') is not None]

            missing_canonical = sorted(
                cid for cid in valid_ids
                if cid not in present_canonical and cid in canonical_last_bbox
            )

            if missing_canonical and extra_anns:
                cost_matrix = np.array([
                    [get_dist(canonical_last_bbox[cid], extra_ann['bbox']) for extra_ann in extra_anns]
                    for cid in missing_canonical
                ])
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for r, c in zip(row_ind, col_ind):
                    dist = cost_matrix[r, c]
                    if dist < max_dist:
                        cid = missing_canonical[r]
                        old_extra_id = extra_anns[c]['track_id']
                        for a in data.get('annotations', []):
                            if a.get('track_id') == old_extra_id:
                                a['track_id'] = cid
                                rescue_count += 1
                        present_canonical.add(cid)

            for ann in anns:
                tid = ann.get('track_id')
                if tid in valid_ids:
                    canonical_last_bbox[tid] = ann['bbox']

        return rescue_count

    def apply_remap(self, video_key, clip_key, clip_mapping):
        """
        Applies mapping with logic: { MASTER_ID: TRACKER_ID }
        """
        pose_path = os.path.join(self.pose_dir, video_key, f"{clip_key}.json")
        refined_path = os.path.join(self.refined_dir, video_key, f"{clip_key}.json")

        if os.path.exists(pose_path):
            os.makedirs(os.path.dirname(refined_path), exist_ok=True)
            shutil.copy2(pose_path, refined_path)
        else:
            print(f"  ! Error: Source annotation not found at {pose_path}")
            return False

        with open(refined_path, 'r') as f:
            data = json.load(f)

        img_to_frame = {img['id']: img.get('frame_id') for img in data.get('images', [])}
        images_sorted = sorted(data.get('images', []), key=lambda img: img.get('frame_id', 0))
        remap_count = 0
        valid_ids = {0, 1, 2, 3, 4}

        def get_centroid(bbox):
            x, y, w, h = bbox
            return (x + w / 2.0, y + h / 2.0)

        def get_dist(bbox1, bbox2):
            c1 = get_centroid(bbox1)
            c2 = get_centroid(bbox2)
            return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5

        def calculate_iou(bbox1, bbox2):
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            xi1, yi1 = max(x1, x2), max(y1, y2)
            xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union_area = (w1 * h1) + (w2 * h2) - inter_area
            return inter_area / union_area if union_area > 0 else 0

        # 1a. Assign orphan trackers to empty master slots before remapping
        orphan_count = self._assign_orphan_trackers(data, clip_mapping, images_sorted, get_dist)

        # 1b. Initial Remapping
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            frame_id = img_to_frame.get(img_id)
            if frame_id is None:
                continue
            for r_range in clip_mapping:
                if r_range['frame_start'] <= frame_id <= r_range['frame_end']:
                    current_remap = r_range['remap']
                    current_tracker_id = str(ann.get('track_id'))
                    for master_id, tracker_id in current_remap.items():
                        if tracker_id != "" and current_tracker_id == str(tracker_id):
                            new_val = int(master_id)
                            if ann['track_id'] != new_val:
                                ann['track_id'] = new_val
                                remap_count += 1
                            break
                    break

        # 2. Pre-Collision Resolution & NMS
        pre_nms_count = len(data.get('annotations', []))
        img_anns = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            img_anns.setdefault(img_id, []).append(ann)

        nms_kept_anns = []
        for img_id in img_anns:
            current_anns = img_anns[img_id]
            if not current_anns:
                continue
            sorted_anns = sorted(current_anns, key=lambda x: x['bbox'][2] * x['bbox'][3], reverse=True)
            kept = []
            while len(sorted_anns) > 0:
                best = sorted_anns.pop(0)
                kept.append(best)
                sorted_anns = [a for a in sorted_anns if calculate_iou(best['bbox'], a['bbox']) < 0.85]
            nms_kept_anns.extend(kept)

        data['annotations'] = nms_kept_anns
        nms_count = pre_nms_count - len(data['annotations'])

        col1 = self._resolve_collisions(data, images_sorted, get_dist)

        # 3. Optimal Rescue (Hungarian)
        rescue_count = self._run_hungarian_rescue(data, images_sorted, get_dist, valid_ids)

        # 4. Final Collision Resolution
        col2 = self._resolve_collisions(data, images_sorted, get_dist)

        # 4b. Post-Col2 rescue to recover IDs reassigned to 99+ by collision resolution
        rescue2_count = self._run_hungarian_rescue(data, images_sorted, get_dist, valid_ids)
        rescue_count += rescue2_count

        # 5. Purge
        original_ann_count = len(data['annotations'])
        data['annotations'] = [ann for ann in data['annotations'] if ann.get('track_id') in valid_ids]
        pruned_count = original_ann_count - len(data['annotations'])

        if (remap_count > 0 or orphan_count > 0 or col1 > 0 or rescue_count > 0
                or col2 > 0 or pruned_count > 0 or nms_count > 0):
            with open(refined_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(
                f"  ✓ Updated {video_key}/{clip_key} "
                f"(Orphan: {orphan_count}, Col1: {col1}, Rescue: {rescue_count}, "
                f"Col2: {col2}, Pruned: {pruned_count}, NMS: {nms_count})"
            )
            return True
        return False

    def _resolve_collisions(self, data, images_sorted, get_dist):
        """
        Internal helper to ensure each track_id appears only once per frame.
        Duplicates are reassigned to dummy IDs (99+).
        """
        img_anns = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in img_anns:
                img_anns[img_id] = []
            img_anns[img_id].append(ann)

        last_frame_bbox = {}
        dummy_id_counter = 99
        collision_count = 0

        for img_entry in images_sorted:
            img_id = img_entry['id']
            frame_id = img_entry.get('frame_id', 0)
            anns = img_anns.get(img_id, [])
            
            track_counts = {}
            for ann in anns:
                tid = ann.get('track_id')
                if tid is not None:
                    track_counts[tid] = track_counts.get(tid, 0) + 1

            duplicated_ids = {tid for tid, count in track_counts.items() if count > 1}
            current_frame_bboxes = {}

            for ann in anns:
                tid = ann.get('track_id')
                if tid is None: continue
                if tid not in duplicated_ids:
                    current_frame_bboxes[tid] = ann['bbox']
                else:
                    candidates = [a for a in anns if a.get('track_id') == tid]
                    if ann != candidates[0]: continue
                        
                    best_candidate_idx = 0
                    if tid in last_frame_bbox and last_frame_bbox[tid] is not None:
                        best_dist = float('inf')
                        for idx, cand in enumerate(candidates):
                            dist = get_dist(cand['bbox'], last_frame_bbox[tid])
                            if dist < best_dist:
                                best_dist = dist
                                best_candidate_idx = idx
                        
                        # --- CRITICAL FIX: Spatial Confidence Gate ---
                        # If the best match is too far from the last known position, 
                        # the anchor is unreliable (e.g., after a long disappearance).
                        # In this case, fallback to area-based resolution to avoid purging the real pig.
                        if best_dist > 300.0:
                            max_area = -1
                            best_candidate_idx = 0
                            for idx, cand in enumerate(candidates):
                                cand_area = cand['bbox'][2] * cand['bbox'][3]
                                if cand_area > max_area:
                                    max_area = cand_area
                                    best_candidate_idx = idx
                    else:
                        max_area = -1
                        for idx, cand in enumerate(candidates):
                            cand_area = cand['bbox'][2] * cand['bbox'][3]
                            if cand_area > max_area:
                                max_area = cand_area
                                best_candidate_idx = idx

                    for idx, cand in enumerate(candidates):
                        if idx != best_candidate_idx:
                            cand['track_id'] = dummy_id_counter
                            dummy_id_counter += 1
                            collision_count += 1
                        else:
                            current_frame_bboxes[tid] = cand['bbox']
            last_frame_bbox = current_frame_bboxes
        return collision_count

    def remap_ids(self, video_id, clip_id, mapping_file):
        self._ensure_refined_exists()
        with open(mapping_file, 'r') as f:
            full_mapping = json.load(f)
        video_key = f"video{video_id}" if not str(video_id).startswith("video") else video_id
        clip_key = str(clip_id).replace(".json", "")
        for v_entry in full_mapping:
            if v_entry.get('video') == video_key:
                for c_entry in v_entry.get('clips', []):
                    if c_entry.get('clip') == clip_key:
                        self.apply_remap(video_key, clip_key, c_entry.get('remaps', []))
                        return

    def remap_all(self, mapping_file):
        self._ensure_refined_exists()
        print(f">>> Starting batch remapping from file: {mapping_file}")
        with open(mapping_file, 'r') as f:
            full_mapping = json.load(f)
        total_clips = 0
        updated_clips = 0
        for v_entry in full_mapping:
            v_key = v_entry.get('video')
            for c_entry in v_entry.get('clips', []):
                total_clips += 1
                if self.apply_remap(v_key, c_entry.get('clip'), c_entry.get('remaps', [])):
                    updated_clips += 1
        print(f"✓ Processed {total_clips} clips ({updated_clips} updated) from {os.path.basename(mapping_file)}")

    def remap_all(self, mapping_file):
        self._ensure_refined_exists()
        print(f">>> Starting batch remapping from file: {mapping_file}")
        with open(mapping_file, 'r') as f:
            full_mapping = json.load(f)
        total_clips = 0
        updated_clips = 0
        for v_entry in full_mapping:
            v_key = v_entry.get('video')
            for c_entry in v_entry.get('clips', []):
                total_clips += 1
                if self.apply_remap(v_key, c_entry.get('clip'), c_entry.get('remaps', [])):
                    updated_clips += 1
        print(f"✓ Processed {total_clips} clips ({updated_clips} updated) from {os.path.basename(mapping_file)}")

    def remap_all_files(self, remappings_dir="data/annotations/remappings"):
        """
        Scans the remappings directory and applies all found .json mapping files.
        """
        self._ensure_refined_exists()
        print(f">>> Starting batch remapping from all files in {remappings_dir}...")
        
        import glob
        mapping_files = sorted(glob.glob(os.path.join(remappings_dir, "*.json")))
        # Filter out manual fix files
        mapping_files = [f for f in mapping_files if not f.endswith("_fixes.json")]
        
        if not mapping_files:
            print(f"No mapping files found in {remappings_dir}")
            return

        total_videos_processed = 0
        for map_file in mapping_files:
            self.remap_all(map_file)
            total_videos_processed += 1
            
        print(f"\n>>> FINISHED. Processed {total_videos_processed} mapping files.")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    init_p = subparsers.add_parser("init")
    init_p.add_argument("--all", action="store_true", default=True, help="Copy all folders")
    init_p.add_argument("--video", help="Copy specific video folder")
    init_p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    del_p = subparsers.add_parser("delete-id")
    del_p.add_argument("--video", required=True)
    del_p.add_argument("--clip", required=True)
    del_p.add_argument("--id", required=True, type=int)
    remap_p = subparsers.add_parser("remap")
    remap_p.add_argument("--video")
    remap_p.add_argument("--clip")
    remap_p.add_argument("--map")
    remap_p.add_argument("--all", action="store_true", help="Remap all videos using all mapping files in data/annotations/remappings/")
    
    del_f = subparsers.add_parser("delete-frames")
    del_f.add_argument("--video", required=True)
    del_f.add_argument("--clip", required=True)
    del_f.add_argument("--start", required=True, type=int)
    del_f.add_argument("--end", required=True, type=int)
    
    args = parser.parse_args()
    manager = AnnotationManager()
    if args.command == "init": manager.initialize_refined(video_id=args.video, overwrite=args.overwrite)
    elif args.command == "delete-id": manager.delete_id(args.video, args.clip, args.id)
    elif args.command == "delete-frames": manager.delete_frames(args.video, args.clip, args.start, args.end)
    elif args.command == "remap":
        if args.all:
            manager.remap_all_files()
        elif args.map:
            if args.video and args.clip: manager.remap_ids(args.video, args.clip, args.map)
            else: manager.remap_all(args.map)
        else:
            print("Error: The 'remap' command requires either --all or --map.")
if __name__ == "__main__":
    main()
