import os
import shutil
import argparse
import json

class AnnotationManager:
    def __init__(self, pose_dir="data/annotations/pose", refined_dir="data/annotations/refined"):
        self.pose_dir = pose_dir
        self.refined_dir = refined_dir

    def initialize_refined(self):
        """Copies missing files/folders from data/annotations/pose to data/annotations/refined."""
        print(f">>> Initializing refined annotations: {self.pose_dir} -> {self.refined_dir}")
        if not os.path.exists(self.pose_dir):
            print(f"Error: Source directory {self.pose_dir} does not exist.")
            return

        os.makedirs(self.refined_dir, exist_ok=True)
        for item in os.listdir(self.pose_dir):
            src_item = os.path.join(self.pose_dir, item)
            dst_item = os.path.join(self.refined_dir, item)
            if os.path.exists(dst_item):
                print(f"--- The subfolder/file {item} already exists. Skipping (leaving unmodified).")
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

    def apply_remap(self, video_key, clip_key, clip_mapping):
        """
        Applies mapping with logic: { MASTER_ID: TRACKER_ID }
        It maps the value (Tracker) to the key (Master).
        Additionally, resolves any ID collisions (multiple annotations with the same track_id in a single frame)
        by keeping the one closest to the previous frame's location and reassigning the duplicates to dummy IDs (99+).
        """
        pose_path = os.path.join(self.pose_dir, video_key, f"{clip_key}.json")
        refined_path = os.path.join(self.refined_dir, video_key, f"{clip_key}.json")
        
        # Always start from original pose source
        if os.path.exists(pose_path):
            os.makedirs(os.path.dirname(refined_path), exist_ok=True)
            shutil.copy2(pose_path, refined_path)
        else:
            print(f"  ! Error: Source annotation not found at {pose_path}")
            return False

        with open(refined_path, 'r') as f:
            data = json.load(f)

        img_to_frame = {img['id']: img.get('frame_id') for img in data.get('images', [])}
        remap_count = 0

        # 1. Apply the initial remapping based on the mapping file
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            frame_id = img_to_frame.get(img_id)
            if frame_id is None: continue

            # Find range
            for r_range in clip_mapping:
                if r_range['frame_start'] <= frame_id <= r_range['frame_end']:
                    current_remap = r_range['remap']
                    
                    # LOGIC: Value is original tracker ID, Key is our Master ID
                    # Example: "1": "5" -> Search for ID 5, change to 1.
                    current_tracker_id = str(ann.get('track_id'))
                    
                    for master_id, tracker_id in current_remap.items():
                        if current_tracker_id == str(tracker_id):
                            new_val = int(master_id)
                            if ann['track_id'] != new_val:
                                ann['track_id'] = new_val
                                remap_count += 1
                            break # Found mapping for this annotation
                    break

        # Helper functions for centroid distance calculations
        def get_centroid(bbox):
            x, y, w, h = bbox
            return (x + w / 2.0, y + h / 2.0)

        def get_dist(bbox1, bbox2):
            c1 = get_centroid(bbox1)
            c2 = get_centroid(bbox2)
            return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5

        # 2. Resolve ID Collisions chronologically frame by frame
        images_sorted = sorted(data.get('images', []), key=lambda img: img.get('frame_id', 0))
        
        # Group annotations by image_id
        img_anns = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in img_anns:
                img_anns[img_id] = []
            img_anns[img_id].append(ann)

        last_frame_bbox = {} # Maps track_id -> bbox of the resolved pig in the previous frame
        dummy_id_counter = 99
        collision_count = 0

        for img_entry in images_sorted:
            img_id = img_entry['id']
            frame_id = img_entry.get('frame_id', 0)
            anns = img_anns.get(img_id, [])
            
            # Count track_id occurrences in this frame
            track_counts = {}
            for ann in anns:
                tid = ann.get('track_id')
                if tid is not None:
                    track_counts[tid] = track_counts.get(tid, 0) + 1

            # Find duplicated track_ids
            duplicated_ids = {tid for tid, count in track_counts.items() if count > 1}

            # Map to keep track of resolved bboxes for the current frame
            current_frame_bboxes = {}

            for ann in anns:
                tid = ann.get('track_id')
                if tid is None:
                    continue

                if tid not in duplicated_ids:
                    # No duplicate, keep as is
                    current_frame_bboxes[tid] = ann['bbox']
                else:
                    # Collision detected! Collect all candidates for this ID
                    candidates = [a for a in anns if a.get('track_id') == tid]
                    
                    # We only process this collision once when we visit the first candidate
                    if ann != candidates[0]:
                        continue
                        
                    # Find which candidate is most consistent with the history
                    best_candidate_idx = 0
                    if tid in last_frame_bbox and last_frame_bbox[tid] is not None:
                        # Find closest candidate based on centroid distance
                        best_dist = float('inf')
                        for idx, cand in enumerate(candidates):
                            dist = get_dist(cand['bbox'], last_frame_bbox[tid])
                            if dist < best_dist:
                                best_dist = dist
                                best_candidate_idx = idx
                    else:
                        # Fallback: keep the candidate with the largest bbox area
                        max_area = -1
                        for idx, cand in enumerate(candidates):
                            cand_area = cand['bbox'][2] * cand['bbox'][3]
                            if cand_area > max_area:
                                max_area = cand_area
                                best_candidate_idx = idx

                    # Reassign duplicates to dummy IDs (99, 100, ...)
                    for idx, cand in enumerate(candidates):
                        if idx != best_candidate_idx:
                            old_id = cand['track_id']
                            cand['track_id'] = dummy_id_counter
                            print(f"    [Collision Resolved] Frame {frame_id}: Resolved duplicate ID {old_id}. Reassigned duplicate to dummy ID {dummy_id_counter}")
                            dummy_id_counter += 1
                            collision_count += 1
                            remap_count += 1 # Mark as modified
                        else:
                            current_frame_bboxes[tid] = cand['bbox']

            # Update history for next frame
            last_frame_bbox = current_frame_bboxes

        if remap_count > 0:
            with open(refined_path, 'w') as f:
                json.dump(data, f, indent=4)
            if collision_count > 0:
                print(f"  ✓ Updated {remap_count} annotations in {video_key}/{clip_key} (Resolved {collision_count} collisions)")
            else:
                print(f"  ✓ Updated {remap_count} annotations in {video_key}/{clip_key}")
            return True
        else:
            print(f"  --- No changes needed in {video_key}/{clip_key}")
            return False

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
        print(f">>> Starting batch remapping (Master <- Tracker)...")
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
        print(f"\n>>> FINISHED. Processed {total_clips} clips.")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("init")
    del_p = subparsers.add_parser("delete-id")
    del_p.add_argument("--video", required=True)
    del_p.add_argument("--clip", required=True)
    del_p.add_argument("--id", required=True, type=int)
    remap_p = subparsers.add_parser("remap")
    remap_p.add_argument("--video")
    remap_p.add_argument("--clip")
    remap_p.add_argument("--map", required=True)

    del_f = subparsers.add_parser("delete-frames")
    del_f.add_argument("--video", required=True)
    del_f.add_argument("--clip", required=True)
    del_f.add_argument("--start", required=True, type=int)
    del_f.add_argument("--end", required=True, type=int)

    args = parser.parse_args()
    manager = AnnotationManager()
    if args.command == "init": manager.initialize_refined()
    elif args.command == "delete-id": manager.delete_id(args.video, args.clip, args.id)
    elif args.command == "delete-frames": manager.delete_frames(args.video, args.clip, args.start, args.end)
    elif args.command == "remap":
        if args.video and args.clip: manager.remap_ids(args.video, args.clip, args.map)
        else: manager.remap_all(args.map)
if __name__ == "__main__":
    main()
