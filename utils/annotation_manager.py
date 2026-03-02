import os
import shutil
import argparse
import json
import yaml

class AnnotationManager:
    def __init__(self, pose_dir="data/annotations/pose", refined_dir="data/annotations/refined"):
        self.pose_dir = pose_dir
        self.refined_dir = refined_dir

    def initialize_refined(self):
        """Copies all files from data/annotations/pose to data/annotations/refined."""
        if not os.path.exists(self.refined_dir):
            print(f">>> Initializing refined annotations: {self.pose_dir} -> {self.refined_dir}")
            os.makedirs(os.path.dirname(self.refined_dir), exist_ok=True)
            shutil.copytree(self.pose_dir, self.refined_dir)
            print("✓ Copy completed successfully.")
        else:
            print(f"--- The directory {self.refined_dir} already exists. Skipping initialization.")

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

    def apply_remap(self, video_key, clip_key, clip_mapping):
        """
        Applies mapping with logic: { MASTER_ID: TRACKER_ID }
        It maps the value (Tracker) to the key (Master).
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

        if remap_count > 0:
            with open(refined_path, 'w') as f:
                json.dump(data, f, indent=4)
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
    args = parser.parse_args()
    manager = AnnotationManager()
    if args.command == "init": manager.initialize_refined()
    elif args.command == "delete-id": manager.delete_id(args.video, args.clip, args.id)
    elif args.command == "remap":
        if args.video and args.clip: manager.remap_ids(args.video, args.clip, args.map)
        else: manager.remap_all(args.map)
if __name__ == "__main__":
    main()
