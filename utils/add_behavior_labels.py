import json
import os
import csv
from pathlib import Path
import shutil

def add_behavior_labels(csv_path, src_dir, dst_dir):
    print(f">>> Loading behavior labels from {csv_path}...")
    
    # Store labels as (video, clip, frame, track_id) -> behavior
    behavior_lookup = {}
    
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Basic cleaning
                v = row['video'].strip()
                c = row['clip'].strip()
                f_id = int(row['frame'])
                t_id = int(row['id'])
                b = row['behavior'].strip()
                
                behavior_lookup[(v, c, f_id, t_id)] = b
    else:
        print(f"!!! CSV file not found: {csv_path}")

    print(f">>> Processing annotations from {src_dir} to {dst_dir}...")
    
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    # Walk through videos
    for video_name in os.listdir(src_dir):
        src_video_path = os.path.join(src_dir, video_name)
        if not os.path.isdir(src_video_path):
            continue
            
        dst_video_path = os.path.join(dst_dir, video_name)
        os.makedirs(dst_video_path, exist_ok=True)
        
        # Walk through clips (JSONs)
        for json_file in os.listdir(src_video_path):
            if not json_file.endswith('.json'):
                continue
            
            src_json_path = os.path.join(src_video_path, json_file)
            dst_json_path = os.path.join(dst_video_path, json_file)
            clip_name = json_file.replace('.json', '')
            
            print(f"  - Processing {video_name}/{clip_name}...")
            
            with open(src_json_path, 'r') as f:
                data = json.load(f)
            
            # Map image_id to frame_id
            image_id_to_frame = {img['id']: img['frame_id'] for img in data['images']}
            
            # Update annotations
            count_special = 0
            for ann in data['annotations']:
                img_id = ann['image_id']
                frame_id = image_id_to_frame.get(img_id)
                track_id = ann.get('track_id')
                
                # Default behavior
                ann['action'] = "Lying"
                
                # Check CSV lookup
                lookup_key = (video_name, clip_name, frame_id, track_id)
                if lookup_key in behavior_lookup:
                    ann['action'] = behavior_lookup[lookup_key]
                    count_special += 1
            
            with open(dst_json_path, 'w') as f:
                json.dump(data, f)
            
            if count_special > 0:
                print(f"    Added {count_special} special behavior labels.")

    print("\n>>> FINISHED. Labeled dataset is in:", dst_dir)

if __name__ == "__main__":
    CSV_FILE = "data/behavior.csv"
    SRC_DIR = "data/annotations/refined"
    DST_DIR = "data/annotations/behavior"
    
    add_behavior_labels(CSV_FILE, SRC_DIR, DST_DIR)
