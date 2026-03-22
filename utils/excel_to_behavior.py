import pandas as pd
import os
import re

# Mapping based on Ear Tags (Color/Mark) from README
pig_mapping = {
    "Yellow K": 0,
    "Green 555": 1,
    "Red NN": 2,
    "Teal YY": 3,
    "Tan 66": 4
}

# Behavior naming consistency based on template.csv
behavior_map_raw = {
    "standing/ walking": "Standing_Walking",
    "standing/walking": "Standing_Walking",
    "lying": "Lying",
    "feeding": "Feeding",
    "sitting": "Sitting",
    "standing": "Standing",
    "walking": "Walking",
    "drinking": "Drinking" # Keep as is, but standardized
}

# Excluded ranges and full clips from Video 4 (poor quality)
excluded_ranges = {
    4: {  # Video 4
        "04": (112, 179),
        "07": (73, 179)
    }
}
excluded_clips = {
    4: ["05", "06"] # Video 4
}

def clean_behavior(b):
    if not isinstance(b, str): return b
    b = b.strip().lower()
    # Check for substring matches or direct map
    if "standing" in b and "walking" in b:
        return "Standing_Walking"
    return behavior_map_raw.get(b, b.capitalize())

def process_excel(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f">>> Reading {input_path}...")
    xls = pd.ExcelFile(input_path)
    all_data = []

    for sheet in xls.sheet_names:
        # Match pattern: Vvideo XX_clip YY or video XX_clip YY (with potential spaces)
        match = re.search(r"video\s*(\d+)_clip\s*(\d+)", sheet, re.IGNORECASE)
        if not match:
            # print(f"Skipping sheet: {sheet}")
            continue
        
        video_num = int(match.group(1))
        clip_id = match.group(2).zfill(2)
        video_key = f"video{video_num}"
        
        # Skip fully excluded clips
        if video_num in excluded_clips and clip_id in excluded_clips[video_num]:
            # print(f"Skipping excluded clip: {video_key}_{clip_id}")
            continue

        df = xls.parse(sheet)
        
        # Clean col names
        df.columns = [str(c).strip() for c in df.columns]

        # Standardize column names
        if 'Frame' not in df.columns:
            if 'frame' in df.columns: df = df.rename(columns={'frame': 'Frame'})
            else: continue
                
        if 'Pig_ID' not in df.columns:
            if 'Pig' in df.columns: df = df.rename(columns={'Pig': 'Pig_ID'})
            else: continue
        
        if 'Behavior' not in df.columns:
            if 'Action' in df.columns: df = df.rename(columns={'Action': 'Behavior'})
            elif 'behavior' in df.columns: df = df.rename(columns={'behavior': 'Behavior'})
            else: continue

        # Filter excluded frames in Video 4
        if video_num in excluded_ranges and clip_id in excluded_ranges[video_num]:
            start, end = excluded_ranges[video_num][clip_id]
            df = df[~((df['Frame'] >= start) & (df['Frame'] <= end))]

        # Filter out rows with NaN Frame
        df = df[df['Frame'].notna()]

        for _, row in df.iterrows():
            pig_tag = str(row['Pig_ID']).strip()
            if pig_tag not in pig_mapping:
                continue
            
            pig_id = pig_mapping[pig_tag]
            behavior = clean_behavior(row['Behavior'])
            
            # Timestamp handling
            ts = row.get('Timestamp', "")
            if pd.isna(ts): ts = ""
            elif hasattr(ts, 'strftime'): ts = ts.strftime('%H:%M:%S')
            else: ts = str(ts)
            
            all_data.append({
                'timestamp': ts,
                'video': video_key,
                'clip': clip_id,
                'frame': int(row['Frame']),
                'id': pig_id,
                'behavior': behavior
            })

    output_df = pd.DataFrame(all_data)
    if not output_df.empty:
        # Sort values to match logical ordering
        output_df = output_df.sort_values(['video', 'clip', 'frame', 'id'])
        output_df.to_csv(output_path, index=False)
        print(f"✓ Success! Processed {len(output_df)} behavior labels into {output_path}")
    else:
        print("--- Error: No data found to process.")

if __name__ == "__main__":
    process_excel('data/pig-actions.xlsx', 'data/behavior.csv')
