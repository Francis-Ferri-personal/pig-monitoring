# Pig Monitoring

Project for monitoring and tracking pig behavior from video feeds.

## Setup

1. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install PyTorch**:
   Install the specific version for CUDA 12.6:
   ```bash
   pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

3. **Load Git Submodules**:
   This project uses Git submodules for **SAM 3** and **MMPose**. 
   - **New Clones**: Use `git clone --recursive [URL]`
   - **Existing Clones**: If you already cloned the repo without submodules, run:
     ```bash
     git submodule update --init --recursive
     ```

4. **Install Other Dependencies**:
   This will install all project requirements, including the local **SAM 3** package with its notebook dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   > ⚠️ **Important**: Before using SAM 3, please request access to the checkpoints on the [SAM 3 Hugging Face repo](https://huggingface.co/meta-sam/sam3). Once accepted, you need to be authenticated to download the checkpoints. You can do this by following the [Hugging Face Hub authentication guide](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication):
   > 1. Generate an access token on Hugging Face.
   > 2. Run `hf auth login` (or `huggingface-cli login`) in your terminal and enter your token.  
   >
   > **Note**: The token needs the permission: _"Read access to contents of all public gated repos you can access"_

5. **Configuration**:
   Modify `config.yaml` to specify your input/output folders and desired clip duration.

---

## Secondary Setup: MMPose (Pose Estimation)

If you need to perform pose estimation, set up the dedicated environment for **MMPose**:

```bash
# 1. Create dedicated conda environment
conda create --prefix ./.venv-pose python=3.8 -y
conda activate ./.venv-pose

# 2. Install PyTorch for MMPose (CUDA 12.4)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install OpenMMLab tools
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0" # Note: wheel building takes time
mim install "mmdet>=3.1.0"

# 4. Install MMPose from source
cd mmpose
pip install -r requirements.txt
pip install -v -e .
cd ..
```

---

## Workflow

### Step 1: Split Videos into Clips
Use the `utils/video-splitter.py` script to batch-process raw videos into shorter segments for easier analysis.

```bash
python utils/video-splitter.py
```

**How it works:**
- **Input**: Reads `.mp4` files from the directory defined in `videos_folder` (default: `data/videos/raw`).
- **Processing**: Splits each video into segments based on `clip_duration_minutes` in `config.yaml`.
- **Output**: Creates a subfolder for each original video in `clips_folder` (default: `data/videos/clips`) containing the numbered clips (e.g., `01.mp4`).

### Step 2: Extract Frames from Clips
Use the `utils/clip-splitter.py` script to extract individual frames from the previously generated clips.

```bash
python utils/clip-splitter.py
```

**How it works:**
- **Input**: Processes all `.mp4` clips found in `clips_folder` (default: `data/videos/clips`).
- **Processing**: Extracts frames at a rate specified by `frames_per_second` in `config.yaml`.
- **Output**: Generates images in `frames_folder` (default: `data/images/frames`), maintaining a folder structure that matches the video and clip names (e.g., `data/images/frames/Video_01/01/00000.png`).

### Step 3: Apply Mask to Frames
Use the `utils/apply_mask.py` script to apply a static mask to all extracted frames. This is useful for obscuring areas of the video that are not relevant to the monitoring task.

```bash
python utils/apply_mask.py --mask data/images/mask.png --input data/images/frames --output data/images/frames_masked
```

**How it works:**
- **Input**: Processes all images found in the `--input` directory (default: `data/images/frames`).
- **Processing**: Applies the bitwise AND operation between the frame and the `--mask` PNG file. Areas that are black in the mask will become black in the output frames.
- **Output**: Generates masked images in the `--output` directory (default: `data/images/frames_masked`), maintaining the original subfolder structure.

### Step 4: Batch Annotation Generation (BBoxes & Tracking)
Use the `tools/gen_anns_videos.py` script to automatically detect and track pigs using SAM 3. This will generate the initial bounding boxes and segmentation masks.

```bash
python tools/gen_anns_videos.py --prompt "pig"
```

**How it works:**
- **Input**: Processes all images in `data/images/frames_masked/`.
- **Processing**: Uses SAM 3 to detect pigs based on the `--prompt` and tracks them across frames in each clip.
- **Output**: Generates COCO-compliant JSON files in `data/annotations/sam/{video_dir}/{clip_id}.json`.
- **Automatic Resume**: Skips clips that already have an annotation file.

### Step 5: Pose Estimation (Keypoints)
Once you have the SAM annotations, you can generate pose estimations (keypoints) for each detected pig using MMPose.

```bash
# Ensure you are using the pose environment
conda activate ./.venv-pose
# Note: Use tools/test.py if it exists, otherwise its logic is integrated into the workflow
python tools/gen_keypoint_anns.py --device cuda:1 --batch-size 32
```

**How it works:**
- **Input**: Reads existing SAM annotations from `data/annotations/sam/`.
- **Processing**: For each pig, it masks the background and runs **MMPose** inference. 
- **Output**: Generates new COCO-compliant JSON files in `data/annotations/pose/` including `keypoints` and `skeleton` metadata.

### Step 6: Refinement Process
The refinement process ensures that tracks are consistent across clips and erroneous detections are removed. This process operates on the `data/annotations/refined/` directory.

**1. Initialize Refined Directory**
Copies the pose annotations (BBoxes + Masks + Keypoints) to the refined folder.
```bash
python utils/annotation_manager.py init
```

**2. Apply Global ID Remapping**
Use the mapping configuration to ensure IDs are consistent throughout the entire dataset. This will **reset** the target clips from the `pose` original before applying the map.
```bash
python utils/annotation_manager.py remap --map data/anns-remap.json
```

**3. Remove Problematic Clips**
In Video 4, clips 05 and 06 are removed from the refined set to prevent errors during training:
```bash
rm data/annotations/refined/video4/05.json data/annotations/refined/video4/06.json
```

**4. Delete Erroneous Detections (Cleanup)**
Perform final cleaning *after* remapping. If you find an incorrect track (e.g., in video 1, clip 06, the track with ID 7 is a false positive), remove it:
```bash
# Note: Always run this after remapping, as remapping resets the file.
python utils/annotation_manager.py delete-id --video 1 --clip 06 --id 7
```

**5. Delete Erroneous Frames (Range-based Cleanup)**
If an entire segment of a clip has poor tracking or errors, remove all annotations in that frame range:
```bash
# Example Video 4 deletions
python utils/annotation_manager.py delete-frames --video 4 --clip 04 --start 112 --end 179
python utils/annotation_manager.py delete-frames --video 4 --clip 07 --start 73 --end 179
```

> ⚠️ **Important**: If you change the default configurations (like splitting parameters or padding), you MUST update `data/anns-remap.json` and delete incorrect IDs to match the new results.

# Behavior Recognition System Overview

This system implements a deep learning pipeline to classify pig behaviors into five categories: **Lying, Sitting, Standing, Walking, and Feeding**.

## 1. Feature Engineering
Instead of raw pixels, our model uses a compact high-level representation for each pig in every frame:
- **Geometry Features (11D)**: Normalized BBox area, Aspect Ratio, Normalized Center coordinates ($x, y$), Normalized Dimensions ($w, h$), and Enriched Motion ($dx, dy$, speed).
- **Engineered Pose Features (40D)** (Optional, toggle in `config.yaml`): 
  - Scale-invariant distances to **Neck** and **Tail** root points.
  - Body orientation (Neck-Tail vector angle).
  - Joint visibility scores.

**Total Input Size**: $512$ (CNN Embedding) $+ 11$ (Geometry) $+ 40$ (Keypoints) $= 563$ features per frame.

## 2. Temporal Processing
- **Sliding Window**: We use a window size of **5 frames** (approx. 1 second at 30 fps) to capture the temporal dynamics of movement and posture.
- **LSTM (Long Short-Term Memory)**: We use a 2-layer LSTM network with 128 hidden units per layer. This architecture is ideal for sequential data where the transition between states (e.g., Standing to Walking) is crucial.

## 3. Training & Balancing
The dataset is highly imbalanced (pigs spend most of their time Lying).
- **Class Weighting**: We automatically calculate class weights based on inverse frequency.
- **Loss Function**: `CrossEntropyLoss` is applied with these weights to penalize the model more heavily for misclassifying rare behaviors (like Feeding or Sitting).
- **Split Strategy**: **Video 1 & 2** are used for training (diverse environments), while **Video 3** is reserved for strictly independent validation/testing.

---

#### Convert Behavior Labels from Excel
Before integrating labels, convert the manual annotations from Excel to the required CSV format. This script also applies the necessary filters for problematic segments in Video 4 (excludes clips 05 and 06 entirely, and filters specific frame ranges in clips 04 and 07).

```bash
python utils/excel_to_behavior.py
```

- **Input**: `data/pig-actions.xlsx` 
- **Output**: `data/behavior.csv` (used in Step 7)
- **Features**: 
  - Standardizes behavior names (e.g., `Standing/ Walking` -> `Standing_Walking`).
  - Maps Ear Tags (Yellow K, etc.) to IDs (**0-4**).
  - Automatically filters out the excluded frame ranges in Video 4 to ensure data quality.

---

### Step 7: Behavior Labeling
Integrate manual behavior annotations from a CSV file into the refined dataset.

```bash
python behavior/add_behavior_labels.py
```

**How it works:**
- **Input**: Reads `data/behavior.csv` (schema: `timestamp,video,clip,frame,id,behavior`).
- **Processing**: Copies `data/annotations/refined` to `data/annotations/behavior`. Sets a default action (**Lying**) and applies specific labels from the CSV.
- **Output**: COCO JSONs in `data/annotations/behavior/` with a new `"action"` field.

### Step 8: Feature Extraction
Convert annotations into compact numerical tensors (NPZ) for training. The script extracts visual embeddings from a ResNet-18 backbone, bounding box geometry, and (optionally) engineered keypoints.

```bash
# Run inside the project virtualenv (.venv)
source .venv/bin/activate
python behavior/feature_extractor.py
```

- **Configuration**: Set `use_keypoints: True` in `config.yaml` to include engineered geometric features.
- **How it works**:
  - Resolves `file_name` paths robustly.
  - Normalizes keypoints to joint-to-joint distances centered on the neck/tail.
- **Output**: 
  - Without keypoints: `data/features/{video}/track_{id}.npz`
  - With keypoints: `data/features_kp/{video}/track_{id}.npz` (automatically handled by the script).

### Step 9: Model Training, Evaluation & Reports
Train the behavior recognition model. The script supports `RNN`, `LSTM`, `GRU`, and `BiLSTM` architectures.

```bash
python behavior/train_behavior.py --rnn_type BiLSTM --epochs 80
```

- **Input**: Reads from `data/features_kp` or `data/features` depending on the `use_keypoints` flag in `config.yaml`.
- **Logic**: Automatically balances classes using inverted frequency weights and uses the "Independent Video" split (Video 1 & 2 for train, Video 3 for validation).
- **Output**: Experiment folder under `out/results/` (e.g., `keypoints-BiLSTM-80_epoch`) containing `best_model.pt`, training logs, confusion matrices, and accuracy plots.


### Step 10: Validation Video Generation
Create overlay videos that show model predictions vs ground truth for inspection.

```bash
python behavior/generate_videos.py --exp keypoints-BiLSTM-80_epoch --video video3
```

- **Logic**: Uses the `best_model.pt` from the experiment folder to generate predictions frame-by-frame and overlays them on the original frames.
- **Output**: MP4 clips saved to `out/results/{experiment_name}/videos_visual/video3/{clip_id}.mp4`.

---

## Utilities

### Visualization Tool
You can verify the quality of the annotations (both SAM masks and Pose keypoints) by visualizing specific frames:

#### 1. Visualize SAM Masks (Segmentation)
```bash
python utils/viz_utils.py --video 1 --clip 01 --frame 100 --output segment_vis.png
```

#### 2. Visualize Pose Keypoints (Skeleton)
```bash
python utils/viz_utils.py --video 1 --clip 01 --frame 100 --pose --output pose_vis.png
```

**Arguments:**
- `--video`: The numeric ID of the video directory (e.g., `1` for `video1`).
- `--clip`: The clip file name (e.g., `01`).
- `--frame`: The specific frame index within that clip (e.g., `100`).
- `--pose`: (Optional) If enabled, loads from `data/annotations/pose` and draws skeletons.
- `--output`: (Optional) Path to save the resulting image.
- `--ann_dir`: (Optional) Custom path to annotations folder.
#### 3. Generate Videos (Raw, Pose, or SAM)
Create video files from the extracted frames. You can generate clean clips (raw), clips with pose skeletons, or clips with segmentation masks.

**Mode 1: Raw Clips (No annotations)**
```bash
python utils/video_generator.py --all
```
- **Output**: `out/videos/clip/videoX/{clip_id}.mp4`

**Mode 2: SAM Masks (Segmentation)**
```bash
python utils/video_generator.py --all --sam
```
- **Output**: `out/videos/sam/videoX/{clip_id}.mp4`

**Mode 3: Pose Keypoints (Skeletons)**
```bash
python utils/video_generator.py --all --pose
```
- **Output**: `out/videos/pose/videoX/{clip_id}.mp4`

**Mode 4: Refined Clips (Cleaned Tracking)**
```bash
python utils/video_generator.py --all --refined
```
- **Output**: `out/videos/refined/videoX/{clip_id}.mp4`

**Arguments:**
- `--all`: Process all available clips. Automatically skips existing output files (resume).
- `--video` & `--clip`: Target a specific clip instead of all.
- `--pose`: Use pose annotations.
- `--sam`: Use SAM annotations.
- `--refined`: Use refined annotations (from `data/annotations/refined`).
- `--fps`: Output frames per second (default: 1).
- `--output`: Custom output path (only for single clip mode).

#### 4. ID Mapping Configuration
To maintain consistency, use the following unique IDs based on ear tags:

| Ear Tag (Color/Mark) | Unique ID |
| :--- | :--- |
| Yellow K | **0** |
| Green 555 | **1** |
| Red NN | **2** |
| Teal YY | **3** |
| Tan 66 | **4** |

**Mapping JSON Structure**
The remapping tool (`--map`) expects a nested structure:
- **Video Level**: Groups clips by video folder.
- **Clip Level**: Defines remapping for individual JSON files.
- **Remap Level**: Allows range-based ID changes.

Example of `data/anns-remap.json`:
```json
[
  {
    "video": "video1",
    "clips": [
      {
        "clip": "06",
        "remaps": [
          {
            "frame_start": 103,
            "frame_end": 179,
            "remap": { "1": "5", "3": "2" }
          }
        ]
      }
    ]
  }
]
```
- `"1": "5"`: Changes the tracker's `track_id: 5` to our master `track_id: 1`.
- `"3": "2"`: Changes the tracker's `track_id: 2` to our master `track_id: 3`.
- Mapping keys must be strings; values are converted to integers in COCO.


