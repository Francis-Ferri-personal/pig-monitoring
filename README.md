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
python tools/test.py --device cuda:1 --batch-size 32
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

**3. Delete Erroneous Detections (Cleanup)**
Perform final cleaning *after* remapping. If you find an incorrect track (e.g., in video 1, clip 06, the track with ID 7 is a false positive), remove it:
```bash
# Note: Always run this after remapping, as remapping resets the file.
python utils/annotation_manager.py delete-id --video 1 --clip 06 --id 7
```

> ⚠️ **Important**: If you change the default configurations (like splitting parameters or padding), you MUST update `data/anns-remap.json` and delete incorrect IDs to match the new results.

# Behavior Recognition System Overview

This system implements a deep learning pipeline to classify pig behaviors into five categories: **Lying, Sitting, Standing, Walking, and Feeding**.

## 1. Feature Engineering
Instead of raw pixels, our model uses a compact high-level representation for each pig in every frame:
- **Geometry Features (6D)**: Normalized BBox area, Aspect Ratio, Normalized Center coordinates ($x, y$), and Normalized Dimensions ($w, h$).
- **Pose Features (17 keypoints x 3)**: For each keypoint (ears, nose, shoulders, hips, etc.):
  - Relative coordinates $(x, y)$ normalized to the individual pig's BBox.
  - Visibility-based confidence score ($v=2 \to 1.0$, $v=1 \to 0.3$, $v=0 \to 0.0$).

**Total Input Size**: $6 + (17 \times 3) = 57$ features per frame.

## 2. Temporal Processing
- **Sliding Window**: We use a window size of **30 frames** (approx. 1 second at 30 fps) to capture the temporal dynamics of movement and posture.
- **LSTM (Long Short-Term Memory)**: We use a 2-layer LSTM network with 128 hidden units per layer. This architecture is ideal for sequential data where the transition between states (e.g., Standing to Walking) is crucial.

## 3. Training & Balancing
The dataset is highly imbalanced (pigs spend most of their time Lying).
- **Class Weighting**: We automatically calculate class weights based on inverse frequency.
- **Loss Function**: `CrossEntropyLoss` is applied with these weights to penalize the model more heavily for misclassifying rare behaviors (like Feeding or Sitting).
- **Split Strategy**: **Video 1 & 2** are used for training (diverse environments), while **Video 3** is reserved for strictly independent validation/testing.

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
Convert the behavioral dataset into compact numerical tensors for model training.

```bash
python behavior/feature_extractor.py
```

**How it works:**
- **Input**: Processes JSONs in `data/annotations/behavior/`.
- **Processing**: 
  - Extracts **bbox and normalized pose features** per pig/frame.
  - Maps visibility to confidence: $v=2 \to 1.0$, $v=1 \to 0.3$.
  - Organizes data temporally per `track_id`.
- **Output**: Compressed NumPy files (`.npz`) in `data/features/{video_dir}/track_{id}.npz`.

### Step 9: Model Training, Evaluation & Reports
Train the behavior recognition model (RNN/LSTM/GRU) and automatically generate performance reports.

```bash
python behavior/train_behavior.py
```

**How it works:**
- **Input**: Uses `.npz` files from `data/features/` and hyperparameters from `config.yaml`.
- **Training Strategy**: Uses **Video 1 & 2** for training, and **Video 3** for strictly independent validation.
- **Processing**: 
  - Trains the RNN model based on the sequence dynamics.
  - Automatically selects properties (Learning Rate, Architecture, etc.) from `config.yaml`.
- **Outputs** (Saved in `out/results/{experiment_name}`):
  - `best_model.pt`: The trained model weights.
  - `train.log`: Full execution logs of the training process.
  - `loss_curve.png` & `accuracy_curve.png`: Training progress visualization.
  - `confusion_matrix.png`: Model performance per class.
  - `clip_reports/`: Detailed 180-frame behavioral timelines for all pigs in a single frame.

### Step 10: Validation Video Generation
Generate prediction overlay videos to visually evaluate the model's performance on the validation set (`video3`).

```bash
python behavior/generate_behavior_videos.py --exp LSTM-17_points-30_epoch
```

**How it works:**
- **Input**: Uses the trained model (`best_model.pt`) from the specified `--exp` folder and the extracted features for `video3`.
- **Processing**: 
  - Runs frame-by-frame inference for all pigs in the validation clips.
  - Overlays the **Predicted Label** and **Ground Truth Label** alongside the bounding box on the original frames.
- **Outputs**: Generates `.mp4` video clips in `out/results/{experiment_name}/videos/video3/` allowing you to visually verify the model's behavior recognition accuracy in real-time.

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

**Mode 2: Pose Keypoints (Skeletons)**
```bash
python utils/video_generator.py --all --pose
```
- **Output**: `out/videos/pose/videoX/{clip_id}.mp4`

**Mode 3: SAM Masks (Segmentation)**
```bash
python utils/video_generator.py --all --sam
```
- **Output**: `out/videos/sam/videoX/{clip_id}.mp4`

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
