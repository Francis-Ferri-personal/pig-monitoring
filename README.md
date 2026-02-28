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
#### 3. Generate a Video from Annotations
To create a "slideshow" video where each annotated frame is shown for a second (matching the clip duration if captured at 1 FPS):

```bash
python utils/video_generator.py --video 1 --clip 01 --pose --fps 1
```

**Arguments:**
- `--video`: Video ID.
- `--clip`: Clip ID.
- `--pose`: (Optional) Use Pose annotations.
- `--fps`: (Optional) Output video frames per second. Default is 1 (each frame stays for 1s).
- `--output`: (Optional) Path to save the `.mp4` video.
