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

3. **Install Other Dependencies**:
   This will install all project requirements, including the local **SAM 3** package with its notebook dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   > ⚠️ **Important**: Before using SAM 3, please request access to the checkpoints on the [SAM 3 Hugging Face repo](https://huggingface.co/meta-sam/sam3). Once accepted, you need to be authenticated to download the checkpoints. You can do this by following the [Hugging Face Hub authentication guide](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication):
   > 1. Generate an access token on Hugging Face.
   > 2. Run `hf auth login` (or `huggingface-cli login`) in your terminal and enter your token.  
   >
   > **Note**: The token needs the permission: _"Read access to contents of all public gated repos you can access"_

4. **Configuration**:
   Modify `config.yaml` to specify your input/output folders and desired clip duration.

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
