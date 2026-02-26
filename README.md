# Pig Monitoring

Project for monitoring and tracking pig behavior from video feeds.

## Setup

1. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**:
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
