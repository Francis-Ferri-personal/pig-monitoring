# Dataset Annotations

This directory contains the files related to the videos, frames, segmentation masks, pose keypoints, and behavior annotations.

## Behavior Classes

The dataset contains the following target behavior classes. The manual annotations were extracted from `pig-actions.xlsx`, normalized, and combined (e.g., merging "Standing" and "Walking" into a single class "Standing_Walking").

| ID | Class Name | Total Annotations (Frames) | Percentage |
| :---: | :--- | :---: | :---: |
| 0 | Lying | 17,996 | 66.7% |
| 1 | Sitting | 710 | 2.6% |
| 2 | Standing_Walking | 5,354 | 19.8% |
| 3 | Feeding | 2,627 | 9.7% |
| 4 | Drinking | 313 | 1.2% |

*Total annotated rows: 27,000*

**Process:**
1. Annotations start in `pig-actions.xlsx`.
2. Normalized to `behavior.csv` via the formatting script (`xlsx_to_behavior_csv.py`).
3. Added to the COCO JSONs in `annotations/behavior/` via `behavior/add_behavior_labels.py`.
4. Extracted as numeric features in `features/` via `behavior/feature_extractor.py`.
