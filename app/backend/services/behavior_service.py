import csv
import logging
import os
from collections import defaultdict
from typing import Dict, Any
import numpy as np
import torch

from utils.model import BehaviorRNN

# Base directory setup relative to project root
MODEL_PATH = "model/model.pt"

WINDOW_SIZE = 30
BEHAVIOR_CLASSES = ["eating", "drinking", "standing", "lying", "moving"]

logger = logging.getLogger(__name__)


class BehaviorPredictionService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model_once()

    def _load_model_once(self) -> BehaviorRNN:
        """Loads and holds the BiLSTM model weights in GPU/CPU memory upon instantiation."""
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model checkpoint not found at: {MODEL_PATH}")
            return None

        model = BehaviorRNN(
            input_size=563,
            geom_dim=11,
            hidden_size=128,
            num_layers=2,
            num_classes=len(BEHAVIOR_CLASSES),
        ).to(self.device)

        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        model.eval()
        logger.info(f"BiLSTM Behavior model pre-loaded successfully on {self.device}")
        return model

    def _process_track_features(self, feats: np.ndarray, track_id: int, counts: dict):
        """Runs sliding window inference over a sequence of features for a single track."""
        if feats.shape[0] < WINDOW_SIZE:
            return

        with torch.no_grad():
            for i in range(feats.shape[0] - WINDOW_SIZE + 1):
                window = (
                    torch.from_numpy(feats[i : i + WINDOW_SIZE])
                    .float()
                    .unsqueeze(0)
                    .to(self.device)
                )
                output = self.model(window)
                pred = torch.argmax(output, dim=1).item()
                counts[track_id][pred] += 1

    def predict_and_count(self, session_id: str, coco_data: Dict[str, Any]) -> str:
        """Reads features from a single session .npz file, runs inference, and writes CSV counts."""
        if self.model is None:
            self.model = self._load_model_once()
            if self.model is None:
                raise FileNotFoundError(f"Model checkpoint missing at absolute path: {MODEL_PATH}")

        # Path to single multi-modal features .npz file
        npz_path = os.path.join("data", "features", f"{session_id}_features.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Feature file not found at: {npz_path}")

        counts = defaultdict(lambda: defaultdict(int))
        logger.info(f"Running behavior inference for session: {session_id}")

        npz_data = np.load(npz_path, allow_pickle=True)

        # Case 1: Features stored in a nested dictionary under 'tracks'
        if "tracks" in npz_data:
            tracks_dict = npz_data["tracks"].item()
            for track_id, feats in tracks_dict.items():
                self._process_track_features(feats, int(track_id), counts)

        # Case 2: Features stored with keys corresponding to track IDs (e.g., 'track_1' or '1')
        else:
            for key in npz_data.files:
                clean_key = key.replace("track_", "")
                if not clean_key.isdigit():
                    continue
                
                track_id = int(clean_key)
                feats = npz_data[key]
                self._process_track_features(feats, track_id, counts)

        # Export prediction counts to CSV
        out_dir = os.path.join("data", "out", "predictions", session_id)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{session_id}_counts.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id"] + BEHAVIOR_CLASSES)

            for tid in sorted(counts.keys()):
                row = [tid] + [counts[tid][cls_idx] for cls_idx in range(len(BEHAVIOR_CLASSES))]
                writer.writerow(row)

        logger.info(f"Predictions saved to: {csv_path}")
        return csv_path