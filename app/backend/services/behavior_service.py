import csv
import logging
import os
from collections import defaultdict
from typing import Dict, Any, Tuple
import numpy as np
import torch

from utils.model import BehaviorRNN

MODEL_PATH = "model/model.pt"
WINDOW_SIZE = 5
BEHAVIOR_CLASSES = ["Lying", "Sitting", "Standing_Walking", "Feeding", "Drinking"]

logger = logging.getLogger(__name__)


class BehaviorPredictionService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model_once()

    def _load_model_once(self) -> BehaviorRNN:
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

    def _process_track_features(
        self, 
        feats: np.ndarray, 
        frames: np.ndarray, 
        track_id: int, 
        counts: dict, 
        pigs_predictions: Dict[int, Dict[int, int]]
    ):
        """Genera inferencia cuadro a cuadro y acumula conteos."""
        num_frames = feats.shape[0]
        if num_frames < WINDOW_SIZE:
            return

        if track_id not in pigs_predictions:
            pigs_predictions[track_id] = {}

        with torch.no_grad():
            for i in range(num_frames - WINDOW_SIZE + 1):
                window = (
                    torch.from_numpy(feats[i : i + WINDOW_SIZE])
                    .float()
                    .unsqueeze(0)
                    .to(self.device)
                )
                output = self.model(window)
                pred = torch.argmax(output, dim=1).item()

                # El frame al que se asigna la predicción de la ventana
                frame_idx = int(frames[i + WINDOW_SIZE - 1]) if frames is not None else (i + WINDOW_SIZE - 1)
                
                # Mapear frame exacto con su clase predicha
                pigs_predictions[track_id][frame_idx] = pred
                counts[track_id][pred] += 1

    def predict_and_count(self, session_id: str, coco_data: Dict[str, Any]) -> Tuple[str, Dict[int, Dict[int, int]]]:
        """
        Ejecuta inferencia y retorna:
        1. La ruta del CSV de conteos.
        2. El diccionario de predicciones cuadro por cuadro: {track_id: {frame_idx: class_idx}}.
        """
        if self.model is None:
            self.model = self._load_model_once()
            if self.model is None:
                raise FileNotFoundError(f"Model checkpoint missing at absolute path: {MODEL_PATH}")

        npz_path = os.path.join("data", "features", f"{session_id}_features.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Feature file not found at: {npz_path}")

        counts = defaultdict(lambda: defaultdict(int))
        pigs_predictions: Dict[int, Dict[int, int]] = {}

        logger.info(f"Running behavior inference for session: {session_id}")
        npz_data = np.load(npz_path, allow_pickle=True)
        keys_found = list(npz_data.files)

        # Caso 1: Estructura anidada 'tracks'
        if "tracks" in npz_data:
            tracks_dict = npz_data["tracks"].item()
            for track_id, data_dict in tracks_dict.items():
                feats = data_dict.get("features", data_dict) if isinstance(data_dict, dict) else data_dict
                frames = data_dict.get("frames", None) if isinstance(data_dict, dict) else None
                self._process_track_features(feats, frames, int(track_id), counts, pigs_predictions)

        # Caso 2: Claves separadas track_X_features y track_X_frames
        else:
            for key in keys_found:
                if key.startswith("track_") and key.endswith("_features"):
                    track_id_str = key.replace("track_", "").replace("_features", "")
                    if not track_id_str.isdigit():
                        continue
                    
                    track_id = int(track_id_str)
                    feats = npz_data[key]
                    
                    frames_key = f"track_{track_id}_frames"
                    frames = npz_data[frames_key] if frames_key in npz_data else None
                    
                    self._process_track_features(feats, frames, track_id, counts, pigs_predictions)

        # Exportar los conteos predichos al CSV
        out_dir = os.path.join("data", "out", "predictions", session_id)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{session_id}_counts.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id"] + BEHAVIOR_CLASSES)

            for tid in sorted(counts.keys()):
                row = [tid] + [counts[tid][cls_idx] for cls_idx in range(len(BEHAVIOR_CLASSES))]
                writer.writerow(row)

        logger.info(f"Predictions saved to: {csv_path} (Tracks procesados: {len(counts)})")
        return csv_path, pigs_predictions