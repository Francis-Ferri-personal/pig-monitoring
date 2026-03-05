import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class PigVisualBehaviorDataset(Dataset):
    def __init__(
        self,
        feature_dir: str,
        video_list: List[str],
        window_size: int = 30,
        stride: int = 5,
        balance_data: bool = False,
    ):
        """
        Dataset for visual embedding sequences.

        Args:
            feature_dir (str): Root directory of extracted .npz visual features.
            video_list (list): List of video names to include (e.g. ['video1', 'video2']).
            window_size (int): Number of frames per sequence.
            stride (int): How many frames to skip between sequences (overlap).
            balance_data (bool): If True, randomly undersamples the majority classes.
        """
        self.window_size = window_size
        self.sequences = []
        self.labels = []

        print(f">>> Loading VISUAL dataset for {video_list}...")

        for video in video_list:
            video_path = os.path.join(feature_dir, video)
            if not os.path.exists(video_path):
                print(f"!!! Warning: Video path {video_path} not found.")
                continue

            for npz_file in os.listdir(video_path):
                if not npz_file.endswith(".npz"):
                    continue

                data = np.load(os.path.join(video_path, npz_file))
                feats = data["features"]  # [num_frames, D]
                lbls = data["labels"]  # [num_frames]

                num_frames = feats.shape[0]
                if num_frames < window_size:
                    # Skip very short tracks
                    continue

                for i in range(0, num_frames - window_size + 1, stride):
                    window = feats[i : i + window_size]
                    label = lbls[i + window_size - 1]  # Target is the action of the last frame

                    self.sequences.append(window)
                    self.labels.append(label)

        print(f"    Loaded {len(self.sequences)} visual sequences.")

        if balance_data and len(self.labels) > 0:
            import random
            from collections import Counter

            counts = Counter(self.labels)
            target_size = int(np.median(list(counts.values())))

            print(f"    Balancing visual dataset to max {target_size} sequences per class...")

            balanced_seqs = []
            balanced_lbls = []

            indices_by_class = {c: [] for c in counts.keys()}
            for i, lbl in enumerate(self.labels):
                indices_by_class[lbl].append(i)

            for class_id, indices in indices_by_class.items():
                if len(indices) > target_size:
                    selected_indices = random.sample(indices, target_size)
                else:
                    selected_indices = indices

                for idx in selected_indices:
                    balanced_seqs.append(self.sequences[idx])
                    balanced_lbls.append(self.labels[idx])

            self.sequences = balanced_seqs
            self.labels = balanced_lbls
            print(f"    Visual dataset balanced. Total sequences: {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.sequences[idx]).float()
        y = torch.tensor(self.labels[idx]).long()
        return x, y


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    FEAT_DIR = "data/visual_features"
    train_vids = ["video1", "video2"]

    dataset = PigVisualBehaviorDataset(FEAT_DIR, train_vids, window_size=30, stride=5)
    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        x, y = next(iter(loader))
        print("Batch X shape:", x.shape)  # Expected: [8, 30, D]
        print("Batch Y shape:", y.shape)  # Expected: [8]

