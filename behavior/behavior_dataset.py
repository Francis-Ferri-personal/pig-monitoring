import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PigBehaviorDataset(Dataset):
    def __init__(self, feature_dir, video_list, window_size=30, stride=5):
        """
        Args:
            feature_dir (str): Root directory of extracted .npz features.
            video_list (list): List of video names to include (e.g. ['video1', 'video2']).
            window_size (int): Number of frames per sequence.
            stride (int): How many frames to skip between sequences (overlap).
        """
        self.window_size = window_size
        self.sequences = []
        self.labels = []

        print(f">>> Loading dataset for {video_list}...")
        
        for video in video_list:
            video_path = os.path.join(feature_dir, video)
            if not os.path.exists(video_path):
                print(f"!!! Warning: Video path {video_path} not found.")
                continue

            for npz_file in os.listdir(video_path):
                if not npz_file.endswith('.npz'):
                    continue
                
                data = np.load(os.path.join(video_path, npz_file))
                feats = data['features'] # [num_frames, 57]
                lbls = data['labels']    # [num_frames]
                
                # Sliding window
                num_frames = feats.shape[0]
                if num_frames < window_size:
                    # Optional: Could pad, but for training it's better to skip very short tracks
                    continue
                
                for i in range(0, num_frames - window_size + 1, stride):
                    window = feats[i : i + window_size]
                    label = lbls[i + window_size - 1] # Target is the action of the last frame
                    
                    self.sequences.append(window)
                    self.labels.append(label)

        print(f"    Loaded {len(self.sequences)} sequences.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Convert to float32 tensors
        x = torch.from_numpy(self.sequences[idx]).float()
        y = torch.tensor(self.labels[idx]).long()
        return x, y

if __name__ == "__main__":
    # Test loading
    from torch.utils.data import DataLoader
    
    FEAT_DIR = "data/features"
    train_vids = ['video1', 'video2']
    
    dataset = PigBehaviorDataset(FEAT_DIR, train_vids, window_size=30, stride=5)
    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        x, y = next(iter(loader))
        print("Batch X shape:", x.shape) # Expected: [8, 30, 57]
        print("Batch Y shape:", y.shape) # Expected: [8]
