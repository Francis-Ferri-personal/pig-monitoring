import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from behavior.behavior_dataset import PigBehaviorDataset
from behavior.behavior_lstm import behaviorLSTM

def evaluate_model():
    # 1. Load Configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    FEAT_DIR = "data/features"
    val_videos = ['video3'] # Evaluation target
    model_path = "out/models/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"!!! Error: Model file '{model_path}' not found. Run training first.")
        return

    window_size = config.get('window_size', 30)
    behavior_classes = config.get('behavior_classes', {})
    num_classes = len(behavior_classes)
    
    # Calculate input size
    kp_to_use = config.get('keypoints_to_use', [])
    input_size = 6 + (len(kp_to_use) * 3)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Evaluating on device: {device}")

    # 2. Setup DataLoader for Validation
    val_dataset = PigBehaviorDataset(FEAT_DIR, val_videos, window_size=window_size, stride=window_size)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    if len(val_dataset) == 0:
        print("!!! Error: Validation dataset is empty.")
        return

    # 3. Load Model
    model = behaviorLSTM(input_size=input_size, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Collection predictions
    all_preds = []
    all_labels = []
    
    print(f">>> Running inference on {len(val_dataset)} sequences...")
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    # 5. Print and Save Detailed Metrics
    class_names = list(behavior_classes.keys())
    class_indices = list(behavior_classes.values())
    
    report = classification_report(all_labels, all_preds, labels=class_indices, target_names=class_names, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=class_indices)
    
    header = "\n" + "="*60 + "\n      DETAILED BEHAVIOR RECOGNITION REPORT (VIDEO 3)\n" + "="*60 + "\n"
    cm_header = "\nConfusion Matrix:\nRows: Actual | Columns: Predicted\n"
    
    full_output = header + report + cm_header + str(cm) + "\n" + "="*60 + "\n"
    
    # print to terminal
    print(full_output)
    
    # save to file
    results_dir = "out/results"
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(full_output)
    
    print(f"\n>>> REPORT SAVED to: {report_path}")

if __name__ == "__main__":
    evaluate_model()
