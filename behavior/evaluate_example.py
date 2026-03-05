import torch
import numpy as np
import os
import yaml
from behavior.behavior_dataset import PigBehaviorDataset
from behavior.models import BehaviorRNN

def evaluate_example(exp_name):
    # 1. Load config from the experiment
    exp_dir = os.path.join("out", "results", exp_name)
    config_path = os.path.join(exp_dir, "config_used.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 2. Setup Device and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Map class IDs back to names
    id_to_action = {v: k for k, v in config['behavior_classes'].items()}
    num_classes = len(id_to_action)
    
    model = BehaviorRNN(
        rnn_type=config.get('rnn_type', 'LSTM'),
        input_size=57, 
        hidden_size=config.get('hidden_size', 64),
        num_layers=config.get('num_layers', 2),
        num_classes=num_classes
    ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pt")))
    model.eval()
    
    # 3. Load Validation Dataset (Video 3)
    val_vids = ['video3']
    dataset = PigBehaviorDataset("data/features", val_vids, 
                                 window_size=config.get('window_size', 30), 
                                 stride=10) # Using larger stride for variety
    
    print(f"\n>>> Evaluating Example on {len(dataset)} validation sequences...")
    print("-" * 60)
    print(f"{'Sample #':<10} | {'Predicted Action':<20} | {'Actual Action':<20} | {'Result':<10}")
    print("-" * 60)
    
    correct_count = 0
    num_to_show = 15
    indices = np.random.choice(len(dataset), min(num_to_show, len(dataset)), replace=False)
    
    with torch.no_grad():
        for i in indices:
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device) # Add batch dimension
            
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            
            pred_name = id_to_action[predicted.item()]
            actual_name = id_to_action[y.item()]
            status = "✅ MATCH" if predicted.item() == y.item() else "❌ MISMATCH"
            
            if predicted.item() == y.item(): correct_count += 1
            print(f"{i:<10} | {pred_name:<20} | {actual_name:<20} | {status}")
            
    print("-" * 60)
    print(f">>> Accuracy on this random batch: {(correct_count/num_to_show)*100:.2f}%")

if __name__ == "__main__":
    # Point to the last experiment folder name
    # You can change this to any folder in results/
    LATEST_EXP = "LSTM-17_points-20_epoch"
    evaluate_example(LATEST_EXP)
