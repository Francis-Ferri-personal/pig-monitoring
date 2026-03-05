import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import yaml
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import sys
from behavior.behavior_dataset import PigBehaviorDataset
from behavior.models import BehaviorRNN
from behavior.generate_behavior_reports import generate_full_report

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # Needed for python 3 compatibility
        pass

def plot_results(train_history, val_history, metric_name, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label='Train')
    plt.plot(val_history, label='Validation')
    plt.title(f'Pig Behavior Recognition - {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(args):
    # Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 1. Hyperparameters from config (can be overridden by args)
    rnn_type = args.rnn_type if args.rnn_type else config.get('rnn_type', 'LSTM')
    epochs = args.epochs if args.epochs else config.get('epochs', 20)
    lr = args.lr if args.lr else config.get('learning_rate', 0.001)
    hidden_size = args.hidden_size if args.hidden_size else config.get('hidden_size', 64)
    num_layers = args.num_layers if args.num_layers else config.get('num_layers', 2)
    batch_size = args.batch_size if args.batch_size else config.get('batch_size', 32)
    
    # Dynamically calculate input size: 6 (bbox) + 3 * num_keypoints
    num_kpts = len(config.get('keypoints_to_use', []))
    input_size = 6 + (num_kpts * 3)

    # 2. Setup Experiment Folder
    exp_name = f"{rnn_type}-{num_kpts}_points-{epochs}_epoch"
    exp_dir = os.path.join("out", "results", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    shutil.copy("config.yaml", os.path.join(exp_dir, "config_used.yaml"))

    # Started logging
    sys.stdout = Logger(os.path.join(exp_dir, "train.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> Experiment: {exp_name}")
    print(f">>> Device: {device} | Input Size: {input_size}")

    # 3. Data Loaders
    FEAT_DIR = "data/features"
    train_vids = ['video1', 'video2']
    val_vids = ['video3']
    
    window_size = config.get('window_size', 30)
    stride = config.get('stride_train', 5)
    
    train_ds = PigBehaviorDataset(FEAT_DIR, train_vids, window_size=window_size, stride=stride)
    val_ds = PigBehaviorDataset(FEAT_DIR, val_vids, window_size=window_size, stride=stride)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    class_names = list(config.get('behavior_classes', {}).keys())
    
    # 4. Initialize Model
    model = BehaviorRNN(
        rnn_type=rnn_type, 
        input_size=input_size, 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        num_classes=len(class_names)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5. Training Loop
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = -1.0
    model_save_path = os.path.join(exp_dir, "best_model.pt")

    print(f">>> Starting Training...")
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
        epoch_loss = train_loss / len(train_ds)
        epoch_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss_total, val_correct, val_total = 0.0, 0, 0
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                v_loss = criterion(outputs, y)
                val_loss_total += v_loss.item() * x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(y.cpu().numpy())
        
        val_acc = 100 * val_correct / val_total
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss_total / len(val_ds))
        history['train_acc'].append(epoch_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            best_preds, best_labels = all_val_preds, all_val_labels
            print(f"    [Saved Best Model]")

    # 6. Final Plots & Summary
    plot_results(history['train_loss'], history['val_loss'], 'Loss', os.path.join(exp_dir, "loss_curve.png"))
    plot_results(history['train_acc'], history['val_acc'], 'Accuracy', os.path.join(exp_dir, "accuracy_curve.png"))
    plot_confusion_matrix(best_labels, best_preds, class_names, os.path.join(exp_dir, "confusion_matrix.png"))

    summary_text = f"Exp: {exp_name}\nRNN: {rnn_type}\nEpochs: {epochs}\nLR: {lr}\nKpts: {num_kpts}\nBest Val Acc: {best_val_acc:.2f}%"
    with open(os.path.join(exp_dir, "summary.txt"), "w") as f: f.write(summary_text)

    print(f"\n>>> Generating detailed clip reports...")
    generate_full_report(exp_name, video_to_eval='video3')
    
    print(f"\n>>> FINISHED. Results in: {exp_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn_type", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()
    train_model(args)
