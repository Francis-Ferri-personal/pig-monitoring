import argparse
import os
import shutil
import sys
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from behavior.visual_dataset import PigVisualBehaviorDataset
from behavior.visual_models import VisualBehaviorRNN

matplotlib.use("Agg")


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def plot_results(train_history, val_history, metric_name, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label="Train")
    plt.plot(val_history, label="Validation")
    plt.title(f"Pig Behavior Recognition (Visual) - {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix (Visual Model)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_visual_model(args):
    # Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Hyperparameters
    rnn_type = args.rnn_type if args.rnn_type else config.get("rnn_type", "LSTM")
    epochs = args.epochs if args.epochs else config.get("epochs", 20)
    lr = args.lr if args.lr else config.get("learning_rate", 0.0005)
    hidden_size = args.hidden_size if args.hidden_size else config.get("hidden_size", 128)
    num_layers = args.num_layers if args.num_layers else config.get("num_layers", 2)
    batch_size = args.batch_size if args.batch_size else config.get("batch_size", 32)

    # Experiment folder
    exp_name = f"Visual-{rnn_type}-{epochs}_epoch"
    exp_dir = os.path.join("out", "results", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    shutil.copy("config.yaml", os.path.join(exp_dir, "config_used.yaml"))

    # Logging
    sys.stdout = Logger(os.path.join(exp_dir, "train.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> Visual Experiment: {exp_name}")
    print(f">>> Device: {device}")

    # Data loaders (visual features)
    FEAT_DIR = "data/visual_features"
    train_vids = ["video1", "video2"]
    val_vids = ["video3"]

    window_size = config.get("window_size", 30)
    stride = config.get("stride_train", 5)

    train_ds = PigVisualBehaviorDataset(
        FEAT_DIR,
        train_vids,
        window_size=window_size,
        stride=stride,
        balance_data=False,
    )
    val_ds = PigVisualBehaviorDataset(
        FEAT_DIR,
        val_vids,
        window_size=window_size,
        stride=stride,
        balance_data=False,
    )

    if len(train_ds) == 0:
        print("!!! Error: Visual training dataset is empty. Did you run visual_feature_extractor.py?")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    class_names = list(config.get("behavior_classes", {}).keys())

    # Infer input_size from one batch (CNN emb + geom + deltas)
    sample_x, _ = next(iter(train_loader))
    input_size = sample_x.shape[-1]
    print(f">>> Inferred input_size from visual features: {input_size}")

    # Model
    model = VisualBehaviorRNN(
        rnn_type=rnn_type,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=len(class_names),
    ).to(device)

    # Class weights
    label_counts = Counter(train_ds.labels)
    total_samples = sum(label_counts.values())

    class_weights = []
    for i in range(len(class_names)):
        count = label_counts.get(i, 1)
        weight = total_samples / (len(class_names) * count)
        class_weights.append(weight)

    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f">>> Class Weights: {np.round(weights_tensor.cpu().numpy(), 3)}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    model_save_path = os.path.join(exp_dir, "best_model.pt")

    print(">>> Starting Visual Training...")
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
        epoch_acc = 100.0 * correct / total

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

        val_loss = val_loss_total / max(1, len(val_ds))
        val_acc = 100.0 * val_correct / max(1, val_total)

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(epoch_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            best_preds, best_labels = list(all_val_preds), list(all_val_labels)
            print("    [Saved Best Visual Model]")

    # Plots & summary
    plot_results(
        history["train_loss"],
        history["val_loss"],
        "Loss",
        os.path.join(exp_dir, "loss_curve.png"),
    )
    plot_results(
        history["train_acc"],
        history["val_acc"],
        "Accuracy",
        os.path.join(exp_dir, "accuracy_curve.png"),
    )
    plot_confusion_matrix(
        best_labels,
        best_preds,
        class_names,
        os.path.join(exp_dir, "confusion_matrix.png"),
    )

    summary_text = (
        f"Exp: {exp_name}\n"
        f"RNN: {rnn_type}\n"
        f"Epochs: {epochs}\n"
        f"LR: {lr}\n"
        f"Hidden: {hidden_size}\n"
        f"Layers: {num_layers}\n"
        f"Best Val Acc: {best_val_acc:.2f}%\n"
        f"Input Size: {input_size}\n"
    )
    
    # Generate detailed classification report dict
    report_dict = classification_report(best_labels, best_preds, target_names=class_names, zero_division=0, output_dict=True)
    
    # Calculate Accuracy per class
    cm = confusion_matrix(best_labels, best_preds)
    total_samples_cm = np.sum(cm)
    accuracies = {}
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = total_samples_cm - (tp + fp + fn)
        accuracies[class_name] = (tp + tn) / total_samples_cm

    # Build custom report string
    report_str = "                   accuracy  precision    recall  f1-score   support\n\n"
    for class_name in class_names:
        metrics = report_dict[class_name]
        acc = accuracies[class_name]
        report_str += f"{class_name:>16}       {acc:.2f}       {metrics['precision']:.2f}      {metrics['recall']:.2f}      {metrics['f1-score']:.2f}      {int(metrics['support'])}\n"
        
    # Calculate global averages for accuracy
    macro_acc = np.mean(list(accuracies.values()))
    weighted_acc = sum(accuracies[c] * report_dict[c]['support'] for c in class_names) / total_samples_cm

    report_str += "\n"
    for avg_type in ['macro avg', 'weighted avg']:
        metrics = report_dict[avg_type]
        avg_acc = macro_acc if avg_type == 'macro avg' else weighted_acc
        report_str += f"{avg_type:>16}       {avg_acc:.2f}       {metrics['precision']:.2f}      {metrics['recall']:.2f}      {metrics['f1-score']:.2f}      {int(metrics['support'])}\n"
    
    print(f"\n>>> Final Evaluation Metrics on Validation Set ({val_vids[0]}):")
    print(report_str)
    
    summary_text += f"\n\n=== Final Evaluation Metrics on Validation Set ({val_vids[0]}) ===\n"
    summary_text += report_str

    with open(os.path.join(exp_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    print(f"\n>>> FINISHED. Visual results in: {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn_type", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()
    train_visual_model(args)

