# Merged AST Model Code
# Includes: working data pipeline, training loop, evaluation, visualizations

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt

# === Configuration ===
class Args:
    dataDir = "D:\Documents\School\SP25-DS677852 Deep Learning\Final Github Repository\Final-Github-Repository\data"
    epochs = 10
    batch_size = 8
    lr = 5e-5
args = Args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Dataset ===
class SpectrogramDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.df = pd.read_csv(labels_path)
        self.data = [os.path.join(data_path, fname) for fname in self.df["clip_name"]]
        self.labels = torch.tensor(self.df["label"].values).long()
        self.feature_extractor = ASTFeatureExtractor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        inputs = self.feature_extractor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")
        return inputs["input_values"].squeeze(0), self.labels[idx]

# === Data Split ===
def split_dataset(dataset, val_pct=0.1, test_pct=0.1):
    total_size = len(dataset)
    val_size = int(val_pct * total_size)
    test_size = int(test_pct * total_size)
    train_size = total_size - val_size - test_size
    return random_split(dataset, [train_size, val_size, test_size])

# === Metrics ===
def extractMetrics(preds, labels):
    preds = preds.ravel()
    labels = labels.ravel()
    precision, recall, _ = precision_recall_curve(labels, preds)
    ap = average_precision_score(labels, preds)
    fpr, tpr, _ = roc_curve(labels, preds)
    aucroc = roc_auc_score(labels, preds)
    return precision, recall, ap, fpr, tpr, aucroc

# === Train / Eval ===
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x).logits
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader.dataset), all_labels, all_preds

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x).logits
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
    return classification_report(all_labels, all_preds, digits=4, zero_division=0), np.array(all_labels), np.array(all_preds)

# === Plotting ===
def plot_loss(train_losses, val_losses):
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("loss_curve.png")
    plt.show()

# === Main Execution ===
def main(args):
    dataset = SpectrogramDataset(
        data_path=os.path.join(args.dataDir, "train"),
        labels_path=os.path.join(args.dataDir, "train.csv")
    )
    train_set, val_set, test_set = split_dataset(dataset)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model.classifier = nn.Linear(model.classifier.dense.in_features, 2)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(args.epochs):
        train_loss, train_labels, train_preds = train(model, train_loader, optimizer, criterion, device)
        val_report, val_labels, val_preds = evaluate(model, val_loader, device)
        val_loss = criterion(model(torch.stack([x.to(device) for x, _ in val_set]).squeeze(1)).logits,
                             torch.tensor([y for _, y in val_set]).to(device)).item()

        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc * 100)
        val_accuracies.append(val_acc * 100)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.2f}, Val Acc={val_acc:.2f}")

    test_report, test_labels, test_preds = evaluate(model, test_loader, device)
    print("\nTest Classification Report:\n", test_report)

    plot_loss(train_losses, val_losses)

if __name__ == '__main__':
    main(args)
