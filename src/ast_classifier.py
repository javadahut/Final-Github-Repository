
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from torchvision import transforms
import argparse
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class SpectrogramDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(os.path.join(data_path, 'tTrainingDataPos'))
        self.data_neg = torch.load(os.path.join(data_path, 'tTrainingDataNeg'))
        self.data = torch.cat([self.data, self.data_neg], dim=0)
        self.labels = torch.cat([
            torch.ones(len(self.data) // 2), 
            torch.zeros(len(self.data) // 2)
        ]).long()

        # AST expects 3 channels, 224x224 input
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # [1, H, W] -> [3, H, W]
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        x = self.transform(x)
        return x, y

def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        outputs = model(x).logits
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x).logits
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
    return classification_report(all_labels, all_preds, digits=4)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = SpectrogramDataset(args.dataDir)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Load model
    model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model.classifier = nn.Linear(model.classifier.in_features, 2)  # Binary classification
    model.to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")

    print("Final Evaluation:")
    print(evaluate(model, train_loader, device))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataDir', type=str, required=True)
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-lr', type=float, default=5e-5)
    args = parser.parse_args()
    main(args)




def split_dataset(dataset, val_pct=0.1, test_pct=0.1):
    total_size = len(dataset)
    val_size = int(val_pct * total_size)
    test_size = int(test_pct * total_size)
    train_size = total_size - val_size - test_size
    return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.close()

# Replace dataset loading and training sections
dataset = WhaleDataset(data_path, transform)
train_set, val_set, test_set = split_dataset(dataset)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)
test_loader = DataLoader(test_set, batch_size=16)

train_losses = []
val_losses = []

for epoch in range(5):  # Keep as-is or increase
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).logits
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation step
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits
            loss = loss_fn(outputs, labels)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

# Save loss curve
plot_loss(train_losses, val_losses)

# Test evaluation
model.eval()
true, preds = [], []
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(device)
        outputs = model(inputs).logits
        preds.extend(outputs.argmax(dim=1).cpu().numpy())
        true.extend(labels.numpy())

print(classification_report(true, preds))
