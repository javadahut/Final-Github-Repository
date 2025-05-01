import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import ASTModel
from sklearn.metrics import classification_report
import argparse
import matplotlib.pyplot as plt

class SpectrogramDataset(Dataset):
    def __init__(self, data_path):
        pos = torch.load(os.path.join(data_path, 'tTrainingDataPos_AST.pt'))
        neg = torch.load(os.path.join(data_path, 'tTrainingDataNeg_AST.pt'))

        self.data = torch.cat([pos, neg], dim=0)
        self.labels = torch.cat([
            torch.ones(pos.size(0), dtype=torch.long),
            torch.zeros(neg.size(0), dtype=torch.long)
        ])

        print(f"[Dataset Init] Data: {self.data.shape}, Labels: {self.labels.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # should be [3,224,224]
        y = self.labels[idx]
        print(f"[Dataset] idx: {idx}, x: {x.shape}, y: {y}")
        return x, y

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

class ASTClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(ASTClassifier, self).__init__()
        self.ast_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        hidden_size = self.ast_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        outputs = self.ast_model(x)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SpectrogramDataset(args.dataDir)

    total = len(dataset)
    val_size = int(0.1 * total)
    test_size = int(0.1 * total)
    train_size = total - val_size - test_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model = ASTClassifier(num_labels=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            print(f"[Train Batch] x: {x.shape}, y: {y.shape}")
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                print(f"[Val Batch] x: {x.shape}, y: {y.shape}")
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = loss_fn(outputs, y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    plot_loss(train_losses, val_losses)

    # Testing
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            print(f"[Test Batch] x: {x.shape}, y: {y.shape}")
            x = x.to(device)
            outputs = model(x)
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            true.extend(y.numpy())

    print("Classification Report on Test Set:")
    print(classification_report(true, preds, digits=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataDir', type=str, required=True)
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-lr', type=float, default=5e-5)
    args = parser.parse_args()
    main(args)
