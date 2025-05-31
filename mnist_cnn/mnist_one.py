import torch
import random
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# ---------- Hyper-Parameters ----------
batch_size   = 128
epochs       = 8
lr           = 1e-3
model_path   = "data/mnist_cnn_debug.pt"
seed         = 42

torch.manual_seed(seed)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ---------- Data --------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# train set: targets → 1 if label==1 else 0
train_ds = datasets.MNIST(
    "data",
    train=True,
    download=True,
    transform=transform,
    target_transform=lambda y: 1 if y == 1 else 0
)

# test set: same idea
test_ds = datasets.MNIST(
    "data",
    train=False,
    download=True,
    transform=transform,
    target_transform=lambda y: 1 if y == 1 else 0
)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)


# --- Model
class CNN_One(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            # → [32,28,28]
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),            # → [32,14,14]

            # → [64,14,14]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),            # → [64,7,7]
        )

        # Classifier head
        # flatten size = 64 * 7 * 7 = 3136
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        """
        x: tensor of shape (N,1,28,28)
        returns: tensor of shape (N,2)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CNN_One().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ---------- Training ----------------
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}  [{batch_idx*len(data):>5}/{len(train_loader.dataset)}]  "
                  f"loss: {loss.item():.4f}")
            

def test():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total   += target.size(0)
    acc = 100. * correct / total
    print(f"\nTest accuracy: {acc:.2f}% ({correct}/{total})")
    return acc


def train_save():
    for ep in range(1, epochs+1):
        train(ep)
        acc = test()

    torch.save(model.state_dict(), model_path)

# train_save()