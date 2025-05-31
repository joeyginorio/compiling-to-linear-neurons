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
epochs       = 6
lr           = 1e-3
model_path   = "data/mnist_cnn.pt"
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

train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST("data", train=True, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# ---------- Model -------------------
class CNN(nn.Module): 
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(128*7*7, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


model = CNN().to(device)
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