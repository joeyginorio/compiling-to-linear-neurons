import os
import torch
import random
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import language.cajal as cj
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from mnist_cnn.mnist import CNN
from torcheval.metrics.functional import peak_signal_noise_ratio as psnr
import torch.nn.functional as F

# ---------- Device ------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ---------- Data --------------------
train_xs = torch.load("data/train_xs.pt")
train_ys = torch.load("data/train_ys.pt")
train_check_ys = torch.load("data/train_check_ys.pt")

test_xs = torch.load("data/test_xs.pt")
test_ys = torch.load("data/test_ys.pt")
test_check_ys = torch.load("data/test_check_ys.pt")

train_ds = TensorDataset(train_xs, train_ys, train_check_ys)
test_ds = TensorDataset(test_xs, test_ys, test_check_ys)

# ---------- Model -------------------
class ModelD(nn.Module):
    def __init__(self):
        super().__init__()

        # R(784) -> R(2)
        # 80k params
        self.check = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )

        # R(784) -> R(784)
        # 600k params (no need to learn)
        layer = nn.Linear(784, 784, bias=True)
        nn.init.eye_(layer.weight)
        nn.init.zeros_(layer.bias)
        layer.weight.requires_grad = False
        layer.bias.requires_grad = False
        self.if_branch = nn.Sequential(
            nn.Flatten(),
            layer,
            nn.Unflatten(1, (1, 28, 28)),
        )

        # R(784) -> R(200)
        # 300k params
        self.then_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 28 * 28),
            nn.Unflatten(1, (1, 28, 28)),
        )

    def forward(self, x):
        b = self.check(x)
        if_val = self.if_branch(x).view(-1, 1, 28 * 28)
        then_val = self.then_branch(x).view(-1, 1, 28 * 28)
        p = cj.TmIf(cj.TmVar('bv'), cj.TmVar('ifv'), cj.TmVar('thenv'))
        env = {'bv' : b, 'ifv' : if_val, 'thenv' : then_val}
        return cj.compile(p, env, x.size(0)).view(-1,1,28,28)


# ---------- Training ----------------
seeds = [0]
batch_sizes = [32]
learning_rates = [.001]
idxs = [0, 3, 5, 200, 603, 8, 14, 23, 24, 40, 59, 67, 70, 72, 77, 78, 99, 102]
lams = [0.0, 1.2]

# CNN measurements
test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)
cnn = CNN().to(device)
cnn_path = "mnist_cnn/data/mnist_cnn.pt"
cnn.load_state_dict(torch.load(cnn_path, map_location=device))
cnn.eval()

# PSNR measurements
ones_test = torch.load("data/ones_test.pt")
one_loader = DataLoader(ones_test, batch_size=6742, shuffle=False)
batch_psnr = torch.vmap(psnr)

loss_train = {}
loss_test = {}
one_test = {}
output_test = {}
psnr_test = {}

models = []

for seed in seeds:
    for batch_size in batch_sizes:
        for lr in learning_rates:   
            for lam in lams:
                # Initialize random seed
                torch.manual_seed(seed)
                random.seed(seed)

                # Initialize data
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

                # Initialize model 
                modelD = ModelD().to(device)
                models += [modelD]
                criterion = nn.L1Loss()
                optimizer = optim.Adam(modelD.parameters(), lr=lr)

                # Calculate initial measurements
                test_loss = 0.0
                test_one = 0
                with torch.no_grad():
                    for data in one_loader:
                        data = data.to(device)
                        output = modelD(data)
                        bpsnr = batch_psnr(output, data).mean().item()
                    for data, target, check in test_loader:
                        data, target = data.to(device), target.to(device)

                        output = modelD(data)
                        test_loss += criterion(output, target).item()

                        isone = cnn(output).softmax(1)
                        isone = isone[:,1].mean().item()
                        test_one += isone

                test_loss /= len(test_loader)
                test_one /= len(test_loader)
                loss_test[(0, seed, batch_size, lr, lam)] = test_loss
                one_test[(0, seed, batch_size, lr, lam)] = test_one
                psnr_test[(0, seed, batch_size, lr, lam)] = bpsnr

                for idx in idxs:
                    output = modelD(test_ds[idx][0].to(device)).squeeze().tolist()
                    output_test[(0, seed, batch_size, lr, idx, lam)] = output
            
                step = 1
                freq = 1000
                for epoch in range(40):
                    print(f"Epoch: {epoch}, Lr: {lr}, Bs: {batch_size}, Seed: {seed}")

                    for data, target, check in train_loader:
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = modelD(data)

                        # intermediate supervision
                        check_logits = modelD.check(data)
                        check = check.squeeze(1).to(device)
                        check_loss = F.cross_entropy(check_logits, check)

                        loss = criterion(output, target) + lam * check_loss
                        loss.backward()
                        optimizer.step()

                        # Record training information
                        loss_train[(step, seed, batch_size, lr, lam)] = loss.item()
                        step += 1

                        if step % freq == 0:
                            print(f"Step: {step}, Lr: {lr}, Bs: {batch_size}, Seed: {seed}")

                            test_loss = 0.0
                            test_one = 0

                            with torch.no_grad():
                                for data in one_loader:
                                    data = data.to(device)
                                    output = modelD(data)
                                    bpsnr = batch_psnr(output, data).mean().item()
                                for data, target, check in test_loader:
                                    data, target = data.to(device), target.to(device)

                                    output = modelD(data)
                                    test_loss += criterion(output, target).item()

                                    isone = cnn(output).softmax(1)
                                    isone = isone[:,1].mean().item()
                                    test_one += isone

                            # Record losses and one accuracy
                            test_loss /= len(test_loader)
                            test_one /= len(test_loader)
                            loss_test[(step, seed, batch_size, lr, lam)] = test_loss
                            one_test[(step, seed, batch_size, lr, lam)] = test_one
                            psnr_test[(step, seed, batch_size, lr, lam)] = bpsnr

                            # Record model outputs
                            for idx in idxs:
                                output = modelD(test_ds[idx][0].to(device)).squeeze().tolist()
                                output_test[(step, seed, batch_size, lr, idx, lam)] = output



with open("data/direct_debug_loss_train.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "seed", "batch size", "lr", "lam", "loss"])
    for (step, seed, batch_size, lr, lam), loss in loss_train.items():
        writer.writerow([step, seed, batch_size, lr, lam, loss])
with open("data/direct_debug_loss_test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "seed", "batch size", "lr", "lam", "loss"])
    for (step, seed, batch_size, lr, lam), loss in loss_test.items():
        writer.writerow([step, seed, batch_size, lr, lam, loss])
with open("data/direct_debug_one_test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "seed", "batch size", "lr", "lam", "one"])
    for (step, seed, batch_size, lr, lam), isone in one_test.items():
        writer.writerow([step, seed, batch_size, lr, lam, isone])
with open("data/direct_debug_output_test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "seed", "batch size", "lr", "idx", "lam", "output"])
    for (step, seed, batch_size, lr, idx, lam), output in output_test.items():
        writer.writerow([step, seed, batch_size, lr, idx, lam, output])
with open("data/direct_debug_psnr_test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "seed", "batch size", "lr", "idx", "lam", "psnr"])
    for (step, seed, batch_size, lr, lam), snr in psnr_test.items():
        writer.writerow([step, seed, batch_size, lr, idx, lam, snr])

modelD = modelD.to("cpu")

idxs = [0, 3, 5, 200, 603, 99, 102]

fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(6, 8))
for row, idx in enumerate(idxs):
    # prepare input
    x = test_ds[idx][0]
    xim = x.squeeze().detach()
    with torch.no_grad():
        y = modelD(x)
    yim = y.squeeze().detach()

    # plot input on the left, output on the right
    ax_in  = axes[row, 0]
    ax_out = axes[row, 1]

    ax_in.imshow(xim,  cmap="gray", vmin=0, vmax=1)
    ax_in.set_title(f"Input #{idx}")
    ax_in.axis("off")

    ax_out.imshow(yim, cmap="gray", vmin=0, vmax=1)
    ax_out.set_title(f"Output #{idx}")
    ax_out.axis("off")

plt.tight_layout()
plt.show()

