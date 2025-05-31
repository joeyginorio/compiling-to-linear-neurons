import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

"""
Generates the dataset for conditional '1' task
"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST("data", train=True, download=True, transform=transform)

# Collect all the images of 1s
xs_ones_train = []
for x, label in train_ds:
    if label == 1:
        xs_ones_train += [x]

xs_ones_test = []
for x, label in test_ds:
    if label == 1:
        xs_ones_test += [x]

avg_one = torch.stack(xs_ones_test).mean(0).squeeze()

xs = []
ys = []
check_ys = []
for x, label in train_ds:
    if label == 1:
        xs += [x]
        ys += [x]
        check_ys += [1]
    else:
        xs += [x]
        ys += [random.choice(xs_ones_train)]
        check_ys += [0]

xs = torch.cat(xs)
ys = torch.cat(ys)
check_ys = torch.tensor(check_ys)

xs = xs.reshape(60000, 1, 28, 28)
ys = ys.reshape(60000, 1 ,28, 28)
check_ys = check_ys.reshape(60000, 1)


torch.save(xs, "data/train_xs.pt")
torch.save(ys, "data/train_ys.pt")
torch.save(check_ys, "data/train_check_ys.pt")
torch.save(avg_one, "data/avg_one.pt")
torch.save(torch.stack(xs_ones_test), "data/ones_test.pt")

xs = []
ys = []
check_ys = []
for x, label in test_ds:
    if label == 1:
        xs += [x]
        ys += [x]
        check_ys += [1]
    else:
        xs += [x]
        ys += [random.choice(xs_ones_test)]
        check_ys += [0]

xs = torch.cat(xs)
ys = torch.cat(ys)
check_ys = torch.tensor(check_ys)

xs = xs.reshape(60000, 1, 28, 28)
ys = ys.reshape(60000, 1 ,28, 28)
check_ys = check_ys.reshape(60000,1)

torch.save(xs, "data/test_xs.pt")
torch.save(ys, "data/test_ys.pt")
torch.save(check_ys, "data/test_check_ys.pt")

