import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from tqdm import tqdm
import os
import json

config = {"log_interval": 1, "batch_size": 128, "max_epochs": 500}
map = nn.Sequential(nn.Linear(28 * 28, 128), nn.LeakyReLU(), nn.Linear(128, 10))

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
mnist_dataset = datasets.MNIST(
    "data/external/", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    mnist_dataset, batch_size=config["batch_size"], shuffle=True
)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RAdam(map.parameters())

os.makedirs(f"results", exist_ok=True)
run_num = [int(name) for name in os.listdir("results")]
run_num = max(run_num) + 1 if len(run_num) > 0 else 0
os.makedirs(f"results/{run_num}")
with open(f"results/{run_num}/config.json", "w", encoding="utf-8") as f:
    json.dump(config, f)

# Train the model
losses = []
for epoch in range(config["max_epochs"]):
    for batch_idx, (src, tgt) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        src = src.reshape(src.shape[0], -1)
        optimizer.zero_grad()
        mapped = map(src)
        loss = criterion(mapped, tgt)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if epoch % config["log_interval"] == 0:
        print(f"Epoch {epoch}, Loss {sum(losses[-(batch_idx+1):])/(batch_idx+1)}")

        # Plot and save loss graph
        plt.figure()
        plt.plot(losses, label="Training Loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve (up to epoch {epoch})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"results/{run_num}/loss_plot_epoch_{epoch}.png")
        plt.close()
