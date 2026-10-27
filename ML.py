import torch, torch.nn as nn
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

X, y = make_moons(500, noise=0.2, random_state=0)
X, y = torch.tensor(X).float(), torch.tensor(y).float()
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(),
                      nn.Linear(16, 16), nn.ReLU(),
                      nn.Linear(16, 1))
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
for epoch in range(500):
    loss = loss_fn(model(X).squeeze(1), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())

acc = ((model(X).squeeze(1) > 0) == y).float().mean()
print(f"accuracy {acc:.1%}", loss.item())

model.eval()
with torch.no_grad():
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float()
    zz = torch.sigmoid(model(grid)).reshape(xx.shape).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cs = axes[0].contourf(xx, yy, zz, levels=50, cmap="RdBu_r", alpha=0.75)
    axes[0].contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.5)
    axes[0].scatter(X[y == 0, 0], X[y == 0, 1], c="tab:blue", s=18, label="class 0", edgecolors="k", linewidths=0.3)
    axes[0].scatter(X[y == 1, 0], X[y == 1, 1], c="tab:red", s=18, label="class 1", edgecolors="k", linewidths=0.3)
    axes[0].set_title(f"Decision boundary (acc {acc:.1%})")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].legend(loc="upper right")
    fig.colorbar(cs, ax=axes[0], label="P(class 1)")

    axes[1].plot(losses, color="tab:green")
    axes[1].set_title("Training loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("BCE loss")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("moons.png", dpi=150)
    print("saved moons.png")

plt.show()
