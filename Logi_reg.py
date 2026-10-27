import numpy as np
import matplotlib.pyplot as plt

X = np.load("exam.npy")
Y = np.load("passed.npy")
w, b, lr = np.zeros(2), 0, 0.01


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


losses = []
for epoch in range(5000):
    p = sigmoid(X @ w + b)
    loss = -(Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean()
    dw = (X.T @ (p - Y)) / len(X)
    db = (p - Y).mean()
    w -= lr * dw
    b -= lr * db
    losses.append(loss)

acc = ((sigmoid(X @ w + b) > 0.5) == Y).mean()
print(sigmoid(np.array([6, 7]) @ w + b))
print(f"accuracy {acc:.1%}  w={w.round(3)}  b={b:.3f}")

x1 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300)
x2 = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300)
xx, yy = np.meshgrid(x1, x2)
zz = sigmoid(xx * w[0] + yy * w[1] + b)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cs = axes[0].contourf(xx, yy, zz, levels=50, cmap="RdBu_r", alpha=0.75)
axes[0].contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.5)
axes[0].scatter(X[Y == 0, 0], X[Y == 0, 1], c="tab:blue", s=25, label="failed (0)",
                edgecolors="k", linewidths=0.3)
axes[0].scatter(X[Y == 1, 0], X[Y == 1, 1], c="tab:red", s=25, label="passed (1)",
                edgecolors="k", linewidths=0.3)
axes[0].scatter(6, 7, marker="*", c="lime", s=350, edgecolors="k",
                linewidths=1.0, zorder=5, label="query (6, 7)")
axes[0].set_title(f"Logistic regression decision boundary (acc {acc:.1%})")
axes[0].set_xlabel("exam score 1")
axes[0].set_ylabel("exam score 2")
axes[0].legend(loc="upper left", fontsize=9)
fig.colorbar(cs, ax=axes[0], label="P(pass)")

axes[1].plot(losses, color="tab:green")
axes[1].set_title("Training loss")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("binary cross-entropy")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("logi_reg.png", dpi=150)
print("saved logi_reg.png")
plt.show()
