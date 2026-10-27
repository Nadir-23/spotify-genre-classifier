import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

CSV = "/home/nadir/.cache/kagglehub/datasets/maharshipandya/-spotify-tracks-dataset/versions/1/dataset.csv"
FEATURES = ["duration_ms", "explicit", "danceability", "energy", "key", "loudness",
            "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
            "valence", "tempo", "time_signature"]

df = pd.read_csv(CSV, index_col=0).dropna(subset=["artists", "album_name", "track_name"])
df = df.drop_duplicates(subset="track_id", keep="first")
X = df[FEATURES].astype(float).to_numpy()
le = LabelEncoder()
y = le.fit_transform(df["track_genre"].to_numpy())
n_classes = len(le.classes_)
print(f"{len(df)} tracks, {len(FEATURES)} features, {n_classes} genres")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=0, stratify=y)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.15, random_state=0, stratify=y_tr)
mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
X_tr, X_va, X_te = [((a - mu) / sd).astype(np.float32) for a in (X_tr, X_va, X_te)]

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(len(FEATURES), 128), nn.ReLU(), nn.Dropout(0.2),
                      nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                      nn.Linear(64, n_classes))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def loader(X, y, shuffle):
    return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                      batch_size=256, shuffle=shuffle)
train_loader, va_ds = loader(X_tr, y_tr, True), (torch.tensor(X_va), torch.tensor(y_va))

history = {"loss": [], "val_acc": []}
sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[12, 20], gamma=0.5)
for epoch in range(30):
    model.train()
    tot = 0.0
    for xb, yb in train_loader:
        loss = loss_fn(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(xb)
    model.eval()
    with torch.no_grad():
        val_acc = (model(va_ds[0]).argmax(1) == va_ds[1]).float().mean().item()
    history["loss"].append(tot / len(X_tr))
    history["val_acc"].append(val_acc)
    sched.step()
    print(f"epoch {epoch+1:2d}  loss {history['loss'][-1]:.4f}  val acc {val_acc:.1%}  lr {sched.get_last_lr()[0]:.4f}")

model.eval()
with torch.no_grad():
    logits = model(torch.tensor(X_te))
    pred = logits.argmax(1)
    acc = (pred == torch.tensor(y_te)).float().mean().item()
    top5 = logits.topk(5, dim=1).indices.eq(torch.tensor(y_te)[:, None]).any(1).float().mean().item()
cm = confusion_matrix(y_te, pred)
per_genre = cm.diagonal() / cm.sum(1)
order = np.argsort(per_genre)

print(f"\ntest accuracy {acc:.1%}   top-5 {top5:.1%}   chance {1/n_classes:.2%}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.6]})
axes[0].plot(history["loss"], label="train loss", color="tab:blue")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy", color="tab:blue")
axes[0].set_title("Training loss"); axes[0].grid(True, alpha=0.3)
ax2 = axes[0].twinx()
ax2.plot(history["val_acc"], label="val accuracy", color="tab:orange")
ax2.set_ylabel("val accuracy", color="tab:orange")
h1, l1 = axes[0].get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
axes[0].legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

colors = plt.cm.RdYlGn(per_genre[order])
axes[1].bar(range(n_classes), per_genre[order], color=colors)
axes[1].set_xticks(range(n_classes))
axes[1].set_xticklabels(le.classes_[order], rotation=90, fontsize=5)
axes[1].set_xlabel("genre (sorted by accuracy)")
axes[1].set_ylabel("accuracy")
axes[1].set_title(f"Per-genre accuracy \u2014 median {np.median(per_genre):.1%}")
axes[1].grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("spotify_curves.png", dpi=150)

fig2, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm, cmap="viridis")
ax.set_title(f"Confusion matrix \u2014 test accuracy {acc:.1%}")
ax.set_xlabel("predicted genre"); ax.set_ylabel("true genre")
ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
ax.set_xticklabels(le.classes_, rotation=90, fontsize=5)
ax.set_yticklabels(le.classes_, fontsize=5)
fig2.colorbar(im, ax=ax, label="tracks")
fig2.tight_layout()
fig2.savefig("spotify_confusion.png", dpi=150)
print("saved spotify_curves.png, spotify_confusion.png")
plt.show()
