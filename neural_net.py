import numpy as np
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
X = X / 16.0
Y = np.eye(10)[y]
np.random.seed(0)
lr = 0.5
W1, b1 = np.random.randn(64, 32) * 0.1, np.zeros(32)
W2, b2 = np.random.randn(32, 10) * 0.1, np.zeros(10)

for epoch in range(300):
    h = np.maximum(0, X @ W1 + b1)
    e = np.exp(h @ W2 + b2)
    p = e / e.sum(axis=1, keepdims=True)

    d2 = (p - Y) / len(X)
    d1 = (d2 @ W2.T) * (h > 0)

    W2 -= lr * (h.T @ d2)
    b2 -= lr * d2.sum(axis=0)
    W1 -= lr * (X.T @ d1)
    b1 -= lr * d1.sum(axis=0)

print((p.argmax(1) == y).mean())
