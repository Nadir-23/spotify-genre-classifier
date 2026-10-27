import numpy as np

rng = np.random.default_rng(0)
n = 200
exam = rng.uniform(0, 10, size=(n, 2))
logit = 1.1 * exam[:, 0] + 0.9 * exam[:, 1] - 11.0
p = 1 / (1 + np.exp(-logit))
passed = (rng.uniform(size=n) < p).astype(float)

np.save("exam.npy", exam)
np.save("passed.npy", passed)
print("exam.npy", exam.shape, "passed.npy", passed.shape, "positives", int(passed.sum()))
