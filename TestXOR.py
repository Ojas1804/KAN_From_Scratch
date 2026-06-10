import numpy as np
from KAN import KAN

np.random.seed(42)

N = 1000

X = np.random.uniform(-1, 1, (N, 2))

y = ((X[:, 0] * X[:, 1]) < 0).astype(int)
y = y.reshape(-1, 1)

split = int(0.8 * N)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

kan = KAN(
    layer_dims=[2, 20, 1],
    polynomial_degree=3,
    grid_size=5,
    learning_rate=0.001
)

kan.train(
    X_train,
    y_train,
    epochs=1000,
    task="binary_classification"
)

probs = 1 / (1 + np.exp(-kan.forward(X_test)))
preds = (probs > 0.5).astype(int)

accuracy = np.mean(preds == y_test)

print("XOR Accuracy:", accuracy)