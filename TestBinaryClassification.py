import numpy as np
from KAN import KAN

np.random.seed(42)

# Class 0
X0 = np.random.randn(500, 2) + np.array([-2, -2])

# Class 1
X1 = np.random.randn(500, 2) + np.array([2, 2])

X = np.vstack([X0, X1])
y = np.vstack([
    np.zeros((500, 1)),
    np.ones((500, 1))
])

# Shuffle
idx = np.random.permutation(len(X))
X = X[idx]
y = y[idx]

# Split
split = int(0.8 * len(X))

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

# KAN
kan = KAN(
    layer_dims=[2, 10, 1],
    polynomial_degree=3,
    grid_size=5,
    learning_rate=0.001
)

# Train
kan.train(
    X_train,
    y_train,
    epochs=500,
    task="binary_classification"
)

# Predict
logits = kan.forward(X_test)

# Convert to probabilities
probs = 1 / (1 + np.exp(-logits))

preds = (probs > 0.5).astype(int)

accuracy = np.mean(preds == y_test)

print("\nClassification Results")
print("Accuracy:", accuracy)