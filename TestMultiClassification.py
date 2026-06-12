import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from KAN import KAN

np.random.seed(42)

digits = load_digits()
X_raw = digits.data.astype(float)
y_raw = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca = PCA(n_components=20, random_state=42)
X = pca.fit_transform(X_scaled)

n_classes = 10
y_onehot = np.zeros((len(y_raw), n_classes))
y_onehot[np.arange(len(y_raw)), y_raw] = 1.0

idx = np.random.permutation(len(X))
X = X[idx]
y_onehot = y_onehot[idx]
y_labels = y_raw[idx]

split = int(0.8 * len(X))

X_train = X[:split]
y_train = y_onehot[:split]

X_test = X[split:]
y_test_labels = y_labels[split:]

kan = KAN(
    layer_dims=[20, 32, 10],
    polynomial_degree=3,
    grid_size=5,
    learning_rate=0.005
)

kan.train(
    X_train,
    y_train,
    epochs=500,
    task="multiclass_classification",
    batch_size=64
)

logits = kan.forward(X_test)
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
preds = np.argmax(probs, axis=1)

accuracy = np.mean(preds == y_test_labels)

print("\nMulti-Class Classification Results (Digits Dataset)")
print("Classes     :", n_classes)
print("Test samples:", len(X_test))
print("PCA features: 20 (reduced from 64)")
print("Accuracy    :", accuracy)
