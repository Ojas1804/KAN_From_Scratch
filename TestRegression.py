import numpy as np
from KAN import KAN

# Generate dataset
np.random.seed(42)

X = np.linspace(-3, 3, 500).reshape(-1, 1)
y = np.sin(X)

# Train/Test split
split = int(0.8 * len(X))

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

# Create KAN
kan = KAN(
    layer_dims=[1, 10, 1],
    polynomial_degree=3,
    grid_size=5,
    learning_rate=0.001
)

# Train
kan.train(
    X_train,
    y_train,
    epochs=500,
    task="regression"
)

# Predict
y_pred = kan.forward(X_test)

# Metrics
mse = np.mean((y_pred - y_test) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_pred - y_test))

print("\nRegression Results")
print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)