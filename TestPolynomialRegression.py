import numpy as np
from KAN import KAN

np.random.seed(42)

N = 1000

X = np.random.uniform(-2, 2, (N, 3))

y = (2 * X[:, 0] ** 3 - 3 * X[:, 1] ** 2 + 5 * X[:, 2] 
     + X[:, 0] * X[:, 1] - 0.5 * X[:, 1] * X[:, 2] + 1.5).reshape(-1, 1)

split = int(0.8 * N)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

kan = KAN(
    layer_dims=[3, 30, 1],
    polynomial_degree=3,
    grid_size=5,
    learning_rate=0.001
)

kan.train(
    X_train,
    y_train,
    epochs=1000,
    task="regression",
    batch_size=64
)

y_pred = kan.forward(X_test)

mse  = np.mean((y_pred - y_test) ** 2)
rmse = np.sqrt(mse)
mae  = np.mean(np.abs(y_pred - y_test))
r2   = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

print("\nPolynomial Regression Results")
print("Target : f(x1,x2,x3) = 2x1³ - 3x2² + 5x3 + x1·x2 - 0.5x2·x3 + 1.5")
print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)
print("R²  :", r2)
