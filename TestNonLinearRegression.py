import numpy as np
from KAN import KAN

np.random.seed(42)

N = 1000

X = np.random.uniform(-3, 3, (N, 2))

y = (np.sin(X[:, 0]) + X[:, 1] ** 2 - 0.5 * X[:, 0] * X[:, 1]).reshape(-1, 1)

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
    task="regression",
    batch_size=64
)

y_pred = kan.forward(X_test)

mse  = np.mean((y_pred - y_test) ** 2)
rmse = np.sqrt(mse)
mae  = np.mean(np.abs(y_pred - y_test))

print("\nNon-Linear Regression Results")
print("Target : f(x1, x2) = sin(x1) + x2^2 - 0.5*x1*x2")
print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)
