# https://chatgpt.com/s/t_6a24163b780c81919d833c13b3739d8f
import numpy as np
from Layer import Layer
from KnotGrid import KnotGrid

class KAN:
    def __init__(self, layer_dims, polynomial_degree=3, 
                 grid_size=5, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.polynomial_degree = polynomial_degree
        self.grid_size = grid_size
        self.layers = []
        self.knot_grids = []

        # Build one Layer + one KnotGrid per consecutive pair of dims
        for i in range(len(layer_dims) - 1):
            n_in = layer_dims[i]
            n_out = layer_dims[i + 1]
            layer = Layer(n_in, n_out, polynomial_degree, grid_size)
            self.layers.append(layer)
            self.knot_grids.append(KnotGrid(polynomial_degree, grid_size))
    
    def __sigmoid(self, x):
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x))
        )

    def __silu(self, x):
        return x * self.__sigmoid(x)

    def __silu_derivative(self, x):
        sig = self.__sigmoid(x)
        return sig + x * sig * (1.0 - sig)

    def __loss_function__(self, y_pred, y_true, task="regression"):
        N = y_true.shape[0]
        if task == "regression":
            diff = y_pred - y_true
            loss = np.mean(diff ** 2)
            grad = (2.0 / N) * diff
            return loss, grad
        elif task == "binary_classification":
            logits = y_pred
            loss = np.mean(np.maximum(logits, 0) - logits * y_true + np.log1p(np.exp(-np.abs(logits))))
            grad = (self.__sigmoid(logits) - y_true) / N
            return loss, grad
    
    def forward(self, X):
        out = X
        for layer, knot_grid in zip(self.layers, self.knot_grids):
            batch_size = out.shape[0]
            n_in = layer.n_in
            n_out = layer.n_out
            num_basis = layer.coefficients.shape[2]
            layer.input = out
            basis_matrices = []
            basis_derivative_matrices = []
            for i in range(n_in):
                col = out[:, i]
                knot_grid.create_knot_grid(col)
                bm = knot_grid.knot_grid
                bm_deriv = knot_grid.knot_grid_derivative
                basis_matrices.append(bm)
                basis_derivative_matrices.append(bm_deriv)
            layer.basis_matrices = np.array(basis_matrices)
            layer.basis_derivative_matrices = np.array(basis_derivative_matrices)
            #   phi_{i,j}(x_i) = w_{i,j} * silu(x_i) + B(x_i) @ c_{i,j}
            layer_out = np.zeros((batch_size, n_out))
            for i in range(n_in):
                x_col = out[:, i]
                bm_i = layer.basis_matrices[i]
                for j in range(n_out):
                    spline_val = bm_i @ layer.coefficients[i,j]
                    residual_val = layer.w_residual[i, j] * self.__silu(x_col)
                    layer_out[:,j] += spline_val + residual_val

            layer.output = layer_out
            out = layer_out

        return out

    def backpropagation(self, grad_output):
        grad = grad_output  # will be updated as we move backward

        for layer in reversed(self.layers):
            # batch_size = layer.input.shape[0]
            n_in = layer.n_in
            n_out = layer.n_out
            grad_input = np.zeros_like(layer.input)
            layer.grad_coefficients = np.zeros_like(layer.coefficients)
            layer.grad_w_residual = np.zeros_like(layer.w_residual)
            for i in range(n_in):
                x_col = layer.input[:, i]
                bm_i = layer.basis_matrices[i]
                bm_deriv_i = layer.basis_derivative_matrices[i]
                for j in range(n_out):
                    g = grad[:, j]
                    # out_j += bm_i @ c[i,j]  =>  dc[i,j] = bm_i.T @ g
                    layer.grad_coefficients[i, j] += bm_i.T @ g   # (num_basis,)
                    # out_j += w[i,j] * silu(x_i)  =>  dw = silu(x_i) · g
                    layer.grad_w_residual[i, j] += np.dot(self.__silu(x_col), g)
                    spline_grad = bm_deriv_i @ layer.coefficients[i, j]
                    residual_grad = layer.w_residual[i, j] * self.__silu_derivative(x_col)
                    grad_input[:, i] += (spline_grad + residual_grad) * g

            grad = grad_input
        return grad

    def __update_parameters(self):
        for layer in self.layers:
            layer.coefficients -= self.learning_rate * layer.grad_coefficients
            layer.w_residual -= self.learning_rate * layer.grad_w_residual
    
    def train(self, X, y, epochs=100, task="regression"):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss, grad_loss = self.__loss_function__(y_pred, y, task=task)
            self.backpropagation(grad_loss)
            self.__update_parameters()
            if epoch % 10 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")
