import numpy as np
from Layer import Layer

class KAN:
    def __init__(self, layer_dims, polynomial_degree=3, 
                 grid_size=5, learning_rate=0.01,
                 beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.polynomial_degree = polynomial_degree
        self.grid_size = grid_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.layers = []

        for i in range(len(layer_dims) - 1):
            n_in = layer_dims[i]
            n_out = layer_dims[i + 1]
            layer = Layer(n_in, n_out, polynomial_degree, grid_size)
            self.layers.append(layer)

        self.adam_m_coeff = [np.zeros_like(l.coefficients) for l in self.layers]
        self.adam_v_coeff = [np.zeros_like(l.coefficients) for l in self.layers]
        self.adam_m_res   = [np.zeros_like(l.w_residual)   for l in self.layers]
        self.adam_v_res   = [np.zeros_like(l.w_residual)   for l in self.layers]
    
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

    def __softmax(self, x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

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
        elif task == "multiclass_classification":
            probs = self.__softmax(y_pred)
            probs_clipped = np.clip(probs, 1e-15, 1.0)
            loss = -np.mean(np.sum(y_true * np.log(probs_clipped), axis=1))
            grad = (probs - y_true) / N
            return loss, grad
    
    def _fit_knots(self, X):
        out = X
        for layer in self.layers:
            for i in range(layer.n_in):
                layer.knot_grids[i].fit(out[:, i])
            batch_size = out.shape[0]
            layer_out = np.zeros((batch_size, layer.n_out))
            for i in range(layer.n_in):
                col = out[:, i]
                layer.knot_grids[i].transform(col)
                bm_i = layer.knot_grids[i].knot_grid
                for j in range(layer.n_out):
                    layer_out[:, j] += bm_i @ layer.coefficients[i, j]
                    layer_out[:, j] += layer.w_residual[i, j] * self.__silu(col)
            out = layer_out

    def forward(self, X):
        out = X
        for layer in self.layers:
            batch_size = out.shape[0]
            n_in = layer.n_in
            n_out = layer.n_out
            layer.input = out
            basis_matrices = []
            basis_derivative_matrices = []
            for i in range(n_in):
                col = out[:, i]
                kg = layer.knot_grids[i]
                if kg.knots is None:
                    kg.fit(col)
                kg.transform(col)
                basis_matrices.append(kg.knot_grid)
                basis_derivative_matrices.append(kg.knot_grid_derivative)
            layer.basis_matrices = np.array(basis_matrices)
            layer.basis_derivative_matrices = np.array(basis_derivative_matrices)
            #   phi_{i,j}(x_i) = w_{i,j} * silu(x_i) + B(x_i) @ c_{i,j}
            layer_out = np.zeros((batch_size, n_out))
            for i in range(n_in):
                x_col = out[:, i]
                bm_i = layer.basis_matrices[i]
                for j in range(n_out):
                    spline_val = bm_i @ layer.coefficients[i, j]
                    residual_val = layer.w_residual[i, j] * self.__silu(x_col)
                    layer_out[:, j] += spline_val + residual_val

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
        self.t += 1
        for idx, layer in enumerate(self.layers):
            for param, m, v, grad in [
                (layer.coefficients, self.adam_m_coeff[idx],
                 self.adam_v_coeff[idx], layer.grad_coefficients),
                (layer.w_residual,   self.adam_m_res[idx],
                 self.adam_v_res[idx], layer.grad_w_residual),
            ]:
                grad_clipped = np.clip(grad, -1.0, 1.0)
                m[:] = self.beta1 * m + (1.0 - self.beta1) * grad_clipped
                v[:] = self.beta2 * v + (1.0 - self.beta2) * (grad_clipped ** 2)
                m_hat = m / (1.0 - self.beta1 ** self.t)
                v_hat = v / (1.0 - self.beta2 ** self.t)
                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def train(self, X, y, epochs=100, task="regression", batch_size=None):
        self._fit_knots(X)
        N = X.shape[0]
        if batch_size is None:
            batch_size = N

        for epoch in range(epochs):
            indices = np.random.permutation(N)
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, N, batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                y_pred = self.forward(X_batch)
                loss, grad_loss = self.__loss_function__(y_pred, y_batch, task=task)
                self.backpropagation(grad_loss)
                self.__update_parameters()
                epoch_loss += loss
                num_batches += 1
            if epoch % 10 == 0:
                print(f"Epoch {epoch:4d} | Loss: {epoch_loss / num_batches:.6f}")
