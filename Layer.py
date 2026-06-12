import numpy as np
from KnotGrid import KnotGrid

class Layer:
    def __init__(self, n_in, n_out, polynomial_degree, grid_size):
        self.n_in  = n_in
        self.n_out = n_out
        num_basis = (grid_size - 1) + polynomial_degree
        self.coefficients = np.random.randn(n_in, n_out, num_basis) * 0.1
        self.w_residual = np.ones((n_in, n_out)) * 0.1
        self.grad_coefficients = np.zeros_like(self.coefficients)
        self.grad_w_residual = np.zeros_like(self.w_residual)
        self.input = None
        self.basis_matrices = None
        self.basis_derivative_matrices = None
        self.output = None
        self.knot_grids = [KnotGrid(polynomial_degree, grid_size) for _ in range(n_in)]
