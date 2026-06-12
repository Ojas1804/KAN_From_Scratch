import numpy as np
import itertools

class KnotGrid:
    def __init__(self, polynomial_degree, grid_size):
        self.polynomial_degree = polynomial_degree
        self.grid_size = grid_size
        self.num_basis_func = polynomial_degree + grid_size - 1
        self.knots = None

    def fit(self, inputs):
        x_min = float(np.min(inputs))
        x_max = float(np.max(inputs))
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0
        self._build_knots(x_min, x_max)

    def _build_knots(self, x_min, x_max):
        grid_points = np.linspace(x_min, x_max, num=self.grid_size).tolist()
        self.knots = list(itertools.chain(
            [x_min] * self.polynomial_degree,
            grid_points,
            [x_max] * self.polynomial_degree
        ))

    def transform(self, inputs):
        self.knot_grid = np.array([self.__cox_deboor(x) for x in inputs])
        self.knot_grid_derivative = np.array([
            self.cox_deboor_derivative(x, self.polynomial_degree)
            for x in inputs
        ])

    # knots = (delta)/G
    def create_knot_grid(self, inputs):
        self.fit(inputs)
        self.transform(inputs)
    
    def __cox_deboor(self, x, degree=None):
        if degree is None:
            degree = self.polynomial_degree

        n = len(self.knots) - 1  # number of intervals
        
        # Base case: degree 0
        B = np.zeros(n)
        for i in range(n):
            if self.knots[i] <= x < self.knots[i + 1]:
                B[i] = 1.0

        if x == self.knots[-1]:
            B[-1] = 1.0

        # Recursive case:
        for d in range(1, degree + 1):
            B_prev = B.copy()
            B = np.zeros(n - d)
            for i in range(n - d):
                denom_left = self.knots[i + d] - self.knots[i]
                left = ((x - self.knots[i]) / denom_left * B_prev[i]
                        if denom_left != 0 else 0.0)
                denom_right = self.knots[i + d + 1] - self.knots[i + 1]
                right = ((self.knots[i + d + 1] - x) / denom_right * B_prev[i + 1]
                        if denom_right != 0 else 0.0)
                B[i] = left + right
        return B

    def cox_deboor_derivative(self, x, degree):
        if degree <= 0:
            return np.zeros(len(self.knots) - 1)

        lower_basis = self.__cox_deboor(x, degree - 1)
        num_basis = len(self.knots) - degree - 1
        derivative = np.zeros(num_basis)

        for i in range(num_basis):
            denom_left = self.knots[i + degree] - self.knots[i]
            left = (degree / denom_left) * lower_basis[i] if denom_left != 0 else 0.0

            denom_right = self.knots[i + degree + 1] - self.knots[i + 1]
            right = ((degree / denom_right) * lower_basis[i + 1]
                     if denom_right != 0 else 0.0)

            derivative[i] = left - right

        return derivative


if __name__ == '__main__':
    k = KnotGrid(3, 5)
    k.create_knot_grid([1, 2, 3, 4, 5, 6, 7])
    print(k.knot_grid)
