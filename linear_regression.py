
import numpy as np


class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, iterations=5000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None
        self.cost_history = []
        self.cost_iterations = []

    def hypothesis(self, X):
        """Calculate predictions using matrix multiplication."""
        return X @ self.theta

    def compute_cost(self, X, y):
        """Calculate the cost function J(theta) = sum(error^2)/(2m)."""
        m = len(y)
        errors = self.hypothesis(X) - y
        return np.sum(errors ** 2) / (2 * m)

    def fit(self, X, y):
        """Train the model using batch gradient descent."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        m, n = X.shape

        if len(y) != m:
            raise ValueError("X and y must have the same number of rows.")

        self.theta = np.zeros(n)
        self.cost_history = []
        self.cost_iterations = []

        for i in range(self.iterations):
            predictions = self.hypothesis(X)
            errors = predictions - y

            # Calculate gradient
            gradient = (X.T @ errors) / m

            # Update parameters
            self.theta -= self.learning_rate * gradient

            # Record cost every 10 iterations and at the end
            if (i + 1) % 10 == 0 or i == self.iterations - 1:
                cost = self.compute_cost(X, y)
                self.cost_history.append(cost)
                self.cost_iterations.append(i + 1)

        return self

    def predict(self, X):
        """Predict target values."""
        if self.theta is None:
            raise ValueError("Train the model before predicting.")

        X = np.asarray(X, dtype=float)
        return self.hypothesis(X)
