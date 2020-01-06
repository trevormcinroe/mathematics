"""

"""

import numpy as np


class GradientDescent:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_b = None

        self._initalize()

    def _initalize(self):
        """"""
        # This creates our matrix for computation
        # It adds a column vector of 1's for the intercept
        self.x_b = np.c_[np.ones(self.x.shape), self.x]

    def batch_gd(self, eta):
        """


        Args:
            eta:

        Returns:

        """
        # Randomly initalizing a vector of theta's, one for each variable (including the
        # intercept)
        theta = np.random.rand(self.x.shape[1] + 1, 1)
        n_iterations = 1000
        m = len(self.x)

        for _ in range(n_iterations):
            gradients = 2/m * self.x_b.T.dot(self.x_b.dot(theta) - self.y)
            theta = theta - eta * gradients

        return theta

    def sgd(self):
        """"""
        pass

    def minibatch_gd(self):
        """"""
        pass


# testing it out
# theta_0 = 12
# theta_1 = 4
x = 2 * np.random.rand(100, 1)
y = 12 + 4 * x + np.random.rand(100, 1)

X_b = np.c_[np.ones((100, 1)), x]
theta_ols = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)



a = GradientDescent(x=x, y=y)

print(f'OLS: {theta_ols}')
print(f'GD: {a.batch_gd(eta=0.05)}')