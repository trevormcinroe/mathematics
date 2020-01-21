"""

"""

import numpy as np


class LogisticRegression:

    def __init__(self,
                 X,
                 y):
        self.X = X
        self.y = y
        self.m = len(self.y)
        self.theta = None
        self.eta = 0.001

    def theta_init(self):

    def lin_combo(self):
        """"""

        return self.X.dot(self.theta)

    def sigmoid(self, t):
        """"""

        return 1 / (1 + np.exp(-t))

    def prediction_boundary(self, p_hat):
        """"""

        return [1 if x >= 0.5 else 0 for x in p_hat]


    def log_loss_gradient(self, p_hat):
        """"""

        return self.X.dot(p_hat - self.y) / self.m

    def fit(self):

        # Calculate the linear combination
        linear_combination = self.lin_combo()

        # Sigmoid it
        p_hat = self.sigmoid(t=linear_combination)

        # Decision boundary
        pred_y = self.prediction_boundary(p_hat)

        # Gradient of logloss function
        theta_grad = self.log_loss_gradient(p_hat=p_hat)

        # Update the params
        self.theta -= self.eta * theta_grad
