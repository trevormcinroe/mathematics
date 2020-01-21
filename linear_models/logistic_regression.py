"""

"""

import numpy as np


class LogisticRegression:

    def __init__(self,
                 X,
                 y,
                 fit_intercept=True,
                 eta=0.001):
        self.X = X
        self.y = y
        self.fit_intercept = fit_intercept
        self.m = len(self.y)
        self.theta = None
        self.eta = eta

        self._theta_init()



    def _add_intercept(self, X):
        """"""
        intercept = np.ones((self.X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _theta_init(self):
        """"""

        if self.fit_intercept:
            # intercept = np.ones((self.X.shape[0], 1))
            self.X = self._add_intercept(X=self.X)
            self.theta = [np.random.rand() for _ in range(self.X.shape[1])]
        else:
            self.theta = [np.random.rand() for _ in range(self.X.shape[1])]

    def lin_combo(self, X):
        """"""

        return X.dot(self.theta)

    def sigmoid(self, t):
        """"""

        return 1 / (1 + np.exp(-t))

    def prediction_boundary(self, p_hat):
        """"""

        return [1 if x >= 0.5 else 0 for x in p_hat]


    def log_loss_gradient(self, p_hat):
        """"""

        return self.X.T.dot(p_hat - self.y) / self.m

    def fit(self, n_iterations):

        # Using gradient descent to solve
        for i in range(n_iterations):

            # Calculate the linear combination
            linear_combination = self.lin_combo(X=self.X)

            # Sigmoid it
            p_hat = self.sigmoid(t=linear_combination)

            # Gradient of logloss function
            theta_grad = self.log_loss_gradient(p_hat=p_hat)

            # Update the params
            self.theta -= self.eta * theta_grad

    def predict(self, X):
        """"""

        # Calculate the linear combination
        X = self._add_intercept(X=X)
        linear_combination = self.lin_combo(X=X)

        # Sigmoid it
        p_hat = self.sigmoid(t=linear_combination)

        return self.prediction_boundary(p_hat=p_hat)


from sklearn.linear_model import LogisticRegression as LR

# import sklearn
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1
mine = LogisticRegression(X=X, y=y)
mine.fit(n_iterations=300000)
preds = mine.predict(X=X)
print(preds)
print(np.mean(preds == y))

print('----')

sklearn_model = LR(solver='lbfgs')
sklearn_model.fit(X=X, y=y)
preds = sklearn_model.predict(X=X)
print(np.mean(preds == y))

# print(sklearn_model.intercept_, sklearn_model.coef_)
