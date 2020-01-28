"""

"""

import numpy as np


class CART:

    def __init__(self,
                 X,
                 y):
        self.X = X
        self.y = y

        self.type = None

        self._safe_start()

    def _safe_start(self):
        """Meant to check the type of data that y is"""

        # Transforming both X and y into np.array if they are not already
        if not type(self.X) == np.ndarray:
            self.X = np.array(self.X)
        if not type(self.y) == np.ndarray:
            self.y = np.array(self.y)

        if not np.any([self.y.dtype == int, self.y.dtype == float]):
            raise TypeError('Given y vector must be numerical.')

        if np.isnan(self.y).any():
            raise AttributeError('There exists some NaNs in your y vector.')

        if np.isnan(self.X).any():
            raise AttributeError('There exists some NaNs in your X matrix.')

        if not self.X.shape[0] == self.y.shape[0]:
            raise AttributeError('Given X and y are of unequal lengths.')

        print(self.y)


X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

y = [1, 2, 3]

a = CART(X=X, y=y)