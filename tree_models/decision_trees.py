"""

"""

import numpy as np


class CART:

    def __init__(self,
                 X,
                 y,
                 type,
                 max_depth=None):
        self.X = X
        self.y = y
        self.type = type
        self.max_depth = max_depth

        self.tree_depth = 0

        self._safe_start()

    def _safe_start(self):
        """Meant to check the type of data that y is"""

        # Checking the type of problem
        if self.type not in ['classification', 'regression']:
            raise AttributeError('Given type must be either classification or regression.')

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


    def fit(self):
        """"""

        if self.type == 'classification':
            self._c_fit()
        else:
            self._r_fit()

    def _c_fit(self):
        """"""

        pass

    def _r_fit(self):
        """"""

        pass

    def _gini(self, node_instances_idx):
        """"""

        # Pulling out the values of the y vector at the given indexes
        current_y = self.y[node_instances_idx]

        # Very quickly, we can see if the split is homogeneous
        if len(np.unique(current_y)) == 1:
            return 0

        # If not, we need to do some calculations
        # First pulling out the count of each one of the classes
        class_counts = {
            x: np.sum([j == x for j in current_y])
            for x in np.unique(current_y)
        }

        m = len(current_y)

        return np.sum([class_counts[x]/m * (1 - class_counts[x]/m) for x in class_counts.keys()])


X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [0, 0, 0]
])

y = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

a = CART(X=X, y=y, type='classification')
print(a._gini(node_instances_idx=[0, 1, 2, 3, 4, 5]))