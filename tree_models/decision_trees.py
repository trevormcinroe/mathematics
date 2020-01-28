"""

"""

import numpy as np
import pandas as pd

class CART:

    def __init__(self,
                 X,
                 y,
                 type,
                 max_depth=None,
                 var_search='exhaustive'):
        self.X = X
        self.y = y
        self.type = type
        self.max_depth = max_depth
        self.var_search = var_search

        self.tree_depth = 0
        self.X_types = None
        self.m_left = None
        self.tree_struct = {
            0: {
                'num_instances': self.y.shape[0],
                'loss': None,
                'child': []
            }
        }

        self._safe_start()

    def _safe_start(self):
        """Meant to check the type of data that y is"""

        # Checking the type of problem
        if self.type not in ['classification', 'regression']:
            raise AttributeError('Given type must be either classification or regression.')

        if not self.var_search in ['exhaustive', 'random']:
            raise AttributeError('var_seach must either be exhaustive or random.')

        # Dealing with the input data
        if not type(self.X) == pd.DataFrame:
            raise TypeError('Given X must be a pd.DataFrame. This helps with determining splits.')

        self.X_types = [str(x) for x in self.X.dtypes]
        self.X = np.array(self.X, dtype=float)

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

        # Making the first split

        pass

    def _r_fit(self):
        """"""

        pass

    def _gini(self, node_instances_idx):
        """

        Args:
            node_instances_idx (list): the indexes of self.y that correspond to the instances in
                                       the currently considered split

        Returns:
            the gini impurity
        """

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

    def _mse(self, node_instances_idx):
        """

        Args:
            node_instances_idx (list): the indexes of self.y that correspond to the instances in
                                       the currently considered split

        Returns:
            the MSE of the split
        """

        current_y = self.y[node_instances_idx]

        mean_y = np.mean(current_y)

        return np.sum([(x-mean_y)**2 for x in current_y])


    def _determine_split(self, current_loss, current_node):
        """

        Args:
            current_loss (float/int): either the gini impurity or MSE of the parent node

        Returns:

        """

        node_instances = self.tree_struct[current_node]['num_instances']

        if self.type == 'classification':
            if self.var_search == 'exhaustive':
                # while self.tree_depth < self.max_depth:

                lowest_gini = current_loss
                lowest_col_idx = None
                split_point = None

                for col_idx in range(self.X.shape[1]):

                    # Within this block, the types are categorical
                    if self.X_types[col_idx] == 'object':
                        splits = np.unique(self.X[:, col_idx])

                        inner_ginis = []

                        for clss in splits:
                            left = self._gini(node_instances_idx=self.X[:, col_idx] == clss)
                            right = self._gini(node_instances_idx=self.X[:, col_idx] != clss)

                            w = (np.sum()/node_instances * left) \
                                + (np.sum()/node_instances * left)

                            inner_ginis.append(w)

                    # If we reach here in the logic flow, the datatype is numeric
                    else:
                        # Let's try this with 10 splits
                        data = self.X[:, col_idx].copy()
                        data.sort()
                        splits = np.unique(np.percentile(data, q=np.linspace(0, 100, 10)))

                        # Computing the gini of each split
                        inner_ginis = []

                        for split in splits:
                            left = self._gini(node_instances_idx=self.X[:, col_idx] < split)
                            right = self._gini(node_instances_idx=self.X[:, col_idx] >= split)

                            w = (np.sum(self.X[:, col_idx] < split)/node_instances * left) \
                                + (np.sum(self.X[:, col_idx] >= split)/node_instances * right)

                            inner_ginis.append(w)

                    # Pulling out the smallest gini as well as the index
                    min_gini = np.min(inner_ginis)
                    min_gini_idx = inner_ginis.index(min_gini)

                    # Checking
                    if min_gini < lowest_gini:
                        lowest_gini = min_gini
                        lowest_col_idx = col_idx
                        split_point = splits[min_gini_idx]

                # Now that we have looked at every column...
                if not lowest_col_idx:
                    return 'none found', None, None
                else:
                    return lowest_gini, lowest_col_idx, split_point

        else:
            pass

        # Need to sweep through each column in X to determine which

X = pd.DataFrame(
    {
        'a': [x for x in range(10)],
        'b': [str(x) for x in range(10)]
    }
)

y = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

a = CART(X=X, y=y, type='classification')
print(a._gini(node_instances_idx=[0, 1, 2, 3, 4, 5]))