"""
This module will eventually contain a custom implementation of the more popular tree-based algorithms

author: trevor mcinroe
date: 1.29.2020
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

        self.X_types = [str(x) for x in self.X.dtypes]

        self._safe_start()

        self.tree_depth = 0
        self.m_left = None
        self.node_queue = []

        # if we are performing classification, set out gini impurity to 1 for the root node
        if self.type == 'classification':
            self.tree_struct = {
                0: {
                    'num_instances': self.y.shape[0],
                    'instances': [True for _ in range(self.y.shape[0])],
                    'loss': 1,
                    'child': [],
                    'split_on': None,
                    'split_value': None
                }
            }

        else:
            self.tree_struct = {
                0: {
                    'num_instances': self.y.shape[0],
                    'instances': [True for _ in range(self.y.shape[0])],
                    'loss': None,
                    'child': [],
                    'split_on': None,
                    'split_value': None
                }
            }



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

        # self.X_types = [str(x) for x in self.X.dtypes]
        self.X = np.array(self.X, dtype=float)

        if not type(self.y) == np.ndarray:
            if self.type == 'classification':
                self.y = np.array(self.y, dtype=int)
            else:
                self.y = np.array(self.y, dtype=float)

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

        # On the first pass, do the split...
        split_result, lowest_col_idx, split_point = self._determine_split(current_node=0,
                                                                          current_loss=self.tree_struct[0]['loss'])

        # Adding on to our tree structure
        self.tree_struct[0]['child'].append((1, 2))
        self.tree_struct[0]['split_on'] = lowest_col_idx
        self.tree_struct[0]['split_value'] = split_point

        # Adding the next node splits to our tree struct
        if self.X_types[lowest_col_idx] == 'object':
            self.tree_struct[0 + 1] = {
                'num_instances': len(self.y[self.X[:, lowest_col_idx] == split_point]),
                'instances': self.X[:, lowest_col_idx] == split_point,
                'loss': self._gini(node_instances_idx=self.X[:, lowest_col_idx] == split_point,
                                   node_conditional=self.tree_struct[0]['instances']),
                'child': [],
                'split_on': None,
                'split_value': None
            }

            self.tree_struct[0 + 2] = {
                'num_instances': len(self.y[self.X[:, lowest_col_idx] != split_point]),
                'instances': self.X[:, lowest_col_idx] != split_point,
                'loss': self._gini(node_instances_idx=self.X[:, lowest_col_idx] != split_point,
                                   node_conditional=self.tree_struct[0]['instances']),
                'child': [],
                'split_on': None,
                'split_value': None
            }

        else:
            self.tree_struct[0 + 1] = {
                'num_instances': len(self.y[self.X[:, lowest_col_idx] < split_point]),
                'instances': self.X[:, lowest_col_idx] < split_point,
                'loss': self._gini(node_instances_idx=self.X[:, lowest_col_idx] < split_point,
                                   node_conditional=self.tree_struct[0]['instances']),
                'child': [],
                'split_on': None,
                'split_value': None
            }

            self.tree_struct[0 + 2] = {
                'num_instances': len(self.y[self.X[:, lowest_col_idx] >= split_point]),
                'instances': self.X[:, lowest_col_idx] >= split_point,
                'loss': self._gini(node_instances_idx=self.X[:, lowest_col_idx] >= split_point,
                                   node_conditional=self.tree_struct[0]['instances']),
                'child': [],
                'split_on': None,
                'split_value': None
            }

        self.tree_depth = 1

        self.node_queue.append((1, 2))


        while np.all([self.tree_depth < self.max_depth,
                      split_result != 'none found']):

            current_node = self.node_queue.pop(0)

            for node in current_node:

                split_result, lowest_col_idx, split_point = self._determine_split(current_node=node,
                                                                                  current_loss=self.tree_struct[node]['loss'])
                # If the split is not made in the above method, we need to catch and continue
                if not type(lowest_col_idx) == int:
                    continue

                # Pulling out the highest node...
                try:
                    highest_node = np.max([x for x in self.node_queue])
                except:
                    highest_node = np.max(current_node)

                # Adding on to our tree structure
                self.tree_struct[node]['child'].append((highest_node+1, highest_node+2))
                self.tree_struct[node]['split_on'] = lowest_col_idx
                self.tree_struct[node]['split_value'] = split_point

                # Adding to the node queue
                self.node_queue.append((highest_node+1, highest_node+2))


                if self.X_types[lowest_col_idx] == 'object':

                    self.tree_struct[highest_node + 1] = {
                        'num_instances': len(self.y[self.X[:, lowest_col_idx] < split_point]),
                        'instances': self.X[:, lowest_col_idx] == split_point,
                        'loss': self._gini(node_instances_idx=self.X[:, lowest_col_idx] == split_point,
                                           node_conditional=self.tree_struct[node]['instances']),
                        'child': [],
                        'split_on': None,
                        'split_value': None
                    }

                    self.tree_struct[highest_node + 2] = {
                        'num_instances': len(self.y[self.X[:, lowest_col_idx] >= split_point]),
                        'instances': self.X[:, lowest_col_idx] != split_point,
                        'loss': self._gini(node_instances_idx=self.X[:, lowest_col_idx] != split_point,
                                           node_conditional=self.tree_struct[node]['instances']),
                        'child': [],
                        'split_on': None,
                        'split_value': None
                    }

                else:
                    self.tree_struct[highest_node + 1] = {
                        'num_instances': len(self.y[self.X[:, lowest_col_idx] < split_point]),
                        'instances': self.X[:, lowest_col_idx] < split_point,
                        'loss': self._gini(node_instances_idx=self.X[:, lowest_col_idx] < split_point,
                                           node_conditional=self.tree_struct[node]['instances']),
                        'child': [],
                        'split_on': None,
                        'split_value': None
                    }

                    self.tree_struct[highest_node + 2] = {
                        'num_instances': len(self.y[self.X[:, lowest_col_idx] >= split_point]),
                        'instances': self.X[:, lowest_col_idx] >= split_point,
                        'loss': self._gini(node_instances_idx=self.X[:, lowest_col_idx] >= split_point,
                                           node_conditional=self.tree_struct[node]['instances']),
                        'child': [],
                        'split_on': None,
                        'split_value': None
                    }
            self.tree_depth += 1


    def _r_fit(self):
        """"""

        pass

    def _gini(self, node_instances_idx, node_conditional):
        """

        Args:
            node_instances_idx (list): the indexes of self.y that correspond to the instances in
                                       the currently considered split
            node_conditional:

        Returns:
            the gini impurity
        """

        # Pulling out the values of the y vector at the given indexes
        current_y = self.y[(node_instances_idx) * (node_conditional)]

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

        num_node_instances = self.tree_struct[current_node]['num_instances']
        node_conditional = self.tree_struct[current_node]['instances']

        if self.type == 'classification':
            # Need an explicit check in here to see if the node already is pure
            # This will help us to avoid any calculation below
            if len(np.unique(self.y[self.tree_struct[current_node]['instances']])) == 1:
                return 'none found', None, None

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
                            left = self._gini(node_instances_idx=self.X[:, col_idx] == clss,
                                              node_conditional=node_conditional)
                            right = self._gini(node_instances_idx=self.X[:, col_idx] != clss,
                                               node_conditional=node_conditional)

                            w = (np.sum(self.X[node_conditional, col_idx] == clss)/num_node_instances * left) \
                                + (np.sum(self.X[node_conditional, col_idx] != clss)/num_node_instances * right)

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
                            left = self._gini(node_instances_idx=self.X[:, col_idx] < split,
                                              node_conditional=node_conditional)
                            right = self._gini(node_instances_idx=self.X[:, col_idx] >= split,
                                              node_conditional=node_conditional)

                            w = (np.sum(self.X[node_conditional, col_idx] < split)/num_node_instances * left) \
                                + (np.sum(self.X[node_conditional, col_idx] >= split)/num_node_instances * right)

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
                # if not lowest_col_idx:
                if lowest_gini == current_loss:
                    return 'none found', None, None
                else:
                    return lowest_gini, lowest_col_idx, split_point

        else:
            pass


    def predict(self, X):
        """"""

        X = self._predict_safety(X=X)

        if self.type == 'classification':
            return self._c_predict(X=X)
        else:
            pass

    def _predict_safety(self, X):
        """"""

        if not type(X) == np.ndarray:
            X = np.array(X, dtype=float)

        if not X.shape[1] == self.X.shape[1]:
            raise AttributeError('Given X matrix does not have the same number of columns as the data this model was trained on.')

        return X

    def _c_predict(self, X):
        """"""

        predictions = []

        # Need to traverse the tree for each row in the given X
        for r in range(X.shape[0]):
            row_data = X[r, :]

            current_node = 0

            at_end = False

            while not at_end:
                # Perhaps we have reached the end of the tree
                if len(self.tree_struct[current_node]['child']) == 0:
                    # For classification, need to take the majority vote
                    prediction = np.argmax(np.bincount(self.y[self.tree_struct[current_node]['instances']]))

                    predictions.append(prediction)
                    at_end = True

                elif self.X_types[self.tree_struct[current_node]['split_on']] == 'object':
                    if row_data[self.tree_struct[current_node]['split_on']] \
                            == self.tree_struct[current_node]['split_value']:
                        # go left
                        current_node = self.tree_struct[current_node]['child'][0][0]

                    else:
                        # go right
                        current_node = self.tree_struct[current_node]['child'][0][1]
                else:
                    # Here in the logic flow, we are dealing with a numeric predictor
                    if row_data[self.tree_struct[current_node]['split_on']] \
                            < self.tree_struct[current_node]['split_value']:
                        # go left
                        current_node = self.tree_struct[current_node]['child'][0][0]

                    else:
                        # go right
                        current_node = self.tree_struct[current_node]['child'][0][1]

        return predictions


from sklearn import datasets
from sklearn.metrics import accuracy_score
iris_data = datasets.load_iris()


X = pd.DataFrame(iris_data['data'])
y = iris_data['target']

a = CART(X=X, y=y, type='classification', max_depth=4)

a.fit()

# print(a.tree_struct)

preds = a.predict(X=X)


print(f'Accuracy: {accuracy_score(y_true=y, y_pred=preds)}')

