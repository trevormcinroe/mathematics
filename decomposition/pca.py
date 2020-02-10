"""

author: trevor mcinroe
date: 2.10.2019
"""

import numpy as np


class PCA:

    def __init__(self,
                 X):

        self.X = X


    def _check_norm(self):
        """If the given data isn't 0-centered with unit variance, transform it to be so"""

        if not np.mean(self.X) == 0:
            self.X = self.X - np.mean(self.X, axis=0)
            self.X = self.X / np.std(self.X, axis=0)

            return self

        else:
            pass

