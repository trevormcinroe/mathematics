"""
Some solvers for simple linear systems.
Last updated: 10.2.2019
"""

import numpy as np
import math
import pandas as pd


def gj_method(mat, augment_c):
    """A (procedural?) function for solving a linear system using the Gauss-Jordan method
    Contrary to the np.lingalg method arsenal, the input to this function expects the matrix in its augmented form.
    That is,

    [ a b c D ]
    [ e f g H ]
    [i j k L ]

    Where the capital letters represent the column to the right of the augment-dividing-line.

    WARNING: The current version of this function only supports 2x2 and 3x3 matricies.

    Args:
        mat (np.matrix): an augmented matrix
        augment_c (int): the column number where the augmented portion of the matrix is located

    Returns:
        a matrix where the left side of the augmentation is the Identity Matrix
        the function will fail with a "divide by zero" error in cases with no or inf solutions

    """

    if augment_c > mat.shape[1]:
        raise ValueError('You have specified a value of augement_c that is outside of the range of the given matrix.')

    # Allows us to iterate through column
    i = 0

    # Allows us to find out which element should be == 1
    protected = 0

    while i < augment_c:

        # Selecting the rows we wish to perform operations on
        # First, create a list with an element for each row
        row_list = [x for x in range(mat.shape[0])]

        # Now removing the row that contains the "protected" element (the 1 in the Identity Matrix)
        # This leaves us with the row numbers that we could be altering
        del row_list[protected]

        # Explicit check for the length of row_list
        # This will allow this algo to generalize to any size matrix
        if len(row_list) > 2:

            # Looping through row_list 2 times
            # selecting two of the elements randomly
            # then reassigning the row_list variable
            rl = []

            for i in range(2):
                rl.append(np.random.choice(row_list))

            row_list = rl

        # Now we will be looping through each row and performing the row update operation
        for row in row_list:

            # Explicit check to see if the given row is already 0, if so, continue
            if mat[row, i] == 0:
                continue

            # Finding the multiplicands that will update the row containing the "protected" element
            # and the loop counter "row"
            # The coefn's are supposed to be dynamically typed
            # the mat_coefn's are our static references to the values in the matricies
            coef1 = mat[protected, i]
            coef2 = mat[row, i]
            mat_coef1 = coef1
            mat_coef2 = coef2

            # Surely there is a better way to do this... Please glaze over the next
            # 10 or so lines and imagine that I wrote some beautifully elegant piece of code :)
            if (mat_coef1 * -coef2) + (mat_coef2 * coef1) == 0:
                coef2 = -1 * coef2

            if (mat_coef1 * coef2) + (mat_coef2 * -coef1) == 0:
                coef1 = -1 * coef1

            # Pulling out each of the row vectors
            protected_vector = mat[protected].copy()
            row_to_update = mat[row].copy()

            # Multiplying them guys
            protected_vector = coef2 * protected_vector
            row_to_update = coef1 * row_to_update

            # Add them guys together
            row_to_update = row_to_update + protected_vector

            # Re-assigning the row to its updated form
            mat[row] = row_to_update

        # Incrementing protected to shift down the "protected" row (the 1 in the Identity Matrix)
        # Incrementing i, our counter that will iterate us through the columns
        protected += 1
        i += 1

    # Now that we have "zero-ed out" our matrix, let's finish the transformation to the Identity Matrix
    # Resetting our counters...
    i = 0
    protected = 0

    # Looping through the columns
    while i < augment_c:
        # Simply dividing the entire row vector by the "protected" value
        mat[protected, :] = mat[protected, :] / mat[protected, i]

        # Incrementing protected to shift down the "protected" row (the 1 in the Identity Matrix)
        # Incrementing i, our counter that will iterate us through the columns
        protected += 1
        i += 1

    return mat


class IO_Model:

    def __init__(self, closed=False, X=None, A=None, D=None, parameter=None):
        """"""

        self.closed = closed
        self.X = X
        self.A = A
        self.D = D
        self.parameter = parameter
        self.multipliers = None
        self.production_used = None

    def solve_for_X(self):
        """
        This function serves as a gateway depending on whether the IO_Model is open or closed
        """

        if self.A is None:
            raise AttributeError('In order to solve for X, A must be known.')

        if self.D is None:
            raise AttributeError('In order to solve for X, D must be known')

        if not self.closed:

            self._open_model_solve_X()

        else:

            self._closed_model_solve_X()

    def solve_for_production_requirements(self):
        """"""

        if self.A is None:
            raise AttributeError('In order to solve for production requirements, A is needed.')

        if self.X is None:
            raise AttributeError('In order to solve for production requirements, X is needed.')

        # Since A gives the amount,in units, of each commodity used to produce 1 unit of each commodity
        # and X gives the number of units of each commodity produced...
        # AX gives the production spend
        self.production_used = self.A * self.X

        print(self.production_used)

    def _open_model_solve_X(self):
        """"""

        # The ultimate equation to solve is X = (I-A)^-1 * D
        # First, creating the Identity Matrix of the same shape as A
        I = np.identity(n=self.A.shape[0])

        # Subtracting A from the Identity Matrix and inverting
        # If it doesn't work, throw an explicit error explaining the problem
        try:

            inverted = (I - self.A) ** -1

            # (I-A)^-1 has "important economic interpretations"
            self.multipliers = inverted

        except:

            raise RuntimeError('Unable to invert (I-A). The resulting matrix is singular.')

        # Multiplying the inverted matrix by D to solve for X
        self.X = inverted * self.D

        # Setting the amount of production used in this process
        self.production_used = self.A * self.X

        print(self.X)

    def _closed_model_solve_X(self):
        """"""

        if self.parameter is None:
            raise AttributeError('In order to solve a closed model, a parameter needs to be declared.')

        # (I-A)X = 0
        # First, creating the Identity Matrix of the same shape as A
        I = np.identity(n=self.A.shape[0])

        # (I - A)
        intermediate = I - self.A

        #

