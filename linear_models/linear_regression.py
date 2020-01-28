"""

"""


class LinearRegression():
    """A class that performs a linear regression, whether simple or multiple.

    Currently, this class support two estimation frameworks:
    (1) Ordinary Least Squares - called with the .ols() method
    (2) Batch Gradient Descent - called with the .gradient_descent() method

    The mathematics for each of the two can be found in the notebook itself... somewhere

    Attributes:
        x_data: a matrix of your independent variables, pandas
        y_data: a matrix  of your dependent variable, pandas
        coefficients: a dictionary of estimated coefficeints that are computed
        intercept: the estimated intercept
        fit_metrics: a dictionoary of computed fit metrics: currently R2, MSE, and MAE
        estimation_framework: a description of the framework chosen, meant as a reminder
    """

    def __init__(self):
        self.x_data = None
        self.y_data = None
        self.coefficients = []
        self.intercept = None
        self.fit_metrics = {}
        self.estimation_framework = None
        self.residuals = []
        self.fitted_values = []

    def r2(self):
        """Computes the r2 values for the fitted regression
        """

        # Calculating the fitted values first
        # Pulling out the coefficients, variables from our x_data
        coefs = [v for k, v in self.coefficients.items()]

        # Finding the fitted values for the model
        y_hat = self.x_data.dot(coefs) + self.intercept

        numerator = np.sum(np.multiply((self.y_data - np.mean(self.y_data)),
                                       (y_hat - np.mean(y_hat)))) ** 2

        denom = (np.sum(((self.y_data - np.mean(self.y_data)) ** 2))) * (np.sum(((y_hat - np.mean(y_hat)) ** 2)))

        return (numerator / denom)

    def mse(self):
        """Computes the MSE for the model
        """

        # Calculating the fitted values first
        # Pulling out the coefficients, variables from our x_data
        coefs = [v for k, v in self.coefficients.items()]

        # Finding the fitted values for the model
        y_hat = self.x_data.dot(coefs) + self.intercept

        return np.sum((self.y_data - y_hat) ** 2) / self.y_data.size

    def mae(self):
        """Computes the MAE for the model
        """

        # Calculating the fitted values first
        # Pulling out the coefficients, variables from our x_data
        coefs = [v for k, v in self.coefficients.items()]

        # Finding the fitted values for the model
        y_hat = self.x_data.dot(coefs) + self.intercept

        return np.sum(np.abs(self.y_data - y_hat)) / self.y_data.size

    def errors(self):
        """Computes a vector of residuals for the model
        """

        # Calculating the fitted values first
        # Pulling out the coefficients, variables from our x_data
        coefs = [v for k, v in self.coefficients.items()]

        # Finding the fitted values for the model
        y_hat = self.x_data.dot(coefs) + self.intercept

        return self.y_data - y_hat

    def fit(self):
        """Computes a vector of fitted values for the model
        """

        # Calculating the fitted values first
        # Pulling out the coefficients, variables from our x_data
        coefs = [v for k, v in self.coefficients.items()]

        # Finding the fitted values for the model
        return self.x_data.dot(coefs) + self.intercept

    def ols(self, standardize=False):
        """Computes the Ordinary Least Squares estimator for a linear regression
        """

        # If the user specifies that they want their data, standardized, do so
        if standardize:

            # Init an empty dict
            # We'll fill this with information about the data
            standardized = {}

            # Looping throught the variables to find the standard deviation
            # This will save us compute later if some of the variables are
            # already standardized
            for var in self.x_data.columns:
                standardized[var] = np.std(self.x_data[var])

            # Init an empty dataframe
            df_std = pd.DataFrame()

            # If the variable is already standardized, simply add it as a column
            # to the dataframe, else standardize first
            for k, v in standardized.items():
                if v == 1:
                    df_std[k] = self.x_data[k]
                else:
                    df_std[k] = np.subtract(self.x_data[k],
                                            np.mean(self.x_data[k])) / v

            self.x_data = df_std

        # First, let's create a np.array of our Independent Variable pd.DataFrame data
        # We save ourselves from forgetting by transposing this matrix upon creation
        x_mat = np.array([
            self.x_data[x].tolist()
            for x in self.x_data
        ]).T

        # Now do the same (+ transposing) for the Dependent Variable matrix
        y_mat = np.array(self.y_data.tolist()).T

        # As shown in the math above, we need to add a column of 1s for the intercept term
        # The fancy [..., None] portion vertically adjusts the data and makes it a
        # list of lists, which is necessary for the combination to the x_mat
        intercept_mat = np.ones(shape=y_mat.shape)[..., None]

        # Combining our intercept term into our x_mat
        x_mat = np.concatenate((intercept_mat, x_mat), 1)

        # Finally, actually perform the math
        result_matrix = np.linalg.inv(x_mat.T.dot(x_mat)).dot(x_mat.T).dot(y_mat)

        # Setting the intercept of the object
        self.intercept = result_matrix[0]

        # Creating a dictionary for the coefficients of the KEY, VALUE form:
        # {'Variable Name': calculated value}
        self.coefficients = dict(zip(self.x_data.columns, result_matrix[1:]))

        # A a reminder for what we have fitted, setting the estimation framework
        self.estimation_framework = 'Ordinary Least Squares'

        # Now adding our fit metrics...
        self.fit_metrics['R2'] = self.r2()
        self.fit_metrics['mse'] = self.mse()
        self.fit_metrics['mae'] = self.mae()

        # Adding residuals
        self.residuals = self.errors()

        # Adding the fitted line
        self.fitted_values = self.fit()

    def gradient_descent(self, eta, n_iterations, standardize=False):
        """Computes an estimate for the regression using gradient descent
        Args:
            eta: learning rate, float
            n_iterations: the number of iterations for the algorithm, integer
            standardize: should the data be standardized?, bool
        """

        # If the user specifies that they want their data, standardized, do so
        if standardize:

            # Init an empty dict
            # We'll fill this with information about the data
            standardized = {}

            # Looping throught the variables to find the standard deviation
            # This will save us compute later if some of the variables are
            # already standardized
            for var in self.x_data.columns:
                standardized[var] = np.std(self.x_data[var])

            # Init an empty dataframe
            df_std = pd.DataFrame()

            # If the variable is already standardized, simply add it as a column
            # to the dataframe, else standardize first
            for k, v in standardized.items():
                if v == 1:
                    df_std[k] = self.x_data[k]
                else:
                    df_std[k] = np.subtract(self.x_data[k],
                                            np.mean(self.x_data[k])) / v

            self.x_data = df_std

        # First, we need to initialize our parameters
        # 'm' gives us the size of the data, which is required to calculate the gradients
        m = self.y_data.size

        # First, let's create a np.array of our Independent Variable pd.DataFrame data
        # We save ourselves from forgetting by transposing this matrix upon creation
        x_mat = np.array([
            self.x_data[x].tolist()
            for x in self.x_data
        ]).T

        # We need to randomly initialize our parameters for the process
        # We add 1 to the number of variables as there needs to be a theta for the intercept
        theta = np.random.randn(self.x_data.shape[1] + 1, 1)

        # Doing the same thing as above... creating our Y vector but using
        # different syntax as an example for the reader
        y_mat = np.array(self.y_data.tolist()).reshape(-1, 1)

        # Doing the same thing as above... appending in our constant to our X matrix
        x_mat = np.concatenate((np.ones(shape=m)[..., None], x_mat), 1)

        # And thusly begins the gradient loop...
        for i in range(n_iterations):
            # We use MSE as the loss metric as, theoretically speaking, it should always
            # be "convex" in the confines of a linear regression setting
            # That is, given any two points on the curve, the line that connects them will
            # never cross the curve itself.
            # In practical terms, this means that there is always one GLOBAL minimum and
            # no LOCAL minima
            gradients = 2 / m * x_mat.T.dot(x_mat.dot(theta) - y_mat)

            theta = theta - eta * gradients

        # Once done, return the values
        # Setting the intercept of the object
        self.intercept = theta[0][0]

        # Creating a dictionary for the coefficients of the KEY, VALUE form:
        # {'Variable Name': calculated value}
        self.coefficients = dict(zip(self.x_data.columns,
                                     [x[0] for x in theta[1:]]))

        # Finally, as a reminder for what we have fitted, setting the estimation framework
        self.estimation_framework = 'Gradient Descent'

        # Now adding our fit metrics...
        self.fit_metrics['R2'] = self.r2()
        self.fit_metrics['mse'] = self.mse()
        self.fit_metrics['mae'] = self.mae()

        # Adding residuals
        self.residuals = self.errors()

        # Adding the fitted line
        self.fitted_values = self.fit()