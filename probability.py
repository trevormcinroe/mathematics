"""

"""

import math
import numpy as np


#######################
# == Distributions == #
#######################
def pdf_normal(x, mu, sigma):
    """Probability that our sample, x, from our random variable, x, is from ~N(mu,sigma)

    Args:
        x (numeric): a single value that represents a sample draw from data
        mu (numeric): a single value that represents the mean of the target distribution
        sigma (numeric): a single value that represents the standard deviation of the target distribution

    Returns:
        (float) - the probability of x being from ~N(mu,sigma)

    """

    return (1 / (math.sqrt(2 * math.pi * sigma**2))) * math.e**((-(x-mu)**2) / (2 * sigma**2))


def log_likelihood(x, mu, sigma):
    """Log-likelihood that our sample, x, from our random variable, x, is from ~N(mu,sigma)

    Args:
        x (numeric): a single value that represents a sample draw from data
        mu (numeric): a single value that represents the mean of the target distribution
        sigma (numeric): a single value that represents the standard deviation of the target distribution

    Returns:
        (float) - the probability of x being from ~N(mu,sigma)
    """

    first_term = -(len(x) / 2) * np.log(2 * math.pi)

    second_term = (len(x) / 2) * np.log(sigma**2)

    summation_term = (1 / (2 * sigma**2)) * np.sum(np.subtract(x, mu)**2)

    return first_term - second_term - summation_term
