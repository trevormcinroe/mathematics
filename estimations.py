"""

"""

import math
import numpy as np
from probability import pdf_normal, log_likelihood

def mle(x, mu, sigma):
    """"""
    # Estimating θ_μ by taking the derivative of the log-likelihood function and
    # finding the maximum by setting it = 0
    # δx / δ θ_μ = 0 --> ∑(x) / n

    # Estimating θ_σ by taking the derivative of the log-likelihood function and
    # finding the maximum by setting it = 0
    # δx / δ θ_σ = 0 --> ∑(x - μ)^2 / n


