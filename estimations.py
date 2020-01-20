"""

"""

import math
import numpy as np
from scipy import stats
from probability import pdf_normal, log_likelihood


def mle(x, mu, sigma):
    """"""
    # Estimating θ_μ by taking the derivative of the log-likelihood function and
    # finding the maximum by setting it = 0
    # δx / δ θ_μ = 0 --> ∑(x) / n

    # Estimating θ_σ by taking the derivative of the log-likelihood function and
    # finding the maximum by setting it = 0
    # δx / δ θ_σ = 0 --> ∑(x - μ)^2 / n

    # MLE Criterion: choose the parameters, θ, wich maximized the log-likelihood
    # function.
    #  If we are lucky, we should be able to solve this analytically, by
    # computing the derivative and setting it equal to 0.
    # Said more precisely:
    # We check the _critical points_ by setting the derivatiive to 0
    # This checks inflection points and bounday points and returns the one
    # With the highest value
    # That is, choose the parameters that give the highest likelihood of
    # generating the training data

    # MLE of a probabiliistic model that uses the Gaussian distribution: mu and
    # σ are simply the empirical values

    # Gamma:
    # This function computes the log of the gamma disctribution given a values
    # of alpha
    stats.special.gammaln()

    # Unfortunately for Gamma, we can't really solve for the maximum analytically,
    # so we turn to the numerical technique of gradient ascent
    # For this, we take the partial derivative of the log-likelihood function
    # with respect to alpha and beta
    stats.special.digamma()

    # MLE isn't the best to use with small amounts of training data (data sparsity).
    # One issue is that it will assign 0 probablity to data not in the training set.
    # Regularization is usually used to help prevent overfitting but this has
    # some issues with MLE -- encourages degenerate solutions of θ=0.
    # Instead, we can use bayesian techniques

    # With MLE, obs are treated as random vars but the parameters were not.
    # With Bayesian methods, we treat all as random vars.
    # p(θ|D) is the joint distribution (posterior)
    # p(θ) is the prior
    # p(D|θ) is the likelihood
    # p(θ|D) propto p(θ)p(D|θ)
    # The "posterior predictive distribution" p(D'|D) which is the distribution
    # over future obs given historical obs

    pass


