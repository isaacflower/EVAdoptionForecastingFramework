import numpy as np

def probit(p):
    """Apply the probit (inverse CDF of normal) transformation."""
    from scipy.stats import norm
    p = np.clip(p, 0.0001, 0.9999)
    return norm.ppf(p)

def invprobit(X):
    """Apply the inverse probit (CDF of normal) transformation."""
    from scipy.stats import norm
    return norm.cdf(X)