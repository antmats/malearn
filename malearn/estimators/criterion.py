import numpy as np


def entropy(distr):
    n = sum(distr)
    ps = [n_i / n for n_i in distr]
    return -sum(p * np.log2(p) if p > 0 else 0 for p in ps)


def info_gain_scorer(n_low, low_distr, n_high, high_distr):
    return -(
        (n_low * entropy(low_distr) + n_high * entropy(high_distr)) 
        / (n_low + n_high)
    )


def gini_impurity(distr):
    n = sum(distr)
    ps = [n_i / n for n_i in distr]
    return 1 - sum(p**2 for p in ps)


def gini_scorer(n_low, low_distr, n_high, high_distr):
    return -(
        (n_low * gini_impurity(low_distr) + n_high * gini_impurity(high_distr))
        / (n_low + n_high)
    )
