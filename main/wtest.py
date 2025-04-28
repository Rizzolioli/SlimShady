import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.utils import resample


def calculate_contingency(X1, X2, Y):
    """
    Create 2x9 contingency table from two SNPs and binary phenotype.
    """
    k = 9
    table_case = np.zeros(k)
    table_control = np.zeros(k)

    pair_code = X1 * 3 + X2  # Unique encoding for genotypes
    for i in range(k):
        table_case[i] = np.sum((pair_code == i) & (Y == 1))
        table_control[i] = np.sum((pair_code == i) & (Y == 0))

    N1 = np.sum(Y == 1)
    N0 = np.sum(Y == 0)
    p1 = table_case / N1
    p0 = table_control / N0
    return p1, p0, table_case, table_control, N1, N0


def compute_SE(n0i, n1i, N0, N1):
    """
    Standard error for log odds ratio.
    """
    return np.sqrt(1 / n0i + 1 / n1i + 1 / (N0 - n0i) + 1 / (N1 - n1i))


def compute_X2(p1, p0, n1, n0, N1, N0):
    """
    Compute unscaled X^2 statistic.
    """
    k = len(p1)
    X2 = 0
    xi_list = []

    for i in range(k):
        if p1[i] in [0, 1] or p0[i] in [0, 1]:  # Avoid division by zero
            continue
        log_OR = np.log((p1[i] / (1 - p1[i])) / (p0[i] / (1 - p0[i])))
        SE = compute_SE(n0[i], n1[i], N0, N1)
        xi = log_OR / SE
        xi_list.append(xi)
        X2 += xi ** 2
    return X2, np.array(xi_list)


def estimate_h_f(xi_list):
    """
    Estimate h and f from xi components using moment matching.
    """
    k = len(xi_list)
    var_x2 = np.var(xi_list ** 2, ddof=1) * k
    c = var_x2 / (2 * k)
    f = (2 * k ** 2) / var_x2
    h = 1 / c
    return h, f


def w_test(X1, X2, Y, use_bootstrap=False, B=200):
    """
    Run the W-test for a pair of SNPs X1, X2 against phenotype Y.
    """
    p1, p0, n1, n0, N1, N0 = calculate_contingency(X1, X2, Y)

    # Continuity correction if needed
    if np.any(n1 == 0) or np.any(n0 == 0):
        n1 = n1 + 0.5
        n0 = n0 + 0.5
        N1 += 0.5 * len(n1)
        N0 += 0.5 * len(n0)
        p1 = n1 / N1
        p0 = n0 / N0

    X2_val, xi_list = compute_X2(p1, p0, n1, n0, N1, N0)

    if use_bootstrap:
        # Estimate h and f with bootstrap
        h_list, f_list = [], []
        for _ in range(B):
            Y_perm = resample(Y, replace=False)
            _, _, n1_b, n0_b, N1_b, N0_b = calculate_contingency(X1, X2, Y_perm)
            _, xi_b = compute_X2(p1, p0, n1_b, n0_b, N1_b, N0_b)
            h_b, f_b = estimate_h_f(xi_b)
            h_list.append(h_b)
            f_list.append(f_b)
        h, f = np.mean(h_list), np.mean(f_list)
    else:
        h, f = estimate_h_f(xi_list)

    W_stat = h * X2_val
    p_val = 1 - chi2.cdf(W_stat, df=f)
    return W_stat, p_val, h, f
