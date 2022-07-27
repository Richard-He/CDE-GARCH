
import numpy as np

import scipy.linalg as linalg
import scipy.stats as stats
# from syndata_generate import data_gen_test

def v_lambda(x):
    """

    :param x: Nxp empirical observations
    :return:
    """
    p = x.shape[1]
    N = x.shape[0]
    sum = np.zeros((p, p))
    for i in range(N):
        sum = sum + np.outer(x[i], x[i])
    sum /= N
    [V_e, lambda_e, Vp_e] = linalg.svd(sum)
    # print(sum)
    # print(V_e)
    return V_e, lambda_e


# Test if the rest of the vectors are independent.
def sigma_tau(x, tau, V_e, s_e):
    p = x.shape[1]
    N = x.shape[0]
    V_esc = V_e[:, s_e:]
    Sigma_tau = np.zeros((p - s_e, p - s_e))
    for t in range(N):
        if t - tau < 0:
            Sigma_tau = Sigma_tau + np.outer(V_esc.T @ x[t], V_esc.T @ x[N + t-tau]) / N
        else:
            Sigma_tau = Sigma_tau + np.outer(V_esc.T @ x[t], V_esc.T @ x[t-tau]) / N
    return Sigma_tau


def Gq(x, V_e, s_e, q):
    sum = 0
    for tau in range(1, q+1):
        Sigma_t = sigma_tau(x, q, V_e, s_e)
        sum += np.trace(Sigma_t @ Sigma_t.T)
    return sum


def sl_e(l, V_e, x, s_e):
    p = x.shape[1]
    if l == 1:
        return np.trace(sigma_tau(x, 0, V_e, s_e)) / (p - s_e)
    elif l == 2:
        return np.trace(sigma_tau(x, 0, V_e, s_e) @ sigma_tau(x, 0, V_e, s_e)) / (p - s_e)


def calc_xi_gaussian(q, cp, s2_e, s1_e):
    return np.sqrt(2 * q * cp ** 2 * (s2_e-cp * s1_e**2) ** 2)



#\wh\sigma_2^2
def sigma2_e(x, lambda_e, V_e):
    N = x.shape[0]
    return np.sum([linalg.norm((V_e.T @ x[i]) ** 4, 1) / (linalg.norm(lambda_e, 2) ** 2) for i in range(N)]) / N


def test_s_gaussian(x, V_e, q, s_e):
    p = x.shape[1]
    N = x.shape[0]
    s1_e = sl_e(1, V_e, x, s_e)
    s2_e = sl_e(2, V_e, x, s_e)
    cp = (p - s_e) / N
    xi = calc_xi_gaussian(q, cp, s2_e, s1_e)
    Z_alpha = 1.68  # The 95 percentile of Normal distribution
    Gq1 = Gq(x, V_e, s_e, q)
    # print(sigma2_e2)
    #print("A", np.abs(Gq1 - q * N * cp ** 2 * s1_e**2))
    #print("B", xi)
    if np.abs(Gq1 - q * N * cp ** 2 * s1_e**2) > np.abs(Z_alpha * xi):
        return False
    else:
        return True


def test_s(x, V_e, q, s_e):
    p = x.shape[1]
    N = x.shape[0]
    s1_e = sl_e(1, V_e, x, s_e)
    s2_e = sl_e(2, V_e, x, s_e)
    cp = (p - s_e) / N
    xi = calc_xi_gaussian(q, cp, s2_e, s1_e)
    Z_alpha = 1.68  # The 95 percentile of Normal distribution
    Gq1 = Gq(x, V_e, s_e, q)
    # print(sigma2_e2)
    #print("A", np.abs(Gq1 - q * N * cp ** 2 * s1_e ** 2))
    #print("B", xi)
    if np.abs(Gq1 - q * N * cp ** 2 * s1_e ** 2) > np.abs(Z_alpha * xi):
        return False
    else:
        return True



def binary_search_s_gaussian(x, V_e, q, lambda_e):
    p = x.shape[1]
    h = p
    l = 1
    s_e = np.int64(np.ceil((h + l) / 2))
    # sigma2_e2 = sigma2_e( x, lambda_e, V_e)
    # print(sigma2_e2)

    #sigma2_e2 = 3
    while h - l > 1:
        if test_s_gaussian(x, V_e, q, s_e):
            h = s_e
        else:
            l = s_e
        s_e = np.int64(np.ceil((h + l) / 2))
    return s_e



def test_power(p, Nop, s,rds, kappa, q, times):
    corr = 0
    for i in range(times):
        x = data_gen_test(p, p*Nop, s, rds, kappa)
        V_e, lambda_e = v_lambda(x)
        if not test_s_gaussian(x, V_e, q, 0):
            corr += 1
    return corr/times



# print(test_fdr(64,1,1,0.5,1,1,1000))