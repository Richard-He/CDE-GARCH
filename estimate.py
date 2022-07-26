import numpy as np
import scipy.linalg as linalg
# Loss function evaluation

ps = [256]
Nops = [2,4,6,8,10]
rds = 0.15
kappa = 2


def v_lambda(x):
    """

    :param x: Nxp empirical observations
    :return:
    """
    N = x.shape[0]
    p = x.shape[1]
    sum = np.zeros((p, p))
    for i in range(N):
        sum += x[i].T @ x[i]/N
    [V_e, lambda_e, Vp_e] = linalg.svd(sum)
    return V_e, lambda_e.T


# Test if the rest of the vectors are independent.
def sigma_tau(x, tau, N, p, V_e, s_e):
    V_esc = V_e[:, s_e:]
    Sigma_tau = np.zeros((p - s_e, p - s_e))
    for t in range(N):
        if t-tau < 0:
            Sigma_tau += V_esc.T @ x[t].T @ x[N + t-tau]@ V_esc/N
        else:
            Sigma_tau += V_esc.T @ x[t].T @ x[t-tau]@ V_esc/N
    return Sigma_tau


def Gq(x, V_e, s_e, q):
    N = x.shape[0]
    p = x.shape[1]
    sum = np.zeros((p - s_e, p - s_e))
    for tau in range(1, q):
        sum += sigma_tau(x, q, N, p, V_e, s_e)
    return sum


def sl_e(l, lambda_e, s_e):
    return np.sum(lambda_e[s_e:]**l)


def calc_xi(q, cp, s2_e, N, sigma2_e2):
    return  2 * q * cp**2 * s2_e**2+1/N*q*cp**2 *( (sigma2_e2-1)*s2_e )**2

# def sigma_c(q, cp, s_2, s1_e, s2_e , sigma2_e):
#     return 2 * q * cp**2 * s2_e**2 +4 * q**2 * cp**3 * ( sigma2_e**2 - 1) * s1_e**2 * s2_e


def sigma2_e(N, x, lambda_e):
    return linalg.norm(x*x, 'fro') ** 2/ N / (linalg.norm(lambda_e, 2)**2)


def test_s(x,  V_e, q, s_e, lambda_e, sigma2_e2):
    N = x.shape[0]
    p = x.shape[1]
    s1_e = sl_e(1, lambda_e, s_e)
    s2_e = sl_e(2, lambda_e, s_e)
    cp = (p - s_e)/N
    xi = calc_xi(q, cp, s2_e, N, sigma2_e2)
    Z_alpha = 1.645 # The 95 percentile of Normal distribution
    if Gq(x, N, p, V_e, s_e, q) - q * N * cp**2 * s1_e**2 + 1 / N * q * cp * \
            ((sigma2_e2-1) * s2_e) > Z_alpha * xi:
        return False
    else:
        return True


def binary_seach_s(x, V_e, q, lambda_e):
    N = x.shape[0]
    p = x.shape[1]
    h = p
    l = 1
    s_e = np.ceil((h+l)/ 2)
    sigma2_e2 = sigma2_e(N, x, lambda_e)
    while h-l > 1:
        if test_s(x, N, p, V_e, q, s_e, lambda_e, sigma2_e2):
            h = s_e
        else:
            l = s_e
        s_e = np.ceil((h+l)/2)
    return s_e


# Calculation of estimate of lambda
def lambda_est(x_h, param, s_e, lambda_e):
    """

    :param x_h: Nx1xp permuted observations
    :param param: vectorized A and B
    :param s_e: estimated s
    :param lambda_e: estimated marginal lambda
    :return: estimated lambda
    """
    N = x_h.shape[0]
    p = x_h.shape[2]
    flat_a = param[:s_e * p]
    flat_b = param[s_e * p:]
    A_es = flat_a.reshape(s_e, p)
    B_es = flat_b.reshape(s_e, p)
    lambda_es = np.zeros(shape=[s_e,1,p])
    lambda_es[0, :s_e] = lambda_e[0, :s_e]
    for i in range(1, N):
        ximinus = x_h[i-1]
        lambda_es[i, :s_e] = lambda_es[0, :s_e] - (A_es + B_es) @ lambda_es[0]\
                                + A_es @ (ximinus **2) + B_es @ lambda_es[i-1]
    return lambda_es #[N, 1, p]


def calc_xh(x, V_e):
    N = x.shape[0]
    p = x.shape[1]
    x_h = np.zeros([N, 1, p])
    for i in range(N):
        x_h[i] = V_e.T @ x[i].T
    return x_h


# Calculation of Loss
def loss(x_h, s_e, lambda_es):
    N = x_h.shape[0]
    return (np.sum(np.log(lambda_es)) + x_h[:, :, :s_e]**2/lambda_es[:, :, s_e])/ N


# Calculation of Gradient
def loss_grad(x_h, param, s_e, p, lambda_es):
    N = x_h.shape[0]
    p = x_h.shape[2]
    lambda_grada = np.zeros(N, p, p*s_e)
    lambda_gradb = np.zeros(N, p, p*s_e)


