import numpy
import numpy as np
import numpy.random as rdm
import scipy.stats as stats
import scipy.linalg as linalg


def sample_x(lambd, V, p, heavytail):
    if heavytail:
        epsilon_t = stats.t.rvs(3, size=p)
        var = 3 / (3-2)
        epsilon_t *= 1/np.sqrt(var)
    else:
        epsilon_t = rdm.randn(p)
    if (lambd < 0).any() == True:
        exit()
    Lambd = np.diag(np.sqrt(lambd))
    x = V @ Lambd @ epsilon_t
    return x


def get_next_lambda(x_t, lambda_t, lambda_s, A, B, Vp, p):
    lambda_tp = (np.eye(p) - A - B) @ lambda_s + A @ ((Vp @ x_t) ** 2) + B @ lambda_t
    mask = lambda_tp <= 0
    lambda_tp[mask] = lambda_s[mask]
    return lambda_tp


def get_next_lambda_abs(x_t, lambda_t, lambda_s, A, B, Vp, p):
    lambda_tp = np.abs((np.eye(p) - A - B) @ lambda_s + A @ ((Vp @ x_t) ** 2) + B @ lambda_t, a_min=0, a_max=None)
    return lambda_tp


def v_lambda(x):
    p = x.shape[1]
    N = x.shape[0]
    sum = np.zeros((p, p))
    for i in range(N):
        sum = sum + np.outer(x[i], x[i])
    sum /= N
    [V_e, lambda_e, Vp_e] = linalg.svd(sum)
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

def calc_xi_ht(q, cp, s2_e, s1_e, sigma22 , T):
    return np.sqrt(2 * q * cp ** 2 * (s2_e-cp * s1_e**2) ** 2 + q * cp ** 2 * (( sigma22 - 1) * (s2_e - s1_e ** 2)) ** 2 / T)


#\wh\sigma_2^2
def sigma2_e(x, lambda_e, V_e ,s_e):
    T = x.shape[0]
    Lambda_es = (lambda_e[:s_e])**(-2)
    sum = 0
    for i in range(T):
        sum += np.sum((V_e[:,:s_e].T @ x[i]) ** 4 * (lambda_e[:s_e])**(-2))
    return sum / s_e / T


def test_s_gaussian(x, V_e, q, s_e ):
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


def test_s_ht(x, V_e, q, s_e, s):
    p = x.shape[1]
    N = x.shape[0]
    s1_e = sl_e(1, V_e, x, s_e)
    s2_e = sl_e(2, V_e, x, s_e)
    V_e, lambda_e = v_lambda(x)
    sigma22 = sigma2_e(x, lambda_e, V_e, s)
    cp = (p - s_e) / N
    xi = calc_xi_ht(q, cp, s2_e, s1_e, sigma22, T=N)
    Z_alpha = 1.68  # The 95 percentile of Normal distribution
    Gq1 = Gq(x, V_e, s_e, q)
    if np.abs(Gq1 - q * N * cp ** 2 * s1_e**2 + q * cp * (sigma22 - 1) * (s2_e - cp*s1_e**2) / N ) > np.abs(Z_alpha * xi):
        return False
    else:
        return True



def binary_search_s_gaussian(x, V_e, q, lambda_e):
    p = x.shape[1]
    h = p
    l = 1
    s_e = np.int64(np.ceil((h + l) / 2))

    while h - l > 1:
        if test_s_gaussian(x, V_e, q, s_e):
            h = s_e
        else:
            l = s_e
        s_e = np.int64(np.ceil((h + l) / 2))
    return s_e


def data_gen(p, N, s, rdsa, rdsb, kappa, ka, kb, heavytail, abs=False):
    lambda_s = np.concatenate((np.flip(np.arange(1, s + 1)) * kappa, np.zeros(p - s)))
    V = stats.ortho_group.rvs(dim=p)
    Vp = V.T

    # Preprocess A
    tempA = np.zeros(s*p)

    # print(U.shape, S_clip.shape, Vh.shape)
    # randommaska = rdm.permutation(s*p)[:ka]
    # tempA[randommaska] = np.random.choice(a=[-1, 1], size=(randommaska.shape[0])) / np.sqrt(ka) * rds
    # A = np.pad(tempA.reshape(s, p), ((0, p-s), (0, 0)), 'constant')
    tempA = np.zeros(s**2)
    randommaska = rdm.permutation(s**2)[:ka]
    tempA[randommaska] = np.random.choice(a=[1,0.9,0.8,0.7,0.6,0.5], size=(randommaska.shape[0]))[:ka]
    _, lam ,_ = linalg.svd(tempA.reshape(s,s), full_matrices=True)
    tempA *= rdsa / lam[0]
    A = np.pad(tempA.reshape(s, s), ((0, p - s), (0, p - s)), 'constant')
    # Preprocess B
    tempB = np.zeros(s**2)
    randommaskb = rdm.permutation(s**2)[:kb]
    tempB[randommaskb] = np.random.choice(a=[1,0.9,0.8,0.7,0.6,0.5], size=(randommaskb.shape[0]))[:kb]
    _, lam, _ = linalg.svd(tempB.reshape(s, s), full_matrices=True)
    tempA *= rdsb / lam[0]
    B = np.pad(tempB.reshape(s, s), ((0, p-s), (0, p-s)), 'constant')

    x = np.zeros([N, p])
    x_t = sample_x(lambda_s, V, p, heavytail)

    lambda_t = lambda_s
    x[0] = x_t
    for i in range(1, N):
        # print(lambda_t[:s])
        lambda_t = get_next_lambda_abs(x_t, lambda_t, lambda_s, A, B, Vp, p)
        x_t = sample_x(lambda_t, V, p, heavytail)
        x[i] = x_t
    return x


def whitenoise_gen(p, N, heavytail):
    x = np.zeros([N,p])
    lambd = np.ones(p)
    V = np.ones(p)
    for i in range(N):
        x[i] = sample_x(lambd, V, p, heavytail)
    return x
