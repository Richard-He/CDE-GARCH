"""
Miss-specified Data Generation, where higher order terms are included.
"""
import numpy
import numpy as np
import numpy.random as rdm
import scipy.stats as stats
import argparse
import logging
from functools import partial
from time import gmtime, strftime
import jax.numpy as jnp
import numpy.linalg as linalg
import jax
import pandas as pd
import seaborn as sns
from jax import grad, jit, vmap
from cgprox import fmin_cgprox
from utils import get_next_lambda,v_lambda,\
                 calc_xh, sample_x, sample_x_from_epsilon, studentize,\
                get_next_estimated_lambda



parser = argparse.ArgumentParser(description='Parsing Input before generating synthetic data')
parser.add_argument('--s', type=int, default=2,
                    help='dimension of the ')
parser.add_argument('--h','--heavytail', action='store_true',
                    help='Using Heavy Tailed white noise')
parser.set_defaults(h=False)
parser.add_argument('--ra','--rda', type=float, default=0.05,
                    help='approximate spectral radius of A ')
parser.add_argument('--rb','--rdb', type=float, default=0.05,
                    help='approximate spectral radius of B ')
parser.add_argument('--rc','--epsilon', type=float, default=0.1,
                    help='approximate spectral radius of C')
parser.add_argument('--k','--kappa', type=float, default=1,
                    help='rate between the last eigenvalue in the dynamic region versus the static region')
parser.add_argument('--ka', type=int, default=10, help='sparsity of A')
parser.add_argument('--kb', type=int, default=10, help='sparsity of B')
parser.add_argument('--kc', type=int, default=10, help='sparsity of C')
parser.add_argument('--d', '--data', type=str, default='data/',
                    help='data path')
parser.add_argument('--re', '--results', type=str, default='results/',
                    help='results path')
parser.add_argument('--c','--convex', action='store_true',
                    help='Using convex loss or not')
parser.set_defaults(c=False)
parser.add_argument('--rtol', type=float, default=1e-6,
                    help='related tolerance')
parser.add_argument('--z','--zeta', type=float, default=1e-1,
                    help='regularization hyperparameter')
parser.add_argument('--a','--alpha',type=float, default=2.5,
                    help='SCAD hyperparameter')
parser.add_argument('--l','--logging',type=str,default='log/',
                    help='logging path')
parser.add_argument('--p', action='store_true',
                    help='Do we use parametric Bootstrap or not ?')
parser.add_argument('--rs', type=int, default=20,
                    help='What is the resample size ?')
args = vars(parser.parse_args())

# Initialize Parameters
ps = np.array([5])
Nops = np.array([2])
M = args['rs']
Parametric = args['p']
# ps = np.array([64, 256, 1024])
# Nops = np.array([0.5, 0.75, 1, 2, 4])
s_e = args['s']
kappa = args['k']
ka = args['ka']
kb = args['kb']
kc = args['kc']
rdsa = args['ra']
rdsb = args['rb']
rdsc = args['rc']
respath = args['re']
zeta = args['z']
parametric = args['p']

if args['h']:
    ht = 'ht'
    heavytail = True
else:
    ht = ''
    heavytail = False
# Functions

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=args['l']+strftime("%Y-%m-%d %H:%M:%S", gmtime())+f'Mis-specify_{ht}_misspecify' + f'ra={rdsa}' +
                             f'ra={rdsb}'+f'parametric_{parametric}'+'.log',format='%(asctime)s %(message)s',
                     level=logging.DEBUG)

class Loss_Convex(object):
    def __init__(self, s_e, lambda_e, x_h, zeta):
        self.s_e = s_e
        self.lambda_e = lambda_e
        self.x_h = x_h
        self.N = x_h.shape[0]
        self.p = x_h.shape[1]
        self.zeta = zeta


    @partial(jit, static_argnums=(0,))
    def loss_convex_reg(self, param):
        """
        F with convex regularizer
            :param x_h: Nx1xp permuted observations
            :param param: vectorized A and B
            :param s_e: estimated s
            :param lambda_e: estimated marginal lambda
            :return: estimated lambda
            """
        flat_a = param[:s_e * self.p]
        flat_b = param[s_e * self.p:]
        A_es = flat_a.reshape(s_e, self.p)
        B_es = flat_b.reshape(s_e, self.p)
        loss = jnp.sum(jnp.log(self.lambda_e[:s_e]) + self.x_h[0, :s_e] ** 2 / self.lambda_e[:s_e]) / self.N
        lambda_eiminus = self.lambda_e[:s_e]
        for i in range(1, self.N):
            ximinus = self.x_h[i - 1]
            lambda_ei = self.lambda_e[:s_e] - (A_es + B_es)[:, :s_e] @ self.lambda_e[:s_e] + A_es @ (ximinus ** 2) + B_es[:, :s_e] @ lambda_eiminus
            loss = loss + jnp.sum(jnp.log(lambda_ei)) + jnp.sum(self.x_h[i, :s_e] ** 2 / lambda_ei) / self.N
            lambda_eiminus = lambda_ei
        return loss


    def g_prox(self, param, stepsize):
        alphazeta = stepsize * self.zeta
        param[param >= alphazeta] -= alphazeta
        param[np.abs(param) < alphazeta] = 0
        param[param < -alphazeta] += alphazeta
        return param


def get_next_lambda_mis(x_t, lambda_t, lambda_s, A, B, C, Vp, p, sigma_22 = 3):
    lambda_tp = (np.eye(p) - A - B) @ lambda_s + A @ ((Vp @ x_t) ** 2) + B @ lambda_t + C @ ((Vp @ x_t)**4) \
                - C @ (lambda_s**2) * sigma_22
    mask = (lambda_tp <= 0) + np.isnan(lambda_tp) + np.isinf(lambda_tp)
    lambda_tp[mask] = lambda_s[mask]
    return lambda_tp


class Kernel(object):
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth

    def bartlett(self, s):
        if np.abs(s) <= self.bandwidth:
            return 1 - np.abs(s/self.bandwidth)
        else:
            return 0


def data_gen_mis(p, N, s, kappa):
    lambda_s = np.concatenate((np.flip(np.arange(1, s + 1)) * kappa, np.zeros(p - s) / p))
    V = stats.ortho_group.rvs(dim=p)
    Vp = V.T

    # Preprocess A
    tempA = np.zeros(s*p)

    # print(U.shape, S_clip.shape, Vh.shape)
    tempA = np.zeros(s * s)

    randommaska = rdm.permutation(s * s)[:ka]
    tempA[randommaska] = np.random.choice(a=[1], size=(randommaska.shape[0])) / np.sqrt(ka) * rdsa
    A = np.pad(tempA.reshape(s, s), ((0, p - s), (0, p - s)), 'constant')
    # Preprocess B
    tempB = np.zeros(s**2)
    randommaskb = rdm.permutation(s**2)[:kb]
    tempB[randommaskb] = np.random.choice(a=[1], size=(randommaskb.shape[0])) / np.sqrt(kb) * rdsb
    B = np.pad(tempB.reshape(s, s), ((0, p-s), (0, p-s)), 'constant')

    tempC = np.zeros(s * s)
    randommaskc = rdm.permutation(s*s)[:kc]
    tempC[randommaskc] = np.random.choice(a=[1], size=(randommaskc.shape[0])) / np.sqrt(kc) * rdsc
    C = np.pad(tempB.reshape(s, s), ((0, p-s), (0, p-s)), 'constant')
    x = np.zeros([N, p], dtype=numpy.float64)
    lambda_ts = np.zeros([N, p])
    x_t = sample_x(lambda_s, V, p, heavytail)
    lambda_t = lambda_s
    x[0] = x_t
    for i in range(1, N):
        lambda_t = get_next_lambda_mis(x_t, lambda_t, lambda_s, A, B, C, Vp, p)
        try:
            lambda_ts[i-1] = lambda_t
        except BaseException as e:
            print(lambda_ts[i-1], lambda_t)
            print('Error' + str(e))
            exit()
        x_t = sample_x(lambda_t, V, p, heavytail)
        x[i] = x_t
    lambda_t = get_next_lambda_mis(x_t, lambda_t, lambda_s, A, B, C, Vp, p)
    lambda_ts[N-1] = lambda_t
    return x, A, B, C, lambda_ts


def estimate_white_noise(x_h, A_se, B_se, lambda_e, N, s_e):
    """

    :param x_h: permuted X
    :param A_se: estimated A
    :param B_se: estimated B
    :param lambda_se: estimated lambda
    :param N: sample size: T
    :param s_e: estimated s
    :return: epsilons: Nxs vector containing estimated innovation
    """
    lambda_se = lambda_e[:s_e]
    lambda_t = lambda_se
    epsilons = np.zeros([N, s_e])
    epsilons[0] = x_h[0, :s_e] / np.sqrt(lambda_t)
    lambda_tm = lambda_t
    for i in range(1, s_e):
        xminus = x_h[i - 1]
        lambda_t = lambda_se - (A_se + B_se)[:, s_e] @ lambda_se + A_se @ (xminus**2) \
                    + B_se[ :, s_e] @ lambda_tm
        lambda_tm = lambda_t
        epsilons[i] = x_h[i, :s_e] / np.sqrt(lambda_t)
    return epsilons


def kth_diag_indices(dim, k):
    a = np.eye(dim)
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def parametric_bootstrap(x_h, A_se, B_se, lambda_e, V_e, s_e, M):
    N = x_h.shape[0]
    p = x_h.shape[1]
    x = np.zeros([M, N+1, p])
    epsilons = estimate_white_noise(x_h, A_se, B_se, lambda_e, N, s_e)
    epsilons = studentize(epsilons)
    for j in range(M):
        x[j, 0] = sample_x_from_epsilon(lambda_e, V_e, epsilons, p)
        x_t = x[j, 0]
        lambda_t = lambda_e
        for i in range(1, N):
            lambda_t = get_next_lambda(x_t, lambda_t, lambda_e, A_se, B_se, V_e, p)
            x_t = sample_x_from_epsilon(lambda_t, V_e, epsilons, p)
            x[j, i] = x_t
    return x


def generate_W(kernel, N):
    Sigma = np.zeros([N, N])
    for i in range(1, N):
        i_val = kernel.bartlett(i)
        Sigma[kth_diag_indices(N, i)] = i_val
        Sigma[kth_diag_indices(N, -i)] = i_val
    U, l, V = linalg.svd(Sigma)
    half_l = np.sqrt(l)
    half_Sigma = U @ np.diag(half_l) @ V
    Z_n = rdm.randn(N)
    return half_Sigma @ Z_n


def wild_bootstrap(x, M, bandwidth=5):
    N = x.shape[0]
    p = x.shape[1]
    x_star = np.zeros([N, p])
    x_bar = np.mean(x, axis=0)
    kernel = Kernel(bandwidth=bandwidth)
    x_stars = np.zeros([M, p])
    for j in range(M):
        w = generate_W(kernel, N)
        for i in range(N):
            x_stars[j, i] = x_bar + (x[i] - x_bar) * w[i]
        return x_star


def get_estimate(x, s_e, zeta):
    p = x.shape[1]
    V_e, lambda_e = v_lambda(x)
    x_h = calc_xh(x, V_e)
    loss = Loss_Convex(s_e, lambda_e, x_h, zeta=zeta)
    loss_prime = grad(loss.loss_convex_reg)
    result = fmin_cgprox(f=loss.loss_convex_reg, f_prime=loss_prime, g_prox=loss.g_prox,
                         x0=np.zeros(s_e * p * 2), verbose=2, rtol=args['rtol'])
    param = result.x
    flat_a = param[:s_e * p]
    flat_b = param[s_e * p:]
    A_es = flat_a.reshape(s_e, p)
    B_es = flat_b.reshape(s_e, p)
    return V_e, lambda_e, A_es, B_es, x_h


def get_lambda_T(x_h, A, B, lambda_s, T):
    """
    Get the lambda_T
    :param x_h:
    :param A_s:
    :param B_s:
    :param lambda_s:
    :param T:
    :return:
    """
    lambda_t = lambda_s
    for i in range(1, T+1):
        lambda_t = get_next_estimated_lambda(x_h[i-1], lambda_t, lambda_s, A, B)
    return lambda_t


def get_quantile(lambda_s, w):
    vals = lambda_s @ w.T
    qs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    quantiles = np.quantile(vals, qs)
    return quantiles


sns.set_theme(style="darkgrid")

for p in ps:
    for nop in Nops:
        N = nop * p
        if args['p']:
            x, A, B, C, lambda_ts = data_gen_mis(p, N, s_e, kappa)
            esti_quantiles = get_quantile(lambda_ts, w=np.ones(p))
            V_e, lambda_e, A_es, B_es, x_h = get_estimate(x, s_e, zeta=zeta)
            pb_x = parametric_bootstrap(x_h, A_es, B_es, lambda_e, V_e, s_e, M)
            lambda_ts = np.zeros([M, p])
            for i in range(M):
                V_e1, lambda_e1, A_es1, B_es1, x_h1 = get_estimate(pb_x[i], s_e, zeta=zeta)
                lambda_ts[i] = get_lambda_T(x_h1, A_es1, B_es1, lambda_e1, T=N)
            bootstrap_quantile = get_quantile(lambda_ts, w=np.ones(p))
        else:
            x, A, B, C, lambda_ts = data_gen_mis(p, N, s_e, kappa)
            esti_quantiles = get_quantile(lambda_ts, w=np.ones(p))
            wd_x = wild_bootstrap(x, p)
            lambda_ts = np.zeros([M, p])
            for i in range(M):
                V_e1, lambda_e1, A_es1, B_es1, x_h1 = get_estimate(wd_x[i], s_e, zeta=zeta)
                lambda_ts[i] = get_lambda_T(x_h1, A_es1, B_es1, lambda_e1, T=N)
            bootstrap_quantile = get_quantile(lambda_ts, w=np.ones(p))
        np.save(respath + f"Parametric_{args['p']}_p={p}_N={N}_M={M}_bootstrap_quan.npy", bootstrap_quantile)
        np.save(respath + f"Parametric_{args['p']}_p={p}_N={N}_M={M}_estimate_quan.npy", esti_quantiles)
        q_index = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 2)
        type = np.array(['bootstrapped'] * 9 + ['estimated'] * 9)
        quantiles = np.concatenate((bootstrap_quantile, esti_quantiles))
        df1 = pd.DataFrame(dict(type=type, quantiles=quantiles, x=q_index))
        qtplot = sns.relplot(x='x', y='quantiles', kind='line', hue='type', data=df1)
        fig = qtplot.fig
        fig.savefig(respath + f"Parametric_{args['p']}_p={p}_N={N}_M={M}_bootstrap.png")







