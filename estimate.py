from cgprox import fmin_cgprox
import numpy as np
from functools import partial
import jax.numpy as jnp
import jax
from utils import v_lambda
from jax import grad, jit, vmap
import scipy.linalg as linalg
from time import gmtime, strftime
import logging
import argparse

parser = argparse.ArgumentParser(description='Parsing Input before estimation')
parser.add_argument('--s', type=int, default=5,
                    help='dimension of the ')
parser.add_argument('--h','--heavytail', action='store_true',
                    help='Using Heavy Tailed white noise')
parser.set_defaults(h=False)
parser.add_argument('--rtol', type=float, default=1e-7,
                    help='approximate spectral radius of A and B')
parser.add_argument('--k','--kappa', type=float, default=10,
                    help='rate between the last eigenvalue in the dynamic region versus the static region')
parser.add_argument('--ka', type=int, default=10, help='sparsity of A')
parser.add_argument('--kb', type=int, default=10, help='sparsity of B')
parser.add_argument('--d', '--data', type=str, default='data/',
                    help='data path')
parser.add_argument('--re', '--results', type=str, default='results/',
                    help='results path')
parser.add_argument('--c','--convex', action='store_true',
                    help='Using convex loss or not')
parser.set_defaults(c=False)
parser.add_argument('--z','--zeta', type=float, default=1,
                    help='regularization hyperparameter')
parser.add_argument('--a','--alpha',type=float, default=2.5,
                    help='SCAD hyperparameter')
parser.add_argument('--l','--logging',type=str,default='log/',
                    help='logging path')
args = vars(parser.parse_args())
# Loss function evaluation

ps = np.array([64, 128, 256, 512, 1024])
Nops = np.array([0.25, 0.5, 1, 2, 4, 8])
s_e = args['s']
zeta = args['z']
alpha = args['a']
convex = args['c']
path = args['d']
respath = args['re']
heavyt = args['h']
if heavyt is True:
    ht = '_heavy_tail'
    htn = 'ht'
else:
    ht = ''
    htn = ''
if convex is True:
    conv = '_convex_'
else:
    conv = ''


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=args['l']+strftime("%Y-%m-%d %H:%M:%S", gmtime())+f'_log_convex_{convex}_heavy_tail_{heavyt}'+f'z={zeta}'
                             +f'alpha={alpha}'+'.log',format='%(asctime)s %(message)s',
                     level=logging.INFO)
# print('finished')


# Calculation of estimate of lambda
def loss_nonconvex_reg(param):
    """
    F with nonconvex regularizer
    :param x_h: Nx1xp permuted observations
    :param param: vectorized A and B
    :param s_e: estimated s
    :param lambda_e: estimated marginal lambda
    :return: estimated lambda
    """
    flat_a = param[:s_e * p]
    flat_b = param[s_e * p:]
    A_es = flat_a.reshape(s_e, p)
    B_es = flat_b.reshape(s_e, p)
    loss = jnp.sum(jnp.log(lambda_e[:s_e]) + x_h[0, :s_e] ** 2 / lambda_e[:s_e])
    lambda_eiminus = lambda_e[:s_e]
    for i in range(1, N):
        ximinus = x_h[i - 1]
        lambda_ei = lambda_e[:s_e] - (A_es + B_es)[:, :s_e] @ lambda_e[:s_e] + A_es @ (ximinus ** 2) + B_es[:, :s_e] @ lambda_eiminus
        loss = loss + jnp.sum(jnp.log(lambda_ei) + x_h[i, :s_e] ** 2 / lambda_ei)
        lambda_eiminus = lambda_ei
    loss = loss / N + rho(param)
    return loss


@jax.jit
def loss_convex_reg(param):
    """
    F with convex regularizer
        :param x_h: Nx1xp permuted observations
        :param param: vectorized A and B
        :param s_e: estimated s
        :param lambda_e: estimated marginal lambda
        :return: estimated lambda
        """
    flat_a = param[:s_e * p]
    flat_b = param[s_e * p:]
    A_es = flat_a.reshape(s_e, p)
    B_es = flat_b.reshape(s_e, p)
    loss = jnp.sum(jnp.log(lambda_e[:s_e]) + x_h[0, :s_e] ** 2 / lambda_e[:s_e]) / N
    lambda_eiminus = lambda_e[:s_e]
    for i in range(1, N):
        ximinus = x_h[i - 1]
        lambda_ei = lambda_e[:s_e] - (A_es + B_es)[:, :s_e] @ lambda_e[:s_e] + A_es @ (ximinus ** 2) + B_es[:, :s_e] @ lambda_eiminus
        loss = loss + jnp.sum(jnp.log(lambda_ei)) + jnp.sum(x_h[i, :s_e] ** 2 / lambda_ei) / N
        lambda_eiminus = lambda_ei
    return loss

# The smooth part of model selection loss:
def rho(param):
    # def pointwise_scad(t):
    #     if jnp.abs(t) <= zeta:
    #         return 0
    #     elif jnp.abs(t) <= alpha * zeta and jnp.abs(t) > zeta:
    #         return -(t**2 - 2*zeta*jnp.abs(t)+zeta**2)/(2*alpha-2)
    #     else:
    #         return (alpha+1)*zeta**2/2 - zeta * jnp.abs(t)
    loss = 0
    mid = param[np.logical_and(zeta < jnp.abs(param), jnp.abs(param) <= zeta * alpha)]
    loss = loss + jnp.sum(-mid**2 - 2*zeta*abs(mid)+zeta**2)/(2*(alpha-1))
    out = param[jnp.abs(param) > alpha*zeta]
    loss = loss + jnp.sum((alpha+1)*zeta**2/2-abs(out)*zeta)
    return loss


def calc_xh(x, V_e):
    x_h = np.zeros((N, p))
    for i in range(N):
        x_h[i] = V_e.T @ x[i]
    return x_h


# Soft thresholding
def g_prox(param, stepsize):
    alphazeta = stepsize * zeta
    param[param >= alphazeta] -= alphazeta
    param[np.abs(param) < alphazeta] = 0
    param[param < -alphazeta] += alphazeta
    return param


def evaluate(param, true_param):
    l2error = linalg.norm(param - true_param, 2)
    fdr = np.sum(true_param[param != 0] == 0) / np.sum(param != 0)
    return l2error, fdr

# parameters for the SCAD loss function


# Start:
l2errors = np.zeros([ps.shape[0], Nops.shape[0]])
fdrs = np.zeros([ps.shape[0], Nops.shape[0]])
V_errs = np.zeros([ps.shape[0], Nops.shape[0]])
lambda_errs = np.zeros([ps.shape[0], Nops.shape[0]])
for i1 in range(ps.shape[0]):
    i = ps[i1]
    for i2 in range(Nops.shape[0]):
        j = Nops[i2]
        p = i
        N = int(p * j)
        x = np.load(path+f"X_p={p}_N={N}"+htn+".npy")
        A_true = np.load(path+f"A_p={p}_N={N}"+htn+".npy")
        B_true = np.load(path+f"B_p={p}_N={N}"+htn+".npy")
        param_true = np.concatenate((A_true.reshape(-1), B_true.reshape(-1)))
        V_true = np.load(path+f"V_p={p}_N={N}"+htn+".npy")
        V_e, lambda_e = v_lambda(x)
        V_error = linalg.norm(V_e[:s_e,:].reshape(-1) - V_true[:s_e,:].reshape(-1), 2)
        lambda_true = np.load(path+f"lambda_s_p={p}_N={N}"+htn+".npy")

        lambda_err = linalg.norm(lambda_e - lambda_true, 2)

        #s_e = binary_search_s_gaussian(x, V_e, q=5, lambda_e=lambda_e)
        x_h = calc_xh(x, V_e)
        if convex == False:
            loss_prime = grad(loss_nonconvex_reg)
            result = fmin_cgprox(f=loss_nonconvex_reg, f_prime=loss_prime, g_prox=g_prox, x0=np.zeros(s_e * p * 2),
                                 verbose=2,
                                 rtol=args['rtol'])
        else:
            loss_prime = grad(loss_convex_reg)
            result = fmin_cgprox(f=loss_convex_reg, f_prime=loss_prime, g_prox=g_prox, x0=np.zeros(s_e * p * 2),
                                 verbose=2,
                                 rtol=args['rtol'])
        if result.success:
            param = result.x
            flat_a = param[:s_e * p]
            flat_b = param[s_e * p:]
            A_es = flat_a.reshape(s_e, p)
            B_es = flat_b.reshape(s_e, p)
            A_final = np.pad(A_es, ((0, p - s_e), (0, 0)), 'constant').reshape(-1)
            B_final = np.pad(B_es, ((0, p - s_e), (0, 0)), 'constant').reshape(-1)
            param = np.concatenate((A_final, B_final))
            l2error, fdr = evaluate(param, param_true)
            fdrs[i1, i2] = fdr
            l2errors[i1, i2] = l2error
            V_errs[i1, i2] = V_error
            lambda_errs[i1, i2] = lambda_err
            logging.info(f"lambda_err {lambda_err}")
            logging.info(f"V_err {V_error}")
            logging.info(f"l2_error {l2error}")
            logging.info(f"fdr {fdr}")
            logging.info(f"nonzeros {(param!=0).nonzero()}")
            logging.info(f"nonzeros {(param_true!=0).nonzero()}")
        else:
            logging.info(f"param difference {param - param_true}")
            logging.info(f"V difference {V_e - V_true}")
            logging.info(f"lambda_diff {lambda_true - lambda_e}")
            exit()
np.save(respath + f"l2errors" + ht + conv + f'z={zeta}'+".npy", l2errors)
np.save(respath + f"fdrs" + ht + conv + f'z={zeta}'+ ".npy", fdrs)
np.save(respath + f"lambda_errors" + ht + conv + f'z={zeta}'+".npy", lambda_errs)
np.save(respath + f"V_errs" + ht + conv + f'z={zeta}'+".npy", V_errs)

# for i1 in range(ps.shape[0]):
#     i = ps[i1]
#     for i2 in range(Nops.shape[0]):
#         j = Nops[i2]
#         p = i
#         N = p * j
#         x = np.load(f"X_p={p}_N={N}.npy")
#         A_true = np.load(f"A_p={p}_N={N}.npy")
#         B_true = np.load(f"B_p={p}_N={N}.npy")
#         param_true = np.concatenate((A_true.reshape(-1), B_true.reshape(-1)))
#         V_true = np.load(f"V_p={p}_N={N}.npy")
#         V_e, lambda_e = v_lambda(x)
#         V_error = linalg.norm(V_e.reshape(-1) - V_true.reshape(-1), 2)
#         logging.info("V_err", V_error)
#         lambda_true = np.load(f"lambda_s_p={p}_N={N}.npy")
#
#         lambda_err = linalg.norm(lambda_e - lambda_true, 2)
#         logging.info("lambda_err", lambda_err)
#         s_e = binary_search_s_gaussian(x, V_e, q=5, lambda_e=lambda_e)
#         logging.info("s_e", s_e)
#         x_h = calc_xh(x, V_e)
#         if convex == False:
#             loss_prime = grad(loss_nonconvex_reg)
#             result = fmin_cgprox(f=loss_nonconvex_reg, f_prime=loss_prime, g_prox=g_prox, x0=np.zeros(s_e * p * 2), verbose=2 )
#         else:
#             loss_prime = jit(grad(loss_convex_reg))
#             result = fmin_cgprox(f=loss_convex_reg, f_prime=loss_prime, g_prox=g_prox, x0=np.zeros(s_e * p * 2), verbose=2)
#         if result.success:
#             l2error, fdr = evaluate(result.x, param_true)
#             fdrs[i1, i2] = fdr
#             l2errors[i1, i2] = l2error
#             V_errs[i1, i2] = V_error
#             lambda_errs[i1, i2] = lambda_err
#         else:
#             logging.info("param difference", result.x - param_true)
#             logging.info("V difference", V_e - V_true)
#             logging.info("lambda_diff", lambda_true - lambda_e)
#             exit()
# np.save(f"l2errors.npy", l2errors)
# np.save(f"fdrs.npy", fdrs)
# np.save(f"lambda_errors.npy", lambda_errs)
# np.save(f"V_errs.npy", V_errs)

# for i2 in range(Nops.shape[0]):
#     j = Nops[i2]
#     p = ps[0]
#     N = p * j
#     x = np.load(f"data/X_p={p}_N={N}.npy")
#     A_true = np.load(f"data/A_p={p}_N={N}.npy")
#     B_true = np.load(f"data/B_p={p}_N={N}.npy")
#     param_true = np.concatenate((A_true.reshape(-1), B_true.reshape(-1)))
#     V_true = np.load(f"data/V_p={p}_N={N}.npy")
#     V_e, lambda_e = v_lambda(x)
#     V_error = linalg.norm(V_e.reshape(-1) - V_true.reshape(-1), 2)
#     logging.info("V_err", V_error)
#     lambda_true = np.load(f"data/lambda_s_p={p}_N={N}.npy")
#
#     lambda_err = linalg.norm(lambda_e - lambda_true, 2)
#     logging.info("lambda_err", lambda_err)
#     s_e = binary_search_s_gaussian(x, V_e, q=1, lambda_e=lambda_e)
#     logging.info("Nops", Nops[i2])
#     logging.info("s_e", s_e)