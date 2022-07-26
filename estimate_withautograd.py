import warnings
from cgprox import fmin_cgprox
import numpy as np
import jax.numpy as jnp
from tests import v_lambda, binary_search_s_gaussian
from jax import grad, jit, vmap
import scipy.linalg as linalg
from scipy import optimize

# Loss function evaluation

ps = [64]
Nops = np.array([1,2,3,4,5,6,7,8,9,10])
s = 5
rds = 0.15
kappa = 2
convex = True


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
    loss = jnp.sum(jnp.log(lambda_e[:s_e]) + x_h[0, :s_e] ** 2 / lambda_e[:s_e]) / N
    lambda_eiminus = lambda_e[:s_e]
    for i in range(1, N):
        ximinus = x_h[i - 1]
        lambda_ei = lambda_e[:s_e] - (A_es + B_es)[:, :s_e] @ lambda_e[:s_e] + A_es @ (ximinus ** 2) + B_es[:, :s_e] @ lambda_eiminus
        loss = loss + jnp.sum(jnp.log(lambda_ei) + x_h[i, :s_e] ** 2 / lambda_ei) / N
        lambda_eiminus = lambda_ei
    loss = loss + rho(param)
    return loss


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
        loss = loss + jnp.sum(jnp.log(lambda_ei) + x_h[i, :s_e] ** 2 / lambda_ei) / N
        lambda_eiminus = lambda_ei
    return loss

# The smooth part of model selection loss:
def rho(param):
    loss = 0
    mask1 = abs(param) <= alpha * zeta
    mask2 = abs(param) >= zeta
    mask = mask1 * mask2
    for i in range(param.shape[0]):
        t = param[i]
        if mask[i]:
            loss = loss + (- t ** 2 -2 * zeta * jnp.abs(t) + zeta**2) / 2 / (alpha - 1)
        elif not mask[1]:
            loss = loss + (alpha+1) * zeta**2 / 2 - zeta*abs(t)
    return loss


def calc_xh(x, V_e):
    x_h = np.zeros((N, p))
    for i in range(N):
        x_h[i] = V_e.T @ x[i]
    return x_h


# Soft thresholding
def g_prox(param, alpha):
    alphazeta = alpha * zeta
    param[param >= alphazeta] -= alphazeta
    param[np.abs(param) < alphazeta] = 0
    param[param < -alphazeta] += alphazeta
    return param


def evaluate(param, true_param):
    param = np.concatenate((param, np.zeros(true_param.shape[0]-param.shape[0])))
    l2error = linalg.norm(param - true_param, 2)
    fdr =  np.sum(true_param[param != 0] == 0) /  np.sum(param != 0)
    return l2error, fdr

N = 0
p = 0

# parameters for the SCAD loss function
zeta = 0.02
alpha = 2.5

# Start:
l2errors = np.zeros([3, 9])
fdrs = np.zeros([3, 9])
V_errs = np.zeros([3, 9])
lambda_errs = np.zeros([3, 9])
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
#         lambda_true = np.load(f"lambda_s_p={p}_N={N}.npy")
#
#         lambda_err = linalg.norm(lambda_e - lambda_true, 2)
#         s_e = binary_seach_s(x, V_e, q=3, lambda_e=lambda_e)
#         x_h = calc_xh(x, V_e)
#         loss_prime = jit(grad(loss))
#         result = fmin_cgprox(f=loss, f_prime=loss_prime, g_prox=g_prox, x0= np.zeros(s_e * p * 2))
#         if result.success:
#             l2error, fdr = evaluate(result.x, param_true)
#             fdrs[i1, i2] = fdr
#             l2errors[i1, i2] = l2error
#             V_errs[i1, i2] = V_error
#             lambda_errs[i1, i2] = lambda_err
#         else:
#             print("param difference", result.x - param_true)
#             print("V difference", V_e - V_true)
#             print("lambda_diff", lambda_true - lambda_e)
#             exit()
# np.save(f"l2errors.npy", l2errors)
# np.save(f"fdrs.npy", fdrs)
# np.save(f"lambda_errors.npy", lambda_errs)
# np.save(f"V_errs.npy", V_errs)


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
#         print("V_err", V_error)
#         lambda_true = np.load(f"lambda_s_p={p}_N={N}.npy")
#
#         lambda_err = linalg.norm(lambda_e - lambda_true, 2)
#         print("lambda_err", lambda_err)
#         s_e = binary_search_s_gaussian(x, V_e, q=5, lambda_e=lambda_e)
#         print("s_e", s_e)
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
#             print("param difference", result.x - param_true)
#             print("V difference", V_e - V_true)
#             print("lambda_diff", lambda_true - lambda_e)
#             exit()
# np.save(f"l2errors.npy", l2errors)
# np.save(f"fdrs.npy", fdrs)
# np.save(f"lambda_errors.npy", lambda_errs)
# np.save(f"V_errs.npy", V_errs)

for i2 in range(Nops.shape[0]):
    j = Nops[i2]
    p = ps[0]
    N = p * j
    x = np.load(f"data/X_p={p}_N={N}.npy")
    A_true = np.load(f"data/A_p={p}_N={N}.npy")
    B_true = np.load(f"data/B_p={p}_N={N}.npy")
    param_true = np.concatenate((A_true.reshape(-1), B_true.reshape(-1)))
    V_true = np.load(f"data/V_p={p}_N={N}.npy")
    V_e, lambda_e = v_lambda(x)
    V_error = linalg.norm(V_e.reshape(-1) - V_true.reshape(-1), 2)
    print("V_err", V_error)
    lambda_true = np.load(f"data/lambda_s_p={p}_N={N}.npy")

    lambda_err = linalg.norm(lambda_e - lambda_true, 2)
    print("lambda_err", lambda_err)
    s_e = binary_search_s_gaussian(x, V_e, q=1, lambda_e=lambda_e)
    print("Nops", Nops[i2])
    print("s_e", s_e)