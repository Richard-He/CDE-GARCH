import numpy
import numpy as np
import numpy.random as rdm
import scipy.stats as stats
import scipy.linalg as linalg
import argparse
import logging
import sys
from utils import sample_x, get_next_lambda, get_next_lambda_abs
from time import gmtime, strftime



parser = argparse.ArgumentParser(description='Parsing Input before generating synthetic data')
parser.add_argument('--s', type=int, default=5,
                    help='dimension of the ')
parser.add_argument('--h','--heavytail', action='store_true',
                    help='Using Heavy Tailed white noise')
parser.set_defaults(h=False)
parser.add_argument('--ra','--rda', type=float, default=0.2,
                    help='approximate spectral radius of A ')
parser.add_argument('--rb','--rdb', type=float, default=0.2,
                    help='approximate spectral radius of B ')
parser.add_argument('--k','--kappa', type=float, default=1,
                    help='rate between the last eigenvalue in the dynamic region versus the static region')
parser.add_argument('--ka', type=int, default=15, help='sparsity of A')
parser.add_argument('--kb', type=int, default=15, help='sparsity of B')
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
parser.add_argument('--t', '--times',type=int, default=20, help='how many datasets do we generate ?')
args = vars(parser.parse_args())

# Initialize Parameters
ps = np.array([64, 128, 256, 512])
Nops = np.array([0.25, 0.5, 1, 2, 4,  6,  8,  10])
# ps = np.array([64])
# Nops = np.array([0.5])
s = args['s']
rdsa = args['ra']
rdsb = args['rb']
kappa = args['k']
ka = args['ka']
kb = args['kb']
times = args['t']

if args['h']:
    ht = 'ht'
    heavytail = True
else:
    ht = ''
    heavytail = False
# Functions

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=args['l']+strftime("%Y-%m-%d %H:%M:%S", gmtime())+f'data_generate_{ht}'+f'ra={rdsa}'+f'ra={rdsb}'+'.log',format='%(asctime)s %(message)s',
                     level=logging.DEBUG)


def data_gen_t(p, N, s, rdsa, rdsb, kappa, ka, kb, heavytail, cur_time):
    lambda_s = np.concatenate((np.flip(np.arange(1, s + 1, dtype=float)) * kappa, np.zeros(p - s)))
    V = stats.ortho_group.rvs(dim=p)
    Vp = V.T

    # Preprocess A
    tempA = np.zeros(s * s)

    randommaska = rdm.permutation(s*s)[:ka]
    tempA[randommaska] = np.random.choice(a=[1], size=(randommaska.shape[0])) / np.sqrt(ka) * rdsa
    A = np.pad(tempA.reshape(s, s), ((0, p-s), (0, p-s)), 'constant')
    tempB = np.zeros(s * s)

    randommaska = rdm.permutation(s*s)[:ka]
    tempA[randommaska] = np.random.choice(a=[1], size=(randommaska.shape[0])) / np.sqrt(kb) * rdsb
    B = np.pad(tempA.reshape(s, s), ((0, p-s), (0, p-s)), 'constant')


    # tempA = np.zeros(s ** 2)
    # randommaska = rdm.permutation(s ** 2)[:ka]
    # tempA[randommaska] = np.random.choice(a=[1, 0.9, 0.8, 0.7, 0.6, 0.5], size=(randommaska.shape[0]))[:ka]
    # _, lam, _ = linalg.svd(tempA.reshape(s, s), full_matrices=True)
    # tempA *= rdsa / lam[0]
    # A = np.pad(tempA.reshape(s, s), ((0, p - s), (0, p - s)), 'constant')
    # Preprocess B
    # tempB = np.zeros(s ** 2)
    # randommaskb = rdm.permutation(s ** 2)[:kb]
    # tempB[randommaskb] = np.random.choice(a=[1, 0.9, 0.8, 0.7, 0.6, 0.5], size=(randommaskb.shape[0]))[:kb]
    # _, lam, _ = linalg.svd(tempB.reshape(s, s), full_matrices=True)
    # tempA *= rdsb / lam[0]
    # B = np.pad(tempB.reshape(s, s), ((0, p - s), (0, p - s)), 'constant')

    x = np.zeros([N, p], dtype=numpy.float64)
    x_t = sample_x(lambda_s, V, p, heavytail)

    lambda_t = lambda_s
    x[0] = x_t
    for i in range(1, N):
        # print(lambda_t[:s])
        lambda_t = get_next_lambda(x_t, lambda_t, lambda_s, A, B, Vp, p)
        x_t = sample_x(lambda_t, V, p, heavytail)
        if np.isnan(x_t).any() or np.isposinf(x_t).any():
            logging.info(f'Error Occured at p={p}, N={N},lambda_t={lambda_t}, lambda_s={lambda_s}, A={A}, B={B}')
            sys.exit()
        x[i] = x_t
    logging.debug(f"Success,X_p={p}_N={N}"+ht)
    np.save(f"data/X_p={p}_N={N}"+ht+f"r={rdsa}_{cur_time}.npy", x)
    np.save(f"data/V_p={p}_N={N}"+ht+f"r={rdsa}_{cur_time}.npy", V)
    np.save(f"data/A_p={p}_N={N}"+ht+f"r={rdsa}_{cur_time}.npy", A)
    np.save(f"data/B_p={p}_N={N}"+ht+f"r={rdsa}_{cur_time}.npy", B)
    np.save(f"data/lambda_s_p={p}_N={N}"+ht+f"r={rdsa}_{cur_time}.npy", lambda_s)


def start():
    for t in range(times):
        for p in ps:
            for Nop in Nops:
                N = int(np.ceil(p * Nop))
                data_gen_t(p, N, s, rdsa, rdsb, kappa, ka, kb, heavytail, t)

start()
