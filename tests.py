import numpy as np
import argparse
import numpy as np
from functools import partial
from jax import grad, jit, vmap
import scipy.linalg as linalg
from scipy import optimize
import pandas as pd
import logging
from time import gmtime, strftime
from utils import data_gen, v_lambda, test_s_gaussian, test_s_ht, whitenoise_gen


parser = argparse.ArgumentParser(description='Parsing Input before estimation')
parser.add_argument('--s', type=int, default=5,
                    help='dimension of the ')
parser.add_argument('--h', '--heavytail', action='store_true',
                    help='Using Heavy Tailed white noise')
parser.set_defaults(h=False)
parser.add_argument('--k', '--kappa', type=float, default=10,
                    help='rate between the last eigenvalue in the dynamic region versus the static region')
parser.add_argument('--ka', type=int, default=10, help='sparsity of A')
parser.add_argument('--kb', type=int, default=10, help='sparsity of B')
parser.add_argument('--ra', '--rda', type=float, default=0.5,
                    help='approximate spectral radius of A ')
parser.add_argument('--rb', '--rdb', type=float, default=0.2,
                    help='approximate spectral radius of B ')
parser.add_argument('--re', '--results', type=str, default='results/',
                    help='results path')
parser.add_argument('--lfile', type=str, default='log/',
                    help='logging path')
parser.add_argument('--q', type=int, default=1)
parser.add_argument('--p', action='store_true', help='Do we test power ? Otherwise we test size.')
parser.set_defaults(p=False)
parser.add_argument('--l','--log', action='store_true', help='Do we use logging ? Otherwise we print')
parser.set_defaults(l=False)
args = vars(parser.parse_args())


ka = args['ka']
kb = args['kb']
s_e = args['s']
rdsa = args['ra']
rdsb = args['rb']
kappa = args['k']
heavytail = args['h']
is_power = args['p']
if is_power:
    ps = np.array([128, 256])
    q = args['q']
else:
    ps = np.array([64, 128, 256, 512])
    qs = np.array([1, 3, 5])
Nops = np.array([0.125, 0.25, 0.5, 1, 2, 4])
Sops = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.25, 0.35, 0.45, 0.5])
# test case
# qs = np.array([1])
# Nops = np.array([0.125])
# Sops = np.array([0.01])
respath = args['re']

times = 1000

if args['l']:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args['lfile']+strftime("%Y-%m-%d %H:%M:%S", gmtime())+
     f'hypotest_rda={rdsa}_rdb={rdsb}_kappa={kappa}_heavytail={heavytail}_q={q}_power={is_power}.log',
                        format='%(asctime)s %(message)s', level=logging.INFO)


def test_power(p, N, s, s_e, rdsa, rdsb, kappa, q, times, ka, kb, heavytail=False):
    corr = 0
    for i in range(times):
        x = data_gen(p, N, s, rdsa, rdsb, kappa, ka ,kb,heavytail)
        V_e, lambda_e = v_lambda(x)
        # print(V_e, lambda_e)
        if not heavytail:
            if not test_s_gaussian(x, V_e, q, s_e):
                corr += 1
        else:
            if not test_s_ht(x, V_e, q, s_e, s):
            # print(f'reject at {i}')
                corr += 1
    return corr/times


def test_size(p, N, q, heavytail, times):
    corr = 0
    for i in range(times):
        x = whitenoise_gen(p, N, heavytail)
        V_e, lambda_e = v_lambda(x)
        if not heavytail:
            if not test_s_gaussian(x, V_e, q, s_e=0):
                corr += 1
        else:
            if not test_s_ht(x, V_e, q, s_e=0, s=p):
                # print(f'reject at {i}')
                corr += 1
    return corr / times


def start_power():
    powers = np.zeros([Sops.shape[0], ps.shape[0], Nops.shape[0]])
    for k in range(ps.shape[0]):
        p = ps[k]
        for i in range(Nops.shape[0]):
            nop = Nops[i]
            for j in range(Sops.shape[0]):
                sop = Sops[j]
                s = np.ceil(sop * p).astype(int)
                N = np.rint(p * nop).astype(int)
                pwr = test_power(p, N, s, 0, rdsa, rdsb, kappa, q, times, ka, kb, heavytail)
                powers[j, k, i] = pwr
                if args['l']:
                    logging.info(f's:{s}, N:{N}, p:{p}, power: {pwr}, heavytail: {heavytail}')
                else:
                    print(f's:{s}, N:{N}, p:{p}, power: {pwr}, heavytail: {heavytail}')
    return powers


def start_size():
    sizes = np.zeros([ps.shape[0], qs.shape[0], Nops.shape[0]])
    for j in range(qs.shape[0]):
        q = qs[j]
        for k in range(ps.shape[0]):
            p = ps[k]
            for i in range(Nops.shape[0]):
                nop = Nops[i]
                N = np.rint(p * nop).astype(int)
                size = test_size(p, N, q, heavytail, times)
                sizes[k,j,i] = size
                if args['l']:
                    logging.info(f'q:{q}, p:{p}, N:{N}, size:{size}, heavytail:{heavytail}')
                else:
                    print(f'q:{q}, p:{p}, N:{N}, size:{size}, heavytail:{heavytail}')
    return sizes


if is_power:
    powers = start_power()
    row_index = Sops.tolist()
    iterables = [ps.tolist(), Nops.tolist()]
    column_index = pd.MultiIndex.from_product(iterables, names=['T/p', 's/p'])
    df = pd.DataFrame(powers.reshape(Sops.shape[0], -1), index=row_index,
                      columns=column_index)
    with open(
            respath + f'hypotest_rda={rdsa}_rdb={rdsb}_kappa={kappa}_heavytail={heavytail}_q={q}_power={is_power}.txt',
              "w") as text_file:
        text_file.write(df.to_latex())
else:
    sizes = start_size()
    iterables = [ps.tolist(), qs.tolist()]
    row_index = pd.MultiIndex.from_product(iterables, names=['q', 'T/p'])
    column_index = Nops.tolist()
    df = pd.DataFrame(sizes.reshape(-1, Nops.shape[0]), index=row_index,
                      columns=column_index)
    with open(
            respath + f'hypotest_rda={rdsa}_rdb={rdsb}_kappa={kappa}_heavytail={heavytail}_power={is_power}.txt',
            "w") as text_file:
        text_file.write(df.to_latex())


# print(test_power(p=64, N=64, s=4, s_e=0, rdsa=0.4, rdsb=0.4, kappa=1, q=1,
#                  times=10, ka=10, kb=10, heavytail=True))
# print(test_size(p=64, N=64, s=4, s_e=4, q=1, heavytail=True, times=100))