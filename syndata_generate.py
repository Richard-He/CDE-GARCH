import numpy as np
import numpy.random as rdm
import scipy.stats as stats
import scipy.linalg as linalg

# Initialize Parameters
ps = [64,]
Nops = [1,2,3]
s = 10
rds = 0.5
kappa = 2
ka = 60
kb = 60

heavy_tail = True
if heavy_tail:
    ht = 'ht'
else:
    ht = ''
# Functions

def sample_x(lambd, V, p):
    if heavy_tail == True:
        epsilon_t = stats.t.rvs(2.4, size=p)
        #print(epsilon_t.shape)
        var = 2.4 / (2.4-2)
        epsilon_t *= 1/np.sqrt(var)
    else:
        epsilon_t = rdm.randn(p)
    # print(epsilon_t.dtype)
    if (lambd < 0).any() == True:
        exit()
    Lambd = np.diag(np.sqrt(lambd))
    #print(lambd.shape)
    x = V @ Lambd @ epsilon_t
    #print(x.shape)
    return x


def get_next_lambda(x_t, lambda_t, lambda_s, A, B, Vp, p):
    #print(x_t.shape, lambda_s.shape, lambda_t.shape, A.shape, B.shape, Vp.shape, p)
    lambda_tp = (np.eye(p) - A - B) @ lambda_s + A @ ((Vp @ x_t) ** 2) + B @ lambda_t
    return lambda_tp


def data_gen(p, N, s, rds, kappa):
    lambda_s = np.concatenate((np.flip(np.arange(1, s + 1)) * kappa, np.ones(p - s) / p/100))
    V = stats.ortho_group.rvs(dim=p)
    Vp = V.T

    # Preprocess A
    tempA = np.zeros(s*p)

    # print(U.shape, S_clip.shape, Vh.shape)
    randommaska = rdm.permutation(s*p)[:ka]
    tempA[randommaska] = np.random.choice(a=[-1, 1], size=(randommaska.shape[0])) * 1/ka * rds

    A = np.pad(tempA.reshape(s, p), ((0, p-s), (0, 0)), 'constant')
    # Preprocess B
    tempB = np.zeros(s*p)
    randommaskb = rdm.permutation(s*p)[:kb]
    tempB[randommaskb] = np.random.choice(a=[-1, 1], size=(randommaskb.shape[0])) * 1/kb * rds
    B = np.pad(tempB.reshape(s, p), ((0, p-s), (0, 0)), 'constant')

    x = np.zeros([N, p])
    x_t = sample_x(lambda_s, V, p)
    lambda_t = lambda_s
    x[0] = x_t
    for i in range(1, N):
        lambda_t = get_next_lambda(x_t, lambda_t, lambda_s, A, B, Vp, p)
        x_t = sample_x(lambda_t, V, p)
        x[i] = x_t
    np.save(f"data/X_p={p}_N={N}"+ht+".npy", x)
    np.save(f"data/V_p={p}_N={N}"+ht+".npy", V)
    np.save(f"data/A_p={p}_N={N}"+ht+".npy", A)
    np.save(f"data/B_p={p}_N={N}"+ht+".npy", B)
    np.save(f"data/lambda_s_p={p}_N={N}"+ht+".npy", lambda_s)


def start():
    for Nop in Nops:
        for p in ps:
            N = p * Nop
            data_gen(p, N, s, rds, kappa)

start()

def data_gen_test(p, N, s, rds, kappa):
    lambda_s = np.concatenate((np.flip(np.arange(1, s + 1)) * kappa, np.ones(p - s) / p/10))
    V = stats.ortho_group.rvs(dim=p)
    Vp = V.T

    # Preprocess A
    tempA = np.zeros(s*p)

    # print(U.shape, S_clip.shape, Vh.shape)
    randommaska = rdm.permutation(s*p)[:ka]
    tempA[randommaska] = 1/ka * rds

    A = np.pad(tempA.reshape(s, p), ((0, p-s), (0, 0)), 'constant')
    # Preprocess B
    tempB = np.zeros(s*p)
    randommaskb = rdm.permutation(s*p)[:kb]
    tempB[randommaskb] = 1/kb * rds
    B = np.pad(tempB.reshape(s, p), ((0, p-s), (0, 0)), 'constant')

    x = np.zeros([N, p])
    x_t = sample_x(lambda_s, V, p)
    lambda_t = lambda_s
    x[0] = x_t
    for i in range(1, N):
        lambda_t = get_next_lambda(x_t, lambda_t, lambda_s, A, B, Vp, p)
        x_t = sample_x(lambda_t, V, p)
        x[i] = x_t
    return x
