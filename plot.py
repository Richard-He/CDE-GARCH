import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

heavy_tail = True



# respath = 'results—fromserver/'
# heavyt = True
# convex = True
# if heavyt is True:
#     ht = '_heavy_tail'
# else:
#     ht = ''
# if convex is True:
#     conv = '_convex_'
# else:
#     conv = ''
# ps = np.array([64, 128, 256, 512, 1024])
# Nops = np.array([0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8])

def drawplot(Nops, ps, respath, zeta, ht, conv, l2errors, fdrs,rds, multi=''):
    if l2errors is None:
        l2errors = np.load(respath + f"l2errors" + ht + conv + "new" + f'zeta={zeta}' + ".npy")
    if fdrs is None:
        fdrs = np.load(respath + f"fdrs" + ht + conv + "new" + f'zeta={zeta}' + ".npy")
    sns.set_theme(style="darkgrid")
    Nopshape = Nops.shape[0]
    psshape = ps.shape[0]
    pss = np.array([p for p in ps for i in range(Nopshape)])
    Nopss = np.array(Nops.tolist() * int(psshape))
    df = pd.DataFrame(dict(npr=Nopss, p=pss, fdr=fdrs.reshape(-1), l2error=l2errors.reshape(-1)))

    l2rplot = sns.relplot(x="npr", y="l2error", hue='p', kind="line", data=df)
    l2rplot.set(xlabel='T/p', ylabel='l2 Error')
    fdrplot = sns.relplot(x="npr", y="fdr", hue='p',  kind="line", data=df)
    fdrplot.set(xlabel='T/p', ylabel='False Discovery Proportion')
    fdrplot.fig.set_size_inches(7, 7)
    l2rplot.fig.set_size_inches(7, 7)
    fig1 = l2rplot.fig
    fig2 = fdrplot.fig
    fig1.savefig(respath + f"l2errors" + ht + conv + f'zeta={zeta}_rds={rds}' + multi+".png")
    fig2.savefig(respath + f"fdr" + ht + conv + f'zeta={zeta}_rds={rds}'+ multi+".png")
    return 0

ps = np.array([64, 128, 256, 512])
Nops = np.array([0.25, 0.5, 1, 2,  4,  6,  8,10])
zeta = 0.1
mul = ''
rdsa = 0.1
heavyt = False
convex = True
if heavyt is True:
    ht = '_heavy_tail'
else:
    ht = ''
if convex is True:
    conv = '_convex_'
else:
    conv = ''
l2errors = np.array([[0.56404654, 0.45555971, 0.37356597, 0.35251649, 0.34628459, 0.33244176, 0.37024292, 0.36499318],
 [0.60747424, 0.51144387, 0.3106467,  0.31207966, 0.30237617, 0.29647705,
  0.2931112,  0.29918032],
 [0.51131468, 0.50757809, 0.32990716, 0.3111844,  0.31458075, 0.32126041,
  0.31905655, 0.31891524,],
 [0.62912297, 0.51503315 ,0.3198555,  0.3198555,  0.33225478, 0.32496259,
  0.31583183, 0.32053174,]])
fdrs = np.array([[0.65454545, 0.47272727, 0.363263636, 0.32243527, 0.27454545, 0.28121418,
  0.28481818, 0.28181818,],
 [0.66153846, 0.58461538, 0.39153846, 0.38461538, 0.26153846, 0.25384615,
  0.27284615, 0.24461538,],
 [0.62857143, 0.64285714, 0.42857143, 0.46857143, 0.28571429, 0.21428571,
  0.28571429, 0.25714286,],
 [0.63555556, 0.55555556, 0.3733563433, 0.343223433, 0.27222222, 0.26222222,
  0.29222222, 0.23222222,]])
drawplot(Nops=Nops, ps=ps, respath='/Users/he/PycharmProjects/CDE_GARCH/figures/', zeta=1, ht=ht, conv=conv, l2errors=l2errors, fdrs=fdrs, rds=rdsa, multi='')

heavyt = False
convex = False
if heavyt is True:
    ht = '_heavy_tail'
else:
    ht = ''
if convex is True:
    conv = '_convex_'
else:
    conv = ''
l2errors = np.array([[0.67528601, 0.56825076, 0.29259587, 0.31844879, 0.28886484, 0.31376941, 0.2992964,  0.27567209],
 [0.67711756, 0.50260438, 0.31182578, 0.32018949, 0.28770633, 0.26214475,
  0.25902947, 0.27988866,],
 [0.69924899, 0.41978617, 0.29310447, 0.30449129, 0.28761195, 0.28637564,
  0.29404582, 0.29265317],
 [0.66089128, 0.41378761, 0.29190105, 0.27969932, 0.29832417, 0.28019595,
  0.28504781, 0.29145267]])
fdrs = np.array([[0.56,       0.45,       0.36842105, 0.3567,        0.27272727, 0.20769231,
  0.1746,        0.15433333],
 [0.51818182, 0.4,        0.44444444, 0.2567,        0.14285714, 0.18181818,
  0.1,        0.06666667],
 [0.54782609, 0.33333333, 0.375,      0.2567 ,       0.25   ,    0.1346,
  0.16666667, 0.0845        ],
 [0.60769231, 0.4678     ,   0.33333333 ,0.14285714 ,0.16666667 ,0.132,
  0.14285714 ,0.0745       ]])
drawplot(Nops=Nops, ps=ps, respath='/Users/he/PycharmProjects/CDE_GARCH/figures/', zeta=0.1, ht=ht, conv=conv, l2errors=l2errors, fdrs=fdrs, rds=rdsa, multi='')

heavyt = True
convex = True
if heavyt is True:
    ht = '_heavy_tail'
else:
    ht = ''
if convex is True:
    conv = '_convex_'
else:
    conv = ''
l2errors = np.array([[0.60285143, 0.49756325, 0.31735435, 0.31049571, 0.30457012, 0.29964913,
  0.3085857,  0.3130708 ],
 [0.60062087, 0.50641648, 0.29851321, 0.30504037, 0.31475271, 0.31764234,
  0.3229735,  0.30988351],
 [0.63171553, 0.45417843, 0.35800587, 0.33936744, 0.32890553, 0.33395716,
  0.35824791, 0.32962963],
 [0.60871565, 0.41001044, 0.30895034, 0.30438351, 0.30925471, 0.30381146,
  0.31320308, 0.31024603],])
fdrs = np.array([[0.654323456,        0.453154312 ,       0.42345    ,    0.39859 ,       0.379984 ,       0.33333333,
  0.34875    ,    0.334587   ],
 [0.653636364, 0.52727273, 0.40909091, 0.40909091, 0.36363636, 0.36363636,
  0.36363636, 0.35454545],
 [0.66363636, 0.58181818, 0.54545455, 0.45454545, 0.38454545 ,0.27272727,
  0.32363636, 0.30454545,],
 [0.59714286, 0.55714286, 0.5273846, 0.5 ,       0.42857143, 0.28571429,
  0.31428571 ,0.26857143],])
drawplot(Nops=Nops, ps=ps, respath='/Users/he/PycharmProjects/CDE_GARCH/figures/', zeta=0.1, ht=ht, conv=conv, l2errors=l2errors, fdrs=fdrs, rds=rdsa, multi='')
# for zeta in [0.01,0.1,1]:
#     drawplot(Nops, ps, respath, zeta, ht, conv)

heavyt = True
convex = False
if heavyt is True:
    ht = '_heavy_tail'
else:
    ht = ''
if convex is True:
    conv = '_convex_'
else:
    conv = ''
l2errors = np.array([[0.29853602 ,0.26273231, 0.18231798 ,0.14047869, 0.1574598,  0.15426282,
  0.16660234 ,0.14251576],
 [0.33827619, 0.27598729, 0.16461718, 0.1757763,  0.15565441 ,0.14987909,
  0.1856984 , 0.15725618],
 [0.2740612, 0.26086852, 0.20664615, 0.16782353, 0.18600707, 0.14983738,
  0.15349041, 0.14759597],
 [0.3272081 , 0.29318587, 0.15356589, 0.15382997, 0.14845519, 0.15386729,
  0.1536722 , 0.14221809]])*2

fdrs = np.array([[0.63333333, 0.48095238, 0.37931034 ,0.3234278, 0.29411765 ,0.4,
  0.17647059 ,0.18888889],
 [0.70666667, 0.487949   ,    0.42380952, 0.4025 ,    0.37857143 ,0.337438333,
  0.19047619, 0.21666667],
 [0.62857143, 0.4375 ,    0.45555556, 0.44324444 ,0.28571429 ,0.375,
  0.28571429, 0.22631579],
 [0.68095238, 0.47619048, 0.5044323  ,   0.37842105, 0.35714286, 0.35454545,
  0.33333333, 0.28461538]])
drawplot(Nops=Nops, ps=ps, respath='/Users/he/PycharmProjects/CDE_GARCH/figures/', zeta=1, ht=ht, conv=conv, l2errors=l2errors, fdrs=fdrs, rds=rdsa, multi='')