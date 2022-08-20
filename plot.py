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

def drawplot(Nops, ps, respath, zeta, ht, conv, l2errors, fdrs):
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
    fdrplot = sns.relplot(x="npr", y="fdr", hue='p',  kind="line", data=df)
    fig1 = l2rplot.fig
    fig2 = fdrplot.fig
    fig1.savefig(respath + f"l2errors" + ht + conv + f'zeta={zeta}' + ".png")
    fig2.savefig(respath + f"fdr" + ht + conv + f'zeta={zeta}'+".png")
    return 0


# for zeta in [0.01,0.1,1]:
#     drawplot(Nops, ps, respath, zeta, ht, conv)