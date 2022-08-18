import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

heavy_tail = True



respath = 'results/'
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
l2errors = np.load(respath + f"l2errors" + ht + conv + "new"+ ".npy")
fdrs = np.load(respath + f"fdrs" + ht + conv + "new"+".npy")
lambda_errs = np.load(respath + f"lambda_errors" + ht + conv +"new"+ ".npy")
V_errs = np.load(respath + f"V_errs" + ht + conv +"new"+ ".npy")
sns.set_theme(style="darkgrid")
Nops = np.array([0.5, 1, 2, 4, 8])


def drawplot(fdrs, l2errors, Nops, ps, graphpath, zeta, ht, conv):
    Nopshape = Nops.shape[0]
    psshape = ps.shape[0]
    pss = np.array([p for p in ps for i in range(Nopshape)])
    Nopss = np.array(Nops.tolist() * int(psshape))
    df = pd.DataFrame(dict(npr=Nopss, p=pss, fdr=fdrs.reshape(-1), l2error=l2errors.reshape(-1)))

    l2rplot = sns.relplot(x="npr", y="l2error", hue='p', kind="line", data=df)
    fdrplot = sns.relplot(x="npr", y="fdr", hue='p',  kind="line", data=df)
    fig1 = l2rplot.fig
    fig2 = fdrplot.fig
    fig1.savefig(graphpath + f"l2errors" + ht + conv + f'zeta={zeta}' + ".png")
    fig2.savefig(graphpath + f"fdr" + ht + conv + f'zeta={zeta}'+".png")
    return 0