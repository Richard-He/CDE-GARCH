import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

heavy_tail = True


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

