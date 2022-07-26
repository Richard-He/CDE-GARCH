import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

heavy_tail = True

if heavy_tail:
    tail = '_ht'
else:
    tail = ''
sns.set_theme(style="darkgrid")
fdrs = np.load("results/fdrs"+ tail +".npy")
l2err = np.load("results/l2errors"+ tail +".npy")
Nprs = [4,5,6,7,8,9,10]*3
ps = [64]*7 + [128] *7 + [256]*7
df = pd.DataFrame(dict(npr=Nprs ,p=ps, fdr=fdrs.reshape(-1), l2error=l2err.reshape(-1)))

l2rplot = sns.relplot(x="npr", y="l2error", hue='p', kind="line", data=df)
fdrplot = sns.relplot(x="npr", y="fdr", hue='p',  kind="line", data=df)
fig1 = l2rplot.fig
fig2 = fdrplot.fig
fig1.savefig("results/l2r" + tail + ".png")
fig2.savefig("results/fdr" + tail + ".png")