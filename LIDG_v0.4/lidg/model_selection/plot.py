import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sb
import lidg.statistics as st

    
def ex_plot(x,y,cmb_list,r_or_q):
    beta_list = []
    r2_list = []
    q2_list = []
    x, norm = st.normalize(x)
    n = x.shape[1]
    for cmb in cmb_list:
        b_list = [0]*n
        b = st.beta(x[:,cmb],y)
        for i,j in enumerate(cmb):
            b_list[j] = b[i]
            #b_list[j] = 1
        b_list = b_list[::-1]
        beta_list.append(b_list)
        r2_list.append(st.r2(x[:,cmb],y))
        q2_list.append(st.q2(x[:,cmb],y))
    
    # Plot for R2 and Q2 curves
    plt.figure(figsize=(8,5))
    plt.rcParams["font.size"]=15
    plt.plot(range(len(cmb_list)),r2_list,"o-",alpha=0.8,label="R2")
    plt.plot(range(len(cmb_list)),q2_list,"o-",alpha=0.8,label="Q2")
    plt.legend(loc="best",prop={"size":12})
    plt.xlabel(f"Ranking of models (sorted by {r_or_q} score)")
    plt.ylabel("Score")
    plt.grid()
    plt.show()
    
    df = pd.DataFrame(beta_list,columns=list(range(n))[::-1])
    
    plt.figure(figsize=(10,5))
    sb.heatmap(df.T,square=False,cmap='coolwarm',
               vmin=-1,vmax=1,
               xticklabels=10,
               #yticklabels=False,
               #yticklabels=5,
               linecolor="black",
               cbar=True)
    plt.xlabel(f"Ranking of models (sorted by {r_or_q} score. Beta value (color) is normalized.)")
    plt.ylabel("Descriptors")

    plt.show()

