import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import itertools
import time

import lidg.statistics as st
from sklearn.linear_model import ElasticNet,ElasticNetCV



#  === Elastic net ===
def elastic_net(x_df,y_se,l1r,eps,kf,intercept,norm,itr,zero):
    labs = x_df.columns.tolist()

    X = x_df.values.copy()
    y = y_se.values.copy()
    syy = np.sum((y-np.mean(y))**2)
    m = len(y)

    enet_cv = ElasticNetCV(l1_ratio=l1r,eps=eps,cv=kf,fit_intercept=intercept,normalize=norm,max_iter=itr).fit(X,y)
    cvm = enet_cv.mse_path_.mean(axis=1)
    cvs = enet_cv.mse_path_.std(axis=1)
    se = cvs/np.sqrt(kf)  # standard error

    # Find "alpha_min" and "alpha_1se"
    amin = enet_cv.alpha_    # = alps[amin_pos]
    amin_pos = np.argmin(cvm)
    cvm_se = cvm[amin_pos] + se[amin_pos]
    a1se_pos = np.argmin(np.absolute(cvm[:amin_pos] - cvm_se))
    a1se = enet_cv.alphas_[a1se_pos] 
    
    # only for mse line
    mse_list = []
    for alp in enet_cv.alphas_:
        enet = ElasticNet(alpha=alp,l1_ratio=l1r,fit_intercept=intercept,normalize=norm,max_iter=itr).fit(X,y)
        r2 = enet.score(X,y)
        mse = (1.- r2)*syy/m
        mse_list.append(mse)
                 
    # Plotting (MSE,CV)
    plt.rcParams["font.size"]=18
    plt.figure(figsize=(8,6))
    plt.errorbar(np.log10(enet_cv.alphas_),cvm,yerr=se,fmt="k",ecolor="k",elinewidth=0.5,label="CVM with SE",lw=3)
    plt.axvline(np.log10(amin),linestyle="--",color="k",label="alpha min")
    plt.axvline(np.log10(a1se),linestyle=":",color="k",label="alpha 1se")
    plt.plot(np.log10(enet_cv.alphas_),mse_list,"b-",label="MSE",lw=3)
    plt.legend(loc="best",prop={"size":15})
    plt.xlabel("log(alpha)")
    plt.ylabel("Mean squared error")
    plt.title(f"{kf}-fold CV (by MSE)\nElastic net (L1 ratio: {l1r})")
    plt.axhline(0,linewidth=1,color="k")
    plt.grid()
    plt.show()
    
    print(f"log10(amin)={np.log10(amin)}")
    print(f"log10(a1se)={np.log10(a1se)}")
    print()
    
    coef_amin = list(enet_cv.coef_)
    coef_amin.insert(0,enet_cv.intercept_)

    enet = ElasticNet(alpha=a1se,l1_ratio=l1r,fit_intercept=intercept,normalize=norm,max_iter=itr).fit(X,y)
    coef_a1se = list(enet.coef_)
    coef_a1se.insert(0,enet.intercept_)
    labs.insert(0,"const")
    coe = pd.DataFrame({"Label":labs,"b_amin":coef_amin,"|b_amin|":np.abs(coef_amin),
                        "b_a1se":coef_a1se,"|b_a1se|":np.abs(coef_a1se)},
                        index=range(1,len(labs)+1))
    print(f"Normalize = {norm}")
    if norm: print(f"  coefficients are for normalized descriptors (for a fair comparison of weights)\n")
    else: print(f"  coefficients are for descriptors with original scale\n")
    print(coe)
    
    print()
    print("Selected descriptors:")
    nz_amin_list = []  # Non-zero labels
    ns_amin_list = []  # Not small labels
    nz_a1se_list = []  # Non-zero labels
    ns_a1se_list = []  # Not small labels
    for i,lab in enumerate(labs):
        if np.abs(coef_amin[i]) > 0.0:
            nz_amin_list.append(lab)
        if np.abs(coef_amin[i]) >= zero:
            ns_amin_list.append(lab)
        if np.abs(coef_a1se[i]) > 0.0:
            nz_a1se_list.append(lab)
        if np.abs(coef_a1se[i]) >= zero:
            ns_a1se_list.append(lab)
            
    print(f"At alpha = amin")
    r2_amin = 1.-m*mse_list[amin_pos]/syy
    print(f"  R2: {r2_amin}") 
    q2_amin = 1.-m*cvm[amin_pos]/syy
    print(f"  Mean R2 (~Q2): {q2_amin}")
    print()
    print(f"  Descriptors with a non-zero coefficient ( |b_amin| > 0.0 ) : ({len(nz_amin_list)})")
    print(f"    {nz_amin_list}\n")
    print(f"  Descriptors with a coefficient |b_amin| >= {zero} : ({len(ns_amin_list)})")
    print(f"    {ns_amin_list}\n")
    
    print(f"At alpha = a1se")
    r2_a1se = 1.-m*mse_list[a1se_pos]/syy
    print(f"  R2: {r2_a1se}")
    q2_a1se = 1.-m*cvm[a1se_pos]/syy
    print(f"  Mean R2 (~Q2): {q2_a1se}")
    print()
    print(f"  Descriptors with a non-zero coefficient ( |b_a1se| > 0.0 ) : ({len(nz_a1se_list)})")
    print(f"    {nz_a1se_list}\n")
    print(f"  Descriptors with a coefficient |b_a1se| >= {zero} : ({len(ns_a1se_list)})")
    print(f"    {ns_a1se_list}")

