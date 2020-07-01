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


    
    
# === Exhaustive search ===
def exhaustive_search(x_df,y_se,k,ntop,const):
    labs_list = x_df.columns.tolist()
    n = len(labs_list)
    x = x_df.values.copy()
    y = y_se.values.copy()
    syy = np.sum((y-np.mean(y))**2)
    if const: 
        k = k-1
        n = n-1
        print("Caution!:")
        print("  Constant term is always included in the search.")
        print("  The number of descriptors contained in each combination is k.")
        print("  (constant term is also regarded as a descriptor.)")
        print("  The total number of combinations should be searched is now (n-1)C(k-1).")
        print("  (constant term should be in the first column of span.)")
        print()
    ncomb = int(st.mCn(n,k))
    print(f"Exhaustive search for {k} descriptors:  {n}C{k} = {ncomb}")

    di = 0.05*ncomb # display interval
    cnt = 1
    e_list = []
    
    t0 = time.time()
    
    if const:
        for i,comb in enumerate(itertools.combinations(range(1,n+1),k), start=1): # for with const
            comb = list(comb)
            comb.insert(0,0)  
            q,r = np.linalg.qr(x[:,comb],mode="reduced")
            qty = np.dot(q.T,y)
            e = y - np.dot(q,qty)
            q_adm = q*q
            lev = np.sum(q_adm,axis=1) # leverage (diagonal part of hat matrix)
            eq = e / (1. - lev) 

            e2 = np.dot(e,e)
            eq2 = np.dot(eq,eq)
            e_list.append([e2,eq2,comb])
            
            if i >= di*cnt:
                print(f"{int(i/ncomb*100): >4} %  ({i}/{ncomb})")
                cnt += 1
        
    else:
        for i,comb in enumerate(itertools.combinations(range(n),k), start=1):
            q,r = np.linalg.qr(x[:,comb],mode="reduced")
            qty = np.dot(q.T,y)
            e = y - np.dot(q,qty)
            q_adm = q*q
            lev = np.sum(q_adm,axis=1) # leverage (diagonal part of hat matrix)
            eq = e / (1. - lev) 

            e2 = np.dot(e,e)
            eq2 = np.dot(eq,eq)
            e_list.append([e2,eq2,comb])

            if i >= di*cnt:
                print(f"{int(i/ncomb*100): >4} %  ({i}/{ncomb})")
                cnt += 1

    print()
    print(f"Exhaustive search is finished.{int(i/ncomb*100): >4} %  ({i}/{ncomb})")
    print()
    t1 = time.time()
    print(f"Elapsed time for score calculatons")
    print(f"time: {t1-t0} [s]")
    print(f"time/{n}C{k} : {(t1-t0)/ncomb} [s/ncomb]")              
    print()

    df = pd.DataFrame(e_list,columns=["e2","eq2","comb"])
    
    t2 = time.time()
    e_sort_df = df.sort_values(by="e2",ascending=True).reset_index()[:ntop]
    t3 = time.time()
    
    print(f"Elapsed time for sorting (by |e|2)")
    print(f"time : {t3-t2} [s]")
    print(f"time/{n}C{k} : {(t3-t2)/ncomb} [s/ncomb]")              
    print()
    
    cmb_list = []
    print(f"Displayed only top {ntop} (sorted by |e|2)")
    print(f"            |e|2           R2           |eq|2           Q2         Descriptors")
    for i,(et,qt,cmb) in enumerate(zip(e_sort_df["e2"],e_sort_df["eq2"],e_sort_df["comb"])):
        print(f"{i: >4}   {et:.6g}   {1.-et/syy:6g}   {qt:.6g}   {1.-qt/syy:6g}    {[labs_list[j] for j in cmb]}")
        cmb_list.append(cmb)
    ex_plot(x,y,cmb_list,"R2")
    print()


    t4 = time.time()
    q_sort_df = df.sort_values(by="eq2",ascending=True).reset_index()[:ntop]
    t5 = time.time()
    
    print(f"Elapsed time for sorting (by |eq2|)")
    print(f"time : {t5-t4} [s]")
    print(f"time/{n}C{k} : {(t5-t4)/ncomb} [s/ncomb]")              
    print()


    cmb_list = []
    print(f"Displayed only top {ntop} (sorted by |eq|2)")
    print(f"            |e|2           R2           |eq|2           Q2         Descriptors")
    for i,(et,qt,cmb) in enumerate(zip(q_sort_df["e2"],q_sort_df["eq2"],q_sort_df["comb"])):
        print(f"{i: >4}   {et:.6g}   {1.-et/syy:6g}   {qt:.6g}   {1.-qt/syy:6g}    {[labs_list[j] for j in cmb]}")
        cmb_list.append(cmb)
    ex_plot(x,y,cmb_list,"Q2")
    
    
# For plotting
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
    plt.xlabel(f"Ranking of models (sorted by {r_or_q} score. beta value (color) is normalized.)")
    plt.ylabel("Descriptors")

    plt.show()

