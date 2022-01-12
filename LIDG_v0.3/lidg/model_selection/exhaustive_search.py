import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import itertools
import time

import lidg.statistics as st
import lidg.model_selection.plot as pt

    
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
   

 
    # === e2 sort ===
    t2 = time.time()
    e_sort_df = df.sort_values(by="e2",ascending=True).reset_index()[:ntop]
    t3 = time.time()

    print(f"Elapsed time for sorting (by e2)")
    print(f"time : {t3-t2} [s]")
    print(f"time/{n}C{k} : {(t3-t2)/ncomb} [s/ncomb]")              
    print()
    
    e_des_lis = []
    r2_lis = []
    q2_lis = []
    print(f"Displayed only top {ntop} (sorted by e2)")
    for i,(et,qt,cmb) in enumerate(zip(e_sort_df["e2"],e_sort_df["eq2"],e_sort_df["comb"])):
        r2_lis.append(1.-et/syy)
        q2_lis.append(1.-qt/syy)
        e_des_lis.append([labs_list[j] for j in cmb])
    e_sort_df["R2"] = r2_lis
    e_sort_df["Q2"] = q2_lis
    e_sort_df["Descriptors"] = e_des_lis
    print(e_sort_df[["e2","R2","eq2","Q2","Descriptors"]])    
    
    pt.ex_plot(x,y,e_sort_df["comb"],"R2")
    print()
 

    # === q2 sort ===
    t4 = time.time()
    q_sort_df = df.sort_values(by="eq2",ascending=True).reset_index()[:ntop]
    t5 = time.time()
    
    print(f"Elapsed time for sorting (by eq2)")
    print(f"time : {t5-t4} [s]")
    print(f"time/{n}C{k} : {(t5-t4)/ncomb} [s/ncomb]")              
    print()

    q_des_lis = []
    r2_lis = []
    q2_lis = []
    print(f"Displayed only top {ntop} (sorted by eq2)")
    for i,(et,qt,cmb) in enumerate(zip(q_sort_df["e2"],q_sort_df["eq2"],q_sort_df["comb"])):
        r2_lis.append(1.-et/syy)
        q2_lis.append(1.-qt/syy)
        q_des_lis.append([labs_list[j] for j in cmb])
    e_sort_df["R2"] = r2_lis
    e_sort_df["Q2"] = q2_lis
    e_sort_df["Descriptors"] = q_des_lis
    print(e_sort_df[["e2","R2","eq2","Q2","Descriptors"]])    

    pt.ex_plot(x,y,q_sort_df["comb"],"Q2")
    
    return e_des_lis,q_des_lis 

