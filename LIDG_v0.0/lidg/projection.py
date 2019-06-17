import numpy as np
import pandas as pd
import math

import lidg.statistics as st
import lidg.plots as plots


class Projection:
    def __init__(self):
        self.beta = None
        self.hy = None


    def prepare(self,x_df):
        labs_list = x_df.columns.tolist()
        x = x_df.values.copy()
        x = np.array(x,dtype="float64")
        m,n = x.shape
        return x,labs_list,m,n


    def mc_check(self,x):
        m,n = x.shape
        rk = np.linalg.matrix_rank(x)
        if not (rk == min(m,n) and m > n):
            raise RankError(f"rank(X) (={rk}) < min({m},{n})  or  m (={m}) <= n (={n})")


    def vector_projection(self,x_df,y_se,sort,normalize):
        x,labs_list,m,n = self.prepare(x_df)
        y = y_se.values.copy()
        y = np.array(y,dtype="float64")
        x, norm = st.normalize(x)
        self.mc_check(x)

        e,e2 = st.res(x,y)
        eq,eq2 = st.res_pre(x,y)
        b_norm = st.beta(x,y)
        b = b_norm / norm
        
        if normalize:
            beta = b_norm
        else:
            beta = b
        
        p_val = st.p_val(x,y)
        log10p = np.log10(p_val)

        giy2_list = []
        for i in range(n):
            xwoi = np.delete(x,i,axis=1)
            giy2 = st.giy2(x,xwoi,y)
            giy2_list.append(giy2)

        tri2_list = []    
        for i in range(n):
            xwoi,xi = np.delete(x,i,axis=1),x[:,i]
            tri2 = st.tri2(xwoi,xi)
            tri2_list.append(tri2)

        print(f"  e2 = {e2: .6f}")
        print(f"  mse = {e2/m: .6f}")
        print(f"  rmse = {np.sqrt(e2/m): .6f}")
        print(f"  R2 = {st.r2(x,y): .6f}")
        #print(f"  TR2 = {st.tr2(x,y): .6f}")
        print(f"  eq2 = {eq2: .6f}")
        print(f"  mseq = {eq2/m: .6f}")
        print(f"  rmseq = {np.sqrt(eq2/m): .6f}")
        print(f"  Q2 = {st.q2(x,y): .6f}")
        #print(f"  TQ2 = {st.tq2(x,y): .6f}")
        #print(f"  AIC = {st.aic(x,y): .6f}")
        print()
        print(f"  TR2, TQ2, AIC = {st.tr2(x,y): .6f},  {st.tq2(x,y): .6f},  {st.aic(x,y): .6f}")
        print()

        nr = 5
        results_df = pd.DataFrame({"Label":labs_list,"b":np.round(beta,nr),
                                   "|b|":np.round(np.abs(beta),nr),
                                   "p_val":np.round(p_val,nr),"-log10(p)":np.round(-log10p,nr),
                                   "G2":np.round(giy2_list,nr),"TRi2":np.round(tri2_list,10)},
                                   columns=["Label","b","|b|","p_val","-log10(p)","G2","TRi2"],index=range(1,n+1))
        if not sort == None:
            results_df = results_df.sort_values(by=sort,ascending=False).reset_index()
        print(results_df)
        self.beta = b
        hy = np.dot(x,b_norm)
        self.hy = pd.Series(hy,name="hy",index=y_se.index)

    
    def rij(self,x_df,ntop):
        x,labs_list,m,n = self.prepare(x_df)
        if ntop == None:
            ntop = int(st.mCn(n,2))
        rij_ar = st.rij(x)         
        rij_list = []
        for i in range(int(st.mCn(n,2))):
            lab1 = labs_list[int(rij_ar[i,0])]
            lab2 = labs_list[int(rij_ar[i,1])]
            val = rij_ar[i,2]
            val2 = rij_ar[i,3]
            rij_list.append([lab1,lab2,val,val2])
        rij_df = pd.DataFrame(rij_list,columns=["i","j","rij","rij2"])
        print(rij_df[:ntop])     


    def nij(self,x_df,ntop):
        x,labs_list,m,n = self.prepare(x_df)
        if ntop == None:
            ntop = int(st.mCn(n,2))
        nij_ar = st.nij(x)         
        nij_list = []
        for i in range(int(st.mCn(n,2))):
            lab1 = labs_list[int(nij_ar[i,0])]
            lab2 = labs_list[int(nij_ar[i,1])]
            val = nij_ar[i,2]
            val2 = nij_ar[i,3]
            nij_list.append([lab1,lab2,val,val2])
        nij_df = pd.DataFrame(nij_list,columns=["i","j","nij","nij2"])
        print(nij_df[:ntop])    
    
    
    def gij(self,x_df,ntop):
        x,labs_list,m,n = self.prepare(x_df)
        self.mc_check(x)
        if ntop == None:
            ntop = int(st.mCn(n,2))
        gij_ar = st.gij(x)         
        gij_list = []
        for i in range(int(st.mCn(n,2))):
            lab1 = labs_list[int(gij_ar[i,0])]
            lab2 = labs_list[int(gij_ar[i,1])]
            val = gij_ar[i,2]
            val2 = gij_ar[i,3]
            gij_list.append([lab1,lab2,val,val2])
        gij_df = pd.DataFrame(gij_list,columns=["i","j","gij","gij2"])
        print(gij_df[:ntop])     


    def gik(self,x_df,lab_k,ntop):
        x,labs_list,m,n = self.prepare(x_df)
        k = labs_list.index(lab_k)
        self.mc_check(x)

        if ntop == None:
            ntop = n-1
        gik_ar = st.gik(x,k)   
        gik_list = []
        for i in range(n-1):
            lab1 = labs_list[int(gik_ar[i,0])]
            lab2 = labs_list[int(gik_ar[i,1])]
            val = gik_ar[i,2]
            val2 = gik_ar[i,3]
            gik_list.append([lab1,lab2,val,val2])
        gik_df = pd.DataFrame(gik_list,columns=["i","k","gik","gik2"])
        print(gik_df[:ntop])        
    
    
    def outlier(self,x_df,y_se,ntop):
        x,labs_list,m,n = self.prepare(x_df)
        x,norm = st.normalize(x)
        y = y_se.values.copy()
        y = np.array(y,dtype="float64")
        self.mc_check(x)

        e,e2 = st.res(x,y)
        eq,eq2 = st.res_pre(x,y)
        h,lev = st.hat(x)
        plots.lev_plot(lev,m,n)
        plots.res_plot(e,eq)
        e_in = st.int_student(x,y)
        e_ex = st.ext_student(x,y)
        dff = st.dffits(x,y)

        plots.res_plot2(e_in,e_ex,dff,m,n)

        lev_sind = np.argsort(lev)
        dff_sind = np.argsort(np.abs(dff))

        print("Outliers (top "+str(ntop)+")")
        outlier_list = []
        for i in range(1,m+1):
            outlier_list.append([np.round(lev[lev_sind[-i]],4),lev_sind[-i],
                                 np.round(np.abs(dff[dff_sind[-i]]),4),dff_sind[-i]])

        results_df = pd.DataFrame(outlier_list,
                                  columns=["Leverages","index","|DFFITS|","index"],
                                  index=range(m))
        print(results_df[:ntop])



    
# === Exception ===
class RankError(Exception):
    def __init__(self,comme):
        self.comme = comme
        
    def __str__(self):
        return str(self.comme)
    
    