import matplotlib.pyplot as plt
#import seaborn as sns

import math
import itertools
import numpy as np
import pandas as pd

               
    
def preplot(x_df_se,y_se=None):
    if type(x_df_se) is pd.core.series.Series:
        x_df = pd.DataFrame()
        x_df[x_df_se.name] = x_df_se
    else:
        x_df = x_df_se
    labs_list = x_df.columns.tolist()
    nlabs = len(labs_list)
    nind = len(x_df.index)
    plt.figure(figsize=(8,3))
    #sns.set_style("whitegrid")
    plt.rcParams["font.size"]=15
    if y_se is not None:
        plt.plot(range(nind),y_se,alpha=0.8,marker="o",label=y_se.name)
    for lab in labs_list:
        plt.plot(range(nind),x_df[lab],alpha=0.8,marker="o",label=lab)
    plt.xlabel("Index")
    plt.ylabel("Descriptor values")
    plt.grid()
    #plt.xlim(-0.5,27)
    #plt.ylim(-.8,)
    plt.legend(loc="best",prop={"size":15})
    #plt.colorbar()
    #plt.savefig("./preplot.png")
    plt.show()


def list_plot(x_list,y_list,label=None):
    plt.figure(figsize=(6,5))
    plt.rcParams["font.size"]=15
    plt.plot(x_list,y_list,"+-",alpha=0.8,label=label)
    #plt.title(title)
    plt.legend(loc="best",prop={"size":12})
    #plt.xlabel("Index")
    #plt.ylabel("Feature values")
    #plt.xlim(-0.5,27)
    #plt.ylim(-.8,)
    plt.grid()
    #plt.colorbar()
    #plt.savefig("./preplot.png")
    plt.show()

    
def vs_plot(list1,list2,pmin,pmax):
    if pmin == None:
        pmin = min(list1) - (max(list1) - min(list1)) / 10.
    if pmax == None:
        pmax = max(list1) + (max(list1) - min(list1)) / 10.
    plt.figure(figsize=(6,5))
    plt.rcParams["font.size"]=15
    plt.scatter(list1,list2,s=50,alpha=0.7)
    plt.plot([pmin,pmax],[pmin,pmax],ls="--",c="k",lw=1.)
    plt.xlabel("First list")
    plt.ylabel("Second list")
    plt.grid()
    plt.xlim(pmin,pmax)
    plt.ylim(pmin,pmax)
    #plt.legend(loc="upper left",prop={"size":12})
    #plt.legend(loc="upper left",prop={"size":14})
    #plt.savefig("vs.eps")
    plt.show()
    

def lev_plot(lev,m,n):
    outlier = 2*n/m 
    plt.figure(figsize=(8,3))
    plt.rcParams["font.size"]=15
    plt.plot(range(m),lev,alpha=0.8,marker="o",label="Leverage")
    plt.axhline(outlier,linestyle="--",color="k",label="Outlier line (for Leverage)")
    plt.xlabel("Index")
    plt.ylabel("Leverages")
    plt.grid()
    plt.legend(loc="best",prop={"size":15})
    plt.show()    
    
def res_plot(e,eq):
    m = len(e)
    plt.figure(figsize=(8,3))
    plt.rcParams["font.size"]=15
    plt.plot(range(m),e,alpha=0.8,marker="o",label="Residuals")
    plt.plot(range(m),eq,alpha=0.8,marker="o",label="Pred_residuals")
    plt.xlabel("Index")
    plt.ylabel("Residuals")
    plt.grid()
    plt.legend(loc="best",prop={"size":15})
    plt.show()    

def res_plot2(e_in,e_ex,dff,m,n):
    outlier = 2*np.sqrt(n/m) 
    plt.figure(figsize=(8,3))
    plt.rcParams["font.size"]=15
    plt.plot(range(m),e_in,alpha=0.8,marker="o",label="Internally studentized residuals")
    plt.plot(range(m),e_ex,alpha=0.8,marker="o",label="Externally studentized residuals")
    plt.plot(range(m),dff,alpha=0.8,marker="o",label="DFFITS")
    plt.axhline(outlier,linestyle="--",color="k",label="Outlier line (for DFFITS)")
    plt.axhline(-outlier,linestyle="--",color="k",label="")
    plt.xlabel("Index")
    plt.ylabel("Residuals and DFFITS")
    plt.grid()
    plt.legend(loc="best",prop={"size":12})
    plt.show()
    
def x_res_plot(x,e,label=None):
    plt.figure(figsize=(5,3))
    plt.rcParams["font.size"]=15
    plt.scatter(x,e,s=50,alpha=0.7,label=label)
    plt.xlabel("Value of the descriptor "+str(label))
    plt.ylabel("Residuals")
    plt.grid()
    #plt.xlim(-0.5,27)
    #plt.ylim(-.8,)
    plt.legend(loc="best",prop={"size":15})
    #plt.colorbar()
    #plt.savefig("./preplot.png")
    plt.show()
    
    
    
    
# --- for Ridge and LASSO(Elastic net) ---
def path_plot(labs,beta,ll,min_pos,one_se_pos):
    bt = np.array(beta).T
    plt.figure(figsize=(8,5))
    axes = plt.gca()
    for b,lab in zip(bt[1:],labs[1:]):
        plt.plot(ll,b)
        axes.text(ll[20],b[20],lab,fontsize=12)
        axes.text(ll[40],b[40],lab,fontsize=12)
        axes.text(ll[60],b[60],lab,fontsize=12)
        axes.text(ll[80],b[80],lab,fontsize=12)
        axes.text(ll[99],b[99],lab,fontsize=12)
    plt.axvline(ll[min_pos],linestyle="--",color="k",label="min lambda")
    plt.axvline(ll[one_se_pos],linestyle=":",color="k",label="1se lambda")
    plt.title("Regularization path")
    plt.xlabel('log10(Lambda)')
    plt.ylabel('Coefficients')
    #plt.xlim(min(log10a)-0.1, max(log10a)+0.1)
    plt.axhline(0,linewidth=1,color="k")
    plt.grid()
    plt.show()
    
    
def mse_plot(labs,ll,mse_ful_arr,mse_arr,se_arr,min_pos,one_se_pos):
    plt.rcParams["font.size"]=15
    plt.figure(figsize=(8,5))
    #plt.plot(log10a,enet_cv.mse_path_,":")
    plt.axvline(ll[min_pos],linestyle="--",color="k",label="min lambda")
    plt.axvline(ll[one_se_pos],linestyle=":",color="k",label="1se lambda")
    plt.errorbar(ll,mse_arr,yerr=se_arr,fmt="k",ecolor="k",elinewidth=0.5,label="Mean MSE",lw=3)
    plt.plot(ll,mse_ful_arr,"b-",label="MSE",lw=3)
    plt.legend(loc="best",prop={"size":15})
    plt.xlabel("log10(Lambda)")
    plt.ylabel("Mean Squared Error")
    #plt.title()
    #plt.axis('tight')
    #plt.xlim(min(log10a_cv)-0.1, max(log10a_cv)+0.1)
    #plt.xlim(-5,3)
    #plt.ylim(10, 40)
    plt.axhline(0,linewidth=1,color="k")
    plt.grid()
    plt.show()
    