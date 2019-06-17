import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score,KFold
from sklearn import linear_model
from sklearn.linear_model import ElasticNet,enet_path,ElasticNetCV


# =============
#     Elastic net 
# =============
def elastic_net(x_df,y_se,almax=2,almin=-10,l1r=1.0,nf=10,intercept=True):
    #l1r = 0.01: Ridge, 1.0: LASSO
    X = x_df.values.copy()
    y = y_se.values.copy()
    
    labs = x_df.columns.tolist()
    log10a = np.linspace(almax,almin,100)  # from large to small alpha (lambda)
    alps = 10**(log10a)                    #  because "ElasticNetCV" obeys this order.
    
    # --- Caution! ---
    #  The hyper parameter "alpha"  in "Lasso" in sklearn is different
    #  from the hyper parameter "lambda" in "glmnet" in R package.
    #    in sklearn: 1/(2m) * (|y - yp|_2^2 + alpha_sk * |c|_1),
    #    in glmnet: 1/m * |y - yp|_2^2 + lambda * |c|_1.
    #  Then, the relationship between them is,
    #      lambda_R = alpha_sk / m,
    #  where m is the number of samples.
    # -------------------

    # MSE calculation (by "ElasticNet")
    ypre = []
    mse = []
    coef = []
    intr = []
    r2 = []
    for i in range(len(alps)):
        enet = ElasticNet(alpha=alps[i],l1_ratio=l1r,fit_intercept=intercept).fit(X,y)
        yp = enet.predict(X)
        ypre.append(yp)
        mse.append(mean_squared_error(y,yp))   # MSE
        r2.append(enet.score(X,y))
        coef.append(list(enet.coef_))    # coefficients
        intr.append(enet.intercept_)    # intercepts    

    # Cross varidation calculation (by "ElasticNetCV")
    kf = KFold(n_splits=nf,shuffle=True)
    enet_cv = ElasticNetCV(l1_ratio=l1r,cv=kf,alphas=alps,fit_intercept=intercept).fit(X,y)
    cvm = enet_cv.mse_path_.mean(axis=1)
    cvs = enet_cv.mse_path_.std(axis=1)
    se = cvs/np.sqrt(nf)  # standard error

    # Find "alpha_min" and "alpha_1se"
    amin = enet_cv.alpha_    # = alps[amin_pos]
    amin_pos = np.argmin(cvm)
    cvm_se = cvm[amin_pos] + se[amin_pos]
    
    a1se_pos = np.argmin(np.absolute(cvm[:amin_pos] - cvm_se))
    a1se = enet_cv.alphas_[a1se_pos]  # = alps[a1se_pos]
    #a1se_pos = amin_pos
    #a1se = amin
    
    # Plotting (MSE,CV)
    plt.rcParams["font.size"]=18
    plt.figure(figsize=(8,6))
    #plt.plot(log10a,enet_cv.mse_path_,":")
    plt.axvline(np.log10(amin),linestyle="--",color="k",label="alpha min")
    plt.axvline(np.log10(a1se),linestyle=":",color="k",label="alpha 1se")
    plt.errorbar(log10a,cvm,yerr=se,fmt="k",ecolor="k",elinewidth=0.5,label="CVM with SE",lw=3)
    plt.plot(log10a,mse,"b-",label="MSE",lw=3)
    plt.legend(loc="best",prop={"size":15})
    plt.xlabel("log(alpha)")
    plt.ylabel("Mean-Squared Error")
    plt.title(str(nf)+"-fold CV (by MSE)\nElasticNet (L1 ratio: "+str(l1r)+")")
    #plt.axis('tight')
    #plt.xlim(min(log10a_cv)-0.1, max(log10a_cv)+0.1)
    #plt.xlim(-2,0)
    #plt.ylim(10, 90)
    plt.axhline(0,linewidth=1,color="k")
    plt.grid()
    plt.show()

    
    # Non-zero count
    elastic_net_non_zero_count(alps,coef,amin,amin_pos,a1se,a1se_pos)

    # Path plot
    elastic_net_path(x_df,y_se,almax,almin,l1r,amin,a1se,intercept)

    
    plt.figure(figsize=(8,6))
    axes = plt.gca()
    coef_t = map(list,zip(*coef))     # transposed list
    for coe,lab in zip(coef_t,labs):
        plt.plot(log10a,coe)
        axes.text(log10a[20],coe[20],lab,fontsize=10)
        axes.text(log10a[40],coe[40],lab,fontsize=10)
        axes.text(log10a[60],coe[60],lab,fontsize=10)
        axes.text(log10a[80],coe[80],lab,fontsize=10)
        axes.text(log10a[99],coe[99],lab,fontsize=10)
    plt.axvline(np.log10(amin),linestyle="--",color="k",label="alpha min")
    plt.axvline(np.log10(a1se),linestyle=":",color="k",label="alpha 1se")
    plt.title("Regularization path\nElasticNet (L1 ratio: "+str(l1r)+")")
    plt.xlabel('log10(alpha)')
    plt.ylabel('Coefficients')
    #plt.xlim(min(log10a)-0.1, max(log10a)+0.1)
    plt.axhline(0,linewidth=1,color="k")
    plt.grid()
    plt.show()

    
    r2_a1se = r2[a1se_pos]
    mse_a1se = mse[a1se_pos]
    cvm_a1se = cvm[a1se_pos]
    cvs_a1se = cvs[a1se_pos] 
    coef_a1se = coef[a1se_pos]
    labs.insert(0,"const")
    coef_a1se.insert(0,intr[a1se_pos])
    coe = pd.DataFrame({"Label":labs,"b":coef_a1se,"|b|":np.abs(coef_a1se)},index=range(1,len(labs)+1))
    cri = pd.DataFrame({"Value":[r2_a1se,mse_a1se,cvm_a1se,cvs_a1se]}
                              ,index=["R2","MSE","CV mean","CV std"])

    yp_a1se_se = pd.Series(ypre[a1se_pos],name="yp_skl",index=y_se.index)
        
    return coe,cri,yp_a1se_se


def elastic_net_non_zero_count(alps,coef,amin,amin_pos,a1se,a1se_pos):
    # Non-zero count
    nzc = []
    for i in range(len(alps)):
        ca = np.array(coef[i])   # coef (list) is from MSE calc.
        ca_nz = np.nonzero(ca)[0].size
        nzc.append([i,alps[i],ca_nz])
    nzca = np.array(nzc)

    # Plotting (non-zero count)
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.log10(nzca[:,1]),nzca[:,2])
    plt.axvline(np.log10(amin),linestyle="--",color="k",label="alpha min")
    plt.axvline(np.log10(a1se),linestyle=":",color="k",label="alpha 1se")
    plt.xlabel('log(alpha)')
    plt.ylabel('Non-zero count')
    plt.axhline(0,linewidth=1,color="k")
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(nzca[:,0],nzca[:,2])
    plt.axvline(amin_pos,linestyle='--',color='k',label='alpha min')
    plt.axvline(a1se_pos,linestyle=':',color='k',label='alpha 1se')
    plt.xlabel('alpha index')
    plt.ylabel('Non-zero count')
    plt.xlim(len(alps),0)
    plt.axhline(0,linewidth=1,color="k")
    plt.grid()
    plt.show()


def elastic_net_path(x_df,y_se,almax,almin,l1r,amin,a1se,intercept):
    X = x_df.values.copy()
    y = y_se.values.copy()
    
    labs = x_df.columns
    log10a = np.linspace(almax,almin,100)  
    alps = 10**(log10a)         
    
    # === Path calculation (by "enet_path") ===
    alphas,coeffs,_ = enet_path(X,y,alphas=alps,l1_ratio=l1r,fit_intercept=intercept)

    # Plotting (path)
    plt.figure(figsize=(8,6))
    axes = plt.gca()
    for coeff,lab in zip(coeffs,labs):
        plt.plot(log10a,coeff)
        axes.text(log10a[20],coeff[20],lab,fontsize=10)
        axes.text(log10a[40],coeff[40],lab,fontsize=10)
        axes.text(log10a[60],coeff[60],lab,fontsize=10)
        axes.text(log10a[80],coeff[80],lab,fontsize=10)
        axes.text(log10a[99],coeff[99],lab,fontsize=10)
    plt.axvline(np.log10(amin),linestyle="--",color="k",label="alpha min")
    plt.axvline(np.log10(a1se),linestyle=":",color="k",label="alpha 1se")
    plt.title("Regularization path\nenet_path (L1 ratio: "+str(l1r)+")")
    plt.xlabel('log10(alpha)')
    plt.ylabel('Coefficients')
    #plt.xlim(min(log10a)-0.1, max(log10a)+0.1)
    plt.axhline(0,linewidth=1,color="k")
    plt.grid()
    plt.show()
