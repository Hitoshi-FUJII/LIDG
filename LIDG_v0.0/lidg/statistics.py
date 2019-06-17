import numpy as np
import math
import scipy.stats



def mCn(m, n):
    return math.factorial(m) / (math.factorial(n) * (math.factorial(m - n)))

def mHn(m, n):
    m = m + n - 1
    return mCn(m,n)


def scale(x):
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0,ddof=0)
    x_scal = (x - x_mean) / x_std   
    return x_scal,x_mean,x_std

def normalize(x):
    norm = np.linalg.norm(x,axis=0)
    x_norm = x / norm
    return x_norm, norm


# --- for projection ---
def xtx_inv(x):
    xtx = np.dot(x.T,x)
    xtx_inv = np.linalg.inv(xtx)
    return xtx_inv

def hat(x):
    q,r = np.linalg.qr(x,mode="reduced")
    h = np.dot(q,q.T)
    lev = np.diag(h)
    return h,lev

def beta(x,y):
    q,r = np.linalg.qr(x,mode="reduced")
    qty = np.dot(q.T,y)
    b = np.linalg.solve(r,qty)
    return b

def res(x,y):
    h,lev = hat(x)
    e = y - np.dot(h,y)
    e2 = np.dot(e.T,e)
    return e, e2

def res_pre(x,y):
    m,n = x.shape
    h,lev = hat(x)
    e, e2 = res(x,y)
    eq = e / (1. - lev)
    eq2 = np.dot(eq.T,eq)
    return eq,eq2

def t_val(x,y):
    m,n = x.shape
    b = beta(x,y)
    e, e2 = res(x,y) 
    v = e2 / (m - n) 
    t = b / np.sqrt(v * np.diag(xtx_inv(x)))
    return t

def p_val(x,y):
    m,n = x.shape
    f = m - n
    t_abs = np.abs(t_val(x,y))
    p = scipy.stats.t.sf(t_abs,f)*2.
    return p


def r2(x,y):
    e, e2 = res(x,y)
    syy = np.sum((y - np.mean(y))**2)
    r2 = 1. - e2 / syy
    return r2

def tr2(x,y):
    e, e2 = res(x,y)
    y2 = np.dot(y.T,y)
    tr2 = 1. - e2 / y2
    return tr2


def q2(x,y):
    eq, eq2 = res_pre(x,y)
    syy = np.sum((y - np.mean(y))**2)
    q2 = 1. - eq2 / syy
    return q2

def tq2(x,y):
    eq, eq2 = res_pre(x,y)
    y2 = np.dot(y.T,y)
    tq2 = 1. - eq2 / y2
    return tq2


def giy2(x,xwoi,y):
    e, e2 = res(x,y)
    ei, ei2 = res(xwoi,y)
    g2 = 1.0 - e2 / ei2  
    return g2


def tri2(xwoi,xi):
    e, e2 = res(xwoi,xi)
    xi2 = np.sum(xi**2)
    tri2 = 1. - e2 / xi2
    return tri2
    
    
def aic(x,y):
    m,n = x.shape
    e,e2 = res(x,y)
    aic = m*np.log(e2) + m + m*np.log(2*np.pi/m) + 2*n
    return aic


# --- for correlation calculation ---
def rij(x):
    m,n = x.shape
    x_scal,x_mean,x_std = scale(x)
    r_mat = np.dot(x_scal.T,x_scal) / m
    rij_list = []
    for i in range(n):
        for j in range(i+1,n):
            rij = r_mat[i,j]
            rij_list.append([i,j,rij,rij**2.])
    rij_ar = np.array(rij_list)
    sort_ind = np.argsort(rij_ar[:,3])
    return rij_ar[sort_ind[::-1],:]

def nij(x):
    m,n = x.shape
    x_norm, norm = normalize(x)
    n_mat = np.dot(x_norm.T,x_norm)
    nij_list = []
    for i in range(n):
        for j in range(i+1,n):
            nij = n_mat[i,j]
            nij_list.append([i,j,nij,nij**2.]) 
    nij_ar = np.array(nij_list)
    sort_ind = np.argsort(nij_ar[:,3])        
    return nij_ar[sort_ind[::-1],:]

def gij(x):
    m,n = x.shape
    gij_list = []
    for i in range(n):
        for j in range(i+1,n):
            xwoij = np.delete(x,[i,j],axis=1)
            ei, ei2 = res(xwoij,x[:,i])
            ei_norm = ei / np.sqrt(ei2)
            ej, ej2 = res(xwoij,x[:,j])
            ej_norm = ej / np.sqrt(ej2)
            gij = np.dot(ei_norm.T,ej_norm)
            gij_list.append([i,j,gij,gij**2.])
    gij_ar = np.array(gij_list)
    sort_ind = np.argsort(gij_ar[:,3])   
    return gij_ar[sort_ind[::-1],:]

def gik(x,k):
    m,n = x.shape
    gik_list = []
    for i in range(n):
        if not i == k:
            xwoik = np.delete(x,[i,k],axis=1)  
            ei, ei2 = res(xwoik,x[:,i])
            ei_norm = ei / np.sqrt(ei2)
            ek, ek2 = res(xwoik,x[:,k])
            ek_norm = ek / np.sqrt(ek2)
            gik = np.dot(ei_norm.T,ek_norm)
            gik_list.append([i,k,gik,gik**2.])
    gik_ar = np.array(gik_list)
    sort_ind = np.argsort(gik_ar[:,3])   
    return gik_ar[sort_ind[::-1],:]


# --- for outlier calculation ---
def int_student(x,y):
    m,n = x.shape
    h, lev = hat(x)
    e, e2 = res(x,y)
    v = e2 / (m - n)    # const should be contained in "n"
    e_in = e / np.sqrt(v * (1 - lev))
    return e_in

def ext_student(x,y):
    m,n = x.shape
    e_in = int_student(x,y)
    e_ex = e_in * np.sqrt((m - n - 1) / (m - n - e_in**2))    
    return e_ex

def dffits(x,y):
    e_ex = ext_student(x,y)
    h, lev = hat(x)
    dffits = e_ex * np.sqrt(lev / (1 - lev))
    return dffits

