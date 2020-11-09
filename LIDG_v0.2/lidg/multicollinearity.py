import numpy as np
import pandas as pd
import itertools
import lidg.statistics as st


def q_matrix(order):
    n = len(order)
    q_mat = np.zeros((n,n))
    for i in range(n):
        q_mat[order[i],i] = 1.
    return q_mat


def get_tolerances(tole1,tole2,m):
    ave_val = 1. / np.sqrt(m)
    print(f"1/sqrt(m) = {ave_val}")
    if tole1 == None:
        tole1 = ave_val /100.
        print("tole1 is given automatically.")
    if tole2 == None:
        tole2 = tole1 / 100.
        print("tole2 is given automatically.")
    print(f"tole1, tole2 = {tole1}, {tole2}\n")    
    return tole1,tole2
        

def model_form(y_name,x_name_list,coe_list,zero=0.00001):
    n = len(x_name_list) 
    form = str(y_name)+" = "
    cnt = 0
    for i in range(n):
        if(np.abs(coe_list[i]) >= zero):
            x_coe = np.round(coe_list[i],3)
            if cnt == 0:
                form = form + "(" + str(x_coe) + ")"
            else:
                form = form + " + (" + str(x_coe) + ")"                
            #form = form + "*" + str(x_name_list[i])
            form = form + str(x_name_list[i])
            cnt += 1
    return form
    
    
def insertion_sort_2val(s,p1,p2):
    # assuming that p1 is more important than p2.
    m = len(s)
    for i in range(1,m):
        if s[i][p1] < s[i-1][p1]:
            for j in range(i):
                if s[i][p1] < s[j][p1]: 
                    s.insert(j,s.pop(i))
                    break
                elif s[i][p1] == s[j][p1]:
                    for k in range(j,i):
                        if s[i][p1] == s[k][p1] and s[i][p2] <= s[k][p2]:
                            s.insert(k,s.pop(i))
                            break
    return s


def find_subspace(labs,c_mat_t,new_order,rk):
    zero = 0.00001
    ns = c_mat_t.shape[0]
    f_list = []    # solution form list
    for i in range(ns):
        y_lab = labs[new_order[rk+i]]
        sol = c_mat_t[i,:rk]
        lab_list = []
        for j in range(rk):
            x_lab = labs[new_order[j]]
            if np.abs(sol[j]) >= zero:
                lab_list.append(x_lab)
        lab_list = [y_lab,lab_list,1,len(lab_list)]
        f_list.append(lab_list)    
    
    ss_list = []
    for i in range(ns):
        if f_list[i][-2] == 1:
            for j in range(i+1,ns):
                if f_list[i][-3] == f_list[j][-3]:
                    f_list[i].insert(-3,f_list[j][0])
                    f_list[i][-2] += 1
                    f_list[j][-2] = 0
            ss_list.append(f_list[i])
            
    ss_list = insertion_sort_2val(ss_list,-1,-2) 
    
    nssl = len(ss_list)       
    print(f"Subspace list: {nssl}")
    for i,ssl in enumerate(ss_list):
        print(f"{i+1} {ssl[:-2]}")
    print("")   


def rref(x_in,tole1,tole2):
    x = x_in.copy()
    m,n = x.shape
    order_list = [i for i in range(n)]  # the order of columns
    
    for i in range(min(m,n)):
        ind_i_max = np.argmax(np.abs(x[i:,i])) + i
        x_i_max = x[ind_i_max,i]
        pivot = np.abs(x_i_max)
        # for complete pivoting
        if pivot <= tole1:
            max_col_ar = np.max(np.abs(x[i:,i:]),axis=0)
            ind_col_max = np.argmax(np.abs(max_col_ar)) + i
            sx_max = np.max(np.abs(max_col_ar)) # maximum value in small matrix x[i:,i:]
            pivot = np.abs(sx_max)
            if pivot <= tole2:
                print(f"X is a rank deficient matrix ( pivot ( = {pivot})  <  tole2 ( = {tole2}) )")
                return x,order_list
            else:
                ind_row_max = np.argmax(np.abs(x[i:,ind_col_max])) + i
                x_row_max = x[ind_row_max,ind_col_max]    
                order_list.insert(i,order_list.pop(ind_col_max)) # not replace but pop & insert, that is, the order is shifted.
                x_col_max = x[:,ind_col_max]
                x = np.delete(x,ind_col_max,axis=1)
                x = np.insert(x,i,x_col_max,axis=1)
                ind_i_max = ind_row_max
                x_i_max = x_row_max

        tmp = x[ind_i_max].copy()
        x[ind_i_max] = x[i]
        x[i] = tmp
        x[i,i:] = x[i,i:] / x_i_max
        for j in range(m):
            if not j == i:
                xji = x[j,i]
                x[j,:] = x[j,:] - xji * x[i,:]
    print(f"X is a full rank matrix ( pivot ( = {pivot})  >  tole2 ( = {tole2}) )")
    return x,order_list


def find_ints(x,tole1,tole2):
    # Find independent non-trivial solutions   
    m,n = x.shape
    rk_np = np.linalg.matrix_rank(x)   # for rank checking

    tole1,tole2 = get_tolerances(tole1,tole2,m)     
    
    x_rref,new_order = rref(x,tole1,tole2)
    
    rk_rref = 0
    for i in range(min(m,n)):
        if x_rref[i,i] == 1.:
            rk_rref += 1    
    print(f"Rank check: rk_np   = {rk_np}")
    print(f"                   rk_rref = {rk_rref}")
    ns = n - rk_rref
    print(f"# of non-trivial solutions: {ns} (= n - rk_rref)")
    print(f"Column order:\n {new_order}")
    
    c_up = -x_rref[:rk_rref,rk_rref:]
    c_eye = np.eye(ns,ns)
    c_mat = np.r_[c_up,c_eye]  # solution matrix
    return c_mat,new_order,rk_rref


def find_multicollinearity(x_df,normalize,tole1,tole2):
    labs = x_df.columns.tolist()
    x = x_df.values.copy()
    x = np.array(x,dtype="float64")
    x_norm,norm = st.normalize(x)  # In any case, descriptors are normalized once for increase of calculation accuracy
    m,n = x.shape
    print(f"Shape: X ({m}, {n})")
    
    c_mat,new_order,rk = find_ints(x_norm,tole1,tole2)
    
    base_list = new_order[:rk]
    extr_list = new_order[rk:]
    spac_list = extr_list.copy()
    spac_list.append(base_list)
    print("Space index: [ extra basis, [ basis ] ]")
    print(spac_list)
    
    # X' = XD :   X' (m,n) is a normalized descriptor matrix.
    #                 D (n,n) is a normalized operator.
    # X'_rref = RX'Q :   X'_rref (m,n) is a matrix with reduced row echelon form (rref).
    #                           R (m,m) is a elementary row operation.
    #                           Q (n,n) is a elementary column operation (supporse an orthogonal matrix).
    # X'_rrefC_rref = 0   : C_rref is a solution matrix.
    # (R^-1)X'_rref(Q^-1)QC_rref = 0,  then,  X'QC_rref = 0,
    # the solution of X'C' = 0 is given by QC_rref.
    # the solution of XC = 0 is given by DQC_rref.
    # In this program, the solutions for XQ or X'Q is calculated (instead of X or X').
    # Therefore, C_rref (for normalized coef.) or (Q^-1)DQC_rref (for original coef.) is obtained as shown below.
    
    if normalize:       # for normalized coefficients
        dc_mat_t = c_mat.T                              # C_rref^T :   solution vectors for X'Q.
    else:              # for original coefficients
        q_mat = q_matrix(new_order)               # Q
        q_mat_inv = q_mat.T                            # Q^-1  (Q^T because Q is orthogonal)
        qc_mat = np.dot(q_mat,c_mat)             # QC_rref
        d_mat = np.linalg.inv(np.diag(norm))     # D
        dqc_mat = np.dot(d_mat,qc_mat)         # DQC_rref
        dc_mat = np.dot(q_mat_inv,dqc_mat)   # (Q^-1)DQC_rref
        dc_mat_t = dc_mat.T                            # ( (Q^-1)DQC_rref)^T :   solution vectors for XQ.
    
    ns = n - rk
    # Print correlaiton form
    print("\nThe form of multi-correlated descriptors")
    for i in range(ns):
        y_lab = labs[new_order[rk+i]]
        x_labs = []
        for j in range(rk):
            x_lab = labs[new_order[j]]               # X'Q
            x_labs.append(x_lab)
        form = model_form(y_lab,x_labs,-dc_mat_t[i]/dc_mat_t[i,rk+i])
        print(f"{i+1} : {form}")
    print("")
    
    # Make subspace list
    find_subspace(labs,c_mat.T,new_order,rk)
    
    lid_list = []
    for bi in base_list:
        blab = labs[bi]
        lid_list.append(blab)
    print(f"Temporal linearly independent descriptors (LIDs): {rk}")
    print(f"{lid_list}\n")
    return lid_list
    
    
    
# === Hypervolume calculation ===
def hypervolume(x):       
    x_norm,norm = st.normalize(x)  
    vol = np.sqrt(np.linalg.det(np.dot(x_norm.T,x_norm)))
    return vol


def make_sub_df(df,sub_list):
    sub_df = pd.DataFrame(index=df.index)
    for sub in sub_list:
        sub_df[sub] = df[sub]
    return sub_df


def make_comb_list(can_list,k):
    comb_list = list(itertools.combinations(can_list,k))
    sub_list = []
    com_list = []
    for tup in comb_list:
        s_list = list(tup)
        c_list = []
        c_list.extend(can_list)
        for t in s_list:
            c_list.remove(t)
        sub_list.append(s_list)
        com_list.append(c_list)
    return sub_list,com_list


def hypervolume_comb(df,defined_list,candidate_list,k,ntop):
    if ntop == None:
        m = len(candidate_list)
        ntop = int(st.mCn(m,k))
    def_df = make_sub_df(df,defined_list)
    can_df = make_sub_df(df,candidate_list)
    use_list,unuse_list = make_comb_list(candidate_list,k)
    ncomb = len(use_list)
    vol_list = []
    for i in range(ncomb):
        use_df = make_sub_df(df,use_list[i])
        base_df = pd.concat([def_df,use_df],axis=1)
        base = base_df.values.copy()
        vol = hypervolume(base)
        vol_list.append([use_list[i],unuse_list[i],vol])
    vol_df = pd.DataFrame(vol_list,columns=["Used","Unused","Volume"])
    sorted_vol_df = vol_df.sort_values(by="Volume",ascending=False).reset_index()
    print(sorted_vol_df[:ntop],"\n")

    
    
# === SIS ===    
def _xty(x,y):
    m,n = x.shape
    x_norm,norm = st.normalize(x)
    xty = np.dot(x_norm.T,y)
    xty_list = []
    for i in range(n):
        xty_list.append([i,xty[i],np.abs(xty[i])])
    xty_ar = np.array(xty_list)
    sort_ind = np.argsort(xty_ar[:,2])
    return xty_ar[sort_ind[::-1],:]    


def xty(df,se,ntop):
    labs = df.columns.tolist()
    x = df.values.copy()
    y = se.values.copy()
    xty_ar = _xty(x,y)
    m,n = x.shape
    if ntop == None:
        ntop = n
    xty_list = []
    for i in range(n):
        lab = labs[int(xty_ar[i,0])]
        val = xty_ar[i,1]
        aval = xty_ar[i,2]
        xty_list.append([lab,val,aval])
    xty_df = pd.DataFrame(xty_list,columns=["Label","xty","|xty|"])
    print(xty_df[:ntop],"\n")
    