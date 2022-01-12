import pandas as pd
import numpy as np
import itertools

from lidg.base_span import BaseSpan
from lidg.projection import Projection
import lidg.multicollinearity as mcl
import lidg.statistics

from lidg.model_selection.elastic_net import elastic_net
from lidg.model_selection.exhaustive_search import exhaustive_search 
from lidg.model_selection.genetic_algorithm import GeneticAlgorithm


class LinearSpan(BaseSpan):
    def __init__(self,spn_name="s"):
        super().__init__(spn_name)
        self.y = None
        self.hy = None
        #self.py = None
        
        self.pro = Projection()
        
        
    def __repr__(self):
        n = self._get_n()
        return f"{self.spn_name} ( {n} ) : {self.x.columns.tolist()}"
    

# === basic methods ===
    def _print(self,method,na0,nb0,s0=None,info="",s2=None):
        na1 = self._get_n()
        nb1 = self._get_n_save()
        s1 = self.spn_name
        print(f"{method}:  {info}")
        if method == ".set_y":
            print(f"  {s1}.y     <---  '{s0}'  in  {s1} ( {na0} )")
            print(f"  {s1} ( {na0} )   -   '{s0}' ")
            print(f"  {s1} ( {na1} )\n")
        elif method == ".new":
            print(f"  {s0} ( {na0} )  <---  {s1} ( {nb0} )\n")
        elif method == ".sub":
            print(f"  {s0} ( {nb0} )  <---  {s1} ( {nb0} )  in  {s1} ( {na1} )\n")
        elif method == ".join":
            print(f"  {s2} ({na1+nb0})  <---  {s1} ( {na0} )   +   {s0} ( {nb0} )\n") 
        elif method == ".gen_bo": 
            print(f"  {s0} ( {nb0} )  <---  {s1} ( {na0} )\n")
        elif method == ".gen_dp": 
            print(f"  {s0} ( {nb0} )  <---  {s1} ( {na0} )\n")
        elif method == ".sym":
            print(f"  {s0} ( {nb0} )  <---  {s1} ( {na0} )\n") 
        elif method == ".lid":
            print(f"  {s0} ( {nb0} )  <---  {s1} ( {na0} )\n") 


    def set_y(self,target):
        if target in self.x.columns:
            na = self._get_n()
            self.y = self.x[target]
            self.x = self.x.drop([target],axis=1)
            self.lcal.origin = self.lcal.origin.drop([target],axis=1)
            self._print(".set_y",na,1,target)
        else:
            raise LabelError(f"there is no target '{target}' in {self.spn_name}")

        
    def new(self,spn_name="s_new",df=None,renew=True):
        # this method renews the label of DF for fast calculaiton.
        spn = LinearSpan(spn_name)
        if df is None:
            spn.x = self.x.copy()
            na = self._get_n()
            self._print(".new",na,na,spn_name)
        else:
            spn.x = df.copy()
        spn.x_save = spn.x.copy()
        if renew: 
            spn.lcal.origin = spn.x.copy()
        else:
            spn.lcal.origin = self.lcal.origin.copy()
            spn.lcal.label_dic = self.lcal.label_dic.copy()
        spn.y = self.y.copy()
        return spn
    
    
    def sub(self,sub_list,spn_name="s_sub"):
        na = self._get_n()
        labs_list = self.x.columns.tolist()
        sub_list = self.lcal._str2list(sub_list)
        sub_list = self._existance_check(sub_list,labs_list)
        nb = len(sub_list)
        spn = self.new(spn_name,self.x[sub_list],renew=False)
        self._print(".sub",na,nb,spn_name,info=sub_list)
        return spn
    
    
    def join(self,spn,spn_name="s_join"):
        na = self._get_n()
        nb0 = spn._get_n()
        labs_list = self.x.columns.tolist()
        slabs_list = spn.x.columns.tolist()
        slabs_list = self._existance_check(slabs_list,labs_list,False)
        nb = len(slabs_list)
        if nb != nb0:
            print(f"{nb0-nb} descriptors already exist. (.join)")
        df = pd.concat([self.x,spn.x[slabs_list]],axis=1)
        spn_new = self.new(spn_name,df,renew=False)
        spn_new.lcal.label_dic.update(spn.lcal.label_dic)
        self._print(".join",na,nb,s0=spn.spn_name,info="",s2=spn_name)
        return spn_new
        
        
# === Descriptor generation ===
    def gen_bo(self,op_list,spn_name="s_gen_bo"):
        print(f"Descriptor generation by basic operations:")
        na = self._get_n()
        labs_list = self.x.columns.tolist() 
        cnt = 0
        gen_list = []
        op_list = self.lcal._str2list(op_list)
        for op in op_list:
            op = op.replace(" ","")
            if op in self.lcal.op1_dic:
                print(f"  '{op}' : n = {na}")
                cnt+=na
                for lab in labs_list:
                    lab = op+"("+lab+")"
                    gen_list.append(lab)
            elif "+" in op or "-" in op or "*" in op:
                opl = list(op)
                if len(opl) ==1:
                    opl.insert(0,"")
                elif opl[0] not in self.lcal.op1_dic:
                    raise OperatorError(f"invalid operator '{op}' (.gen_bo)")
                combs = list(itertools.combinations(range(na),2))
                ncomb = len(combs)
                print(f"  '{op}' : {na}C2 = {ncomb}")
                cnt+=ncomb
                for comb in combs:
                    lab = opl[0] + "("+labs_list[comb[0]]+opl[1]+labs_list[comb[1]]+")"
                    gen_list.append(lab)
            else:
                raise OperatorError(f"invalid operator '{op}' (.gen_bo)")
        print(f"  # of generated new descriptors: {cnt}")
        df = self.lcal.rabin2df(gen_list)
        spn = self.new(spn_name,df,renew=False)
        nb = spn._get_n()
        self._print(".gen_bo",na,nb,spn_name,info=op_list)
        return spn
                    
                    
    def gen_dp(self,k=2,spn_name="s_gen_dp"):
        print(f"Descriptor generation by direct product:")
        const_list = self._get_const()
        x_df = self.x.copy()
        x_df = x_df.drop(const_list,axis=1)
        print(f"  Constant descriptors {const_list} are ignored.")
        
        labs_list = x_df.columns.tolist()
        na = len(labs_list)
        gen_list = []
        if type(k) is not int or k < 2: 
            raise OperatorError(f"integer operator should be greater than 1: '{k}' (.gen_dp)")
        combs = list(itertools.combinations_with_replacement(range(na),k))
        for comb in combs:
            lab = labs_list[comb[0]]
            for i in range(1,k):
                lab += "*"+labs_list[comb[i]]
            gen_list.append(lab)
        print(f"  # of generated new descriptors: {na}H{k} = {na+k-1}C{k} = {len(combs)}")
        df = self.lcal.rabin2df(gen_list)
        spn = self.new(spn_name,df,renew=False)
        nb = spn._get_n()
        self._print(".gen_dp",na,nb,spn_name,info="k="+str(k))
        return spn
    
        
        
# === Descriptor symmetrization ===
    def _sym_op(self,lab,sop_tup):
        sop_lab = lab.replace(" ","")
        for i,j in enumerate(sop_tup):
            sop_lab = sop_lab.replace("("+str(i)+")","(_"+str(j)+"_)")
            sop_lab = sop_lab.replace("("+str(i)+",","(_"+str(j)+"_,")
            sop_lab = sop_lab.replace(","+str(i)+")",",_"+str(j)+"_)")
            sop_lab = sop_lab.replace(","+str(i)+",",",_"+str(j)+"_,")
        return sop_lab.replace("_","")
        
        
    def _group_check(self,sop_tups):
        for sop1 in sop_tups:
            sop1_str = str(sop1)
            sop1_list = []
            sop2_list = []
            for sop2 in sop_tups:
                sop2_str = str(sop2)
                sop2_str = sop2_str.replace(" ","")
                sym_sop = self._sym_op(sop1_str,sop2)
                sop1_list.append(sym_sop)
                sop2_list.append(sop2_str)
            if set(sop1_list) != set(sop2_list):
                raise OperatorError(f"the symmetry operation set is not a group: {sop_tups}")
        
        
    def sym(self,sop_tups,spn_name="s_sym"):
        print(f"Descriptor symmetrization:")
        self._group_check(sop_tups)
        na = self._get_n() 
        n = len(sop_tups)
        sym_labs_list = []
        for lab in self.x.columns:
            sym_lab = self._sym_op(lab,sop_tups[0])
            for i in range(1,n):
                sop_lab = self._sym_op(lab,sop_tups[i])
                sym_lab += "+" + sop_lab
            sym_labs_list.append(sym_lab)
        df = self.lcal.rabin2df(sym_labs_list)
        spn = self.new(spn_name,df,renew=False)
        nb = spn._get_n()
        self._print(".sym",na,nb,spn_name,info=sop_tups)
        return spn

    
# === Multicollinearity ===    
    def lid(self,spn_name="s_lid",normalize=False,tole1=None,tole2=None):         
        print(f"Find and remove multicollinearities:")
        print(f"  normalize = {normalize}")
        if normalize: print(f"  coefficients are for normalized descriptors (for a fair comparison of weights)\n")
        else: print(f"  coefficients are for descriptors with original scale (to obtain exact MCL relationships)\n")
        na = self._get_n()
        lid_list = mcl.find_multicollinearity(self.x,normalize,tole1,tole2)
        df = self.x[lid_list]
        spn = self.new(spn_name,df,renew=False)
        nb = spn._get_n()
        self._print(".lid",na,nb,spn_name)
        return spn
    

    def volume_comb(self,defi_list,cand_list,k,ntop=None):
        mcl.hypervolume_comb(self.x,defi_list,cand_list,k,ntop)
              
            
    def xty(self,ntop=None):
        mcl.xty(self.x,self.y,ntop)

        
        
# === Linear regresson and finding near multicollinearities) ===
    def project(self,sort=None,normalize=False,n_round=3):
        print(f"Ordinary Least Squares regression:")
        print(f"  normalize = {normalize}")
        if normalize: print(f"  coefficients are for normalized descriptors (for a fair comparison of weights)\n")
        else: print(f"  coefficients are for descriptors with original scale\n")
        self.pro.vector_projection(self.x,self.y,sort,normalize,n_round)
        self.hy = self.pro.hy
        form = mcl.model_form(self.hy.name,self.x.columns,self.pro.beta,n_round)
        print(f"\n{form}\n")
        y_tmp_se = self.y.copy()
        y_tmp_se.name = "y"
        self.plot(y_tmp_se,self.hy)
        self.vs_plot(self.y,self.hy,xlab="y",ylab="hy")
        
        
        
    def rij(self,ntop=None):
        const_list = self._get_const()
        x_df = self.x.copy()
        x_df = x_df.drop(const_list,axis=1)
        print(f"Constant descriptors {const_list} are ignored in rij calculation.")
        self.pro.rij(x_df,ntop)  
        print()
    
    
    def nij(self,ntop=None):
        self.pro.nij(self.x,ntop)  
        print()
        
    def gij(self,ntop=None):
        self.pro.gij(self.x,ntop)  
        print()
    
    
    def corr(self,ntop=None):
        self.rij(ntop)
        self.nij(ntop)
        self.gij(ntop)
      
    
    def gik(self,lab_k=None,ntop=None):
        if lab_k is None:
            raise LabelError("Please give a descriptor label.")
        self.pro.gik(self.x,lab_k,ntop)  
        print()
    
        
    def outlier(self,ntop=10):
        self.pro.outlier(self.x,self.y,ntop)
        print()
        
        
        
# === Model selection ===        
    def eln_search(self,l1r=1.0,eps=0.001,kf=20,intercept=True,norm=False,itr=1000,zero=0.001):
        #l1r = 0.01: Ridge, 1.0: LASSO
        print("Elastic net calculation (by sklearn)")
        const_list = self._get_const()
        x_df = self.x.copy()
        x_df = x_df.drop(const_list,axis=1)
        print(f"Constant descriptors {const_list} are ignored.")
        elastic_net(x_df,self.y,l1r,eps,kf,intercept,norm,itr,zero)
        
        
    def exh_search(self,k,ntop=20,const=False):
        e_des,q_des = exhaustive_search(self.x,self.y,k,ntop,const)
        return e_des,q_des
 
    def ga_search(self,k,score="e2",ntop=20,
                  n_gen=30,nmax_pop=100,n_tou=5,p_gap=0.9,p_crs=0.9,p_mut=0.2,n_mut=1): 
        ga = GeneticAlgorithm(self.x, self.y, len(self.x.columns), k, score, ntop,
                              n_gen, nmax_pop, n_tou, p_gap, p_crs, p_mut, n_mut) 
        des = ga.search()
        return des
        
# === Exception ===
class LabelError(Exception):
    def __init__(self,comme):
        self.comme = comme
    def __str__(self):
        return str(self.comme)
    
    
class OperatorError(Exception):
    def __init__(self,comme):
        self.comme = comme
    def __str__(self):
        return str(self.comme)
    
