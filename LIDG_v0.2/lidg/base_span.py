import pandas as pd
import numpy as np

from lidg.label_calculator import LabelCalculator
import lidg.plots as plots


class BaseSpan:
    def __init__(self,spn_name="s"):
        self.spn_name = spn_name
        self.x = None
        self.x_save = None
        self.lcal = LabelCalculator()
        

    def read(self,file_or_df):
        if type(file_or_df) == str:
            print(f".read: text file")
            df = pd.read_table(file_or_df,delim_whitespace=True,index_col=0)
        else:
            print(f".read: DataFrame")
            df = file_or_df.copy()
        df = self.lcal.label_check(df)
        self.x = df.copy()
        self.x_save = df.copy()


    def write(self,fi,ff="%.16f"):
        print(f".write:")
        print(f"  {self.spn_name}.x   --->   '{fi}'\n")
        self.x.to_csv(fi,sep=" ",float_format=ff)
    
    
    def _existance_check(self,in_list,ref_list,exist=True):
        out_list = []
        if exist:
            for l in in_list:
                if l in ref_list:
                    out_list.append(l)
        else:
            for l in in_list:
                if not l in ref_list:
                    out_list.append(l)
        return out_list
    
    
    def _get_n(self):
        n = len(self.x.columns)
        return n
    
    
    def _get_n_save(self):
        n = len(self.x_save.columns)
        return n
    
    
    def show(self):
        n = self._get_n()
        print(f".show: {self.spn_name} ( {n} )")
        print(f"  {self.x.columns.tolist()}")
        for i,lab in enumerate(self.x.columns):
            if lab in self.lcal.label_dic:
                print(f"{i: >5}  {lab}      (  =  {self.lcal.label_dic[lab]}  )")
            else:
                print(f"{i: >5}  {lab}")  
        print()
        
        
    def show_save(self):
        n = self._get_n_save()
        print(f".show_save: {self.spn_name}_save ( {n} )") 
        print(f"  {self.x_save.columns.tolist()}") 
        for i,lab in enumerate(self.x_save.columns):
            if lab in self.lcal.label_dic:
                print(f"{i: >5}  {lab}      (  =  {self.lcal.label_dic[lab]}  )")
            else:
                print(f"{i: >5}  {lab}")     
        print()
        
        
    def show_all(self):
        self.show()
        self.show_save()
    
    
    def _base_print(self,method,na0,nb0,info=""):
        na1 = self._get_n()
        nb1 = self._get_n_save()
        s1 = self.spn_name
        print(f"{method}:  {info}")
        if method == ".save": 
            print(f"  {s1} ( {na0} )  --->  {s1}_save ( {nb0} )") 
            print(f"  {s1} ( {na1} )         {s1}_save ( {nb1} )\n") 
        elif method == ".load":
            print(f"  {s1} ( {na0} )   +   {s1}_save ( {nb0} )   in   {s1}_save ( {nb1} )") 
            print(f"  {s1} ( {na1} )        {s1}_save ( {nb1} )\n")
        elif method == ".remove":
            print(f"  {s1} ( {na0} )   -   {s1} ( {nb0} )") 
            print(f"  {s1} ( {na1} )\n") 
        elif method == ".add":
            print(f"  {s1} ( {na0} )   +   new ( {nb0} ) ") 
            print(f"  {s1} ( {na1} )\n")
        elif method == ".add_const":
            print(f"  {s1} ( {na0} )   +   const ( {nb0} ) ") 
            print(f"  {s1} ( {na1} )\n")
        elif method == ".add_rh":
            print(f"  {s1} ( {na0} )   +   random_h ( {nb0} ) ") 
            print(f"  {s1} ( {na1} )\n")
        elif method == ".add_rg":
            print(f"  {s1} ( {na0} )   +   random_g ( {nb0} ) ") 
            print(f"  {s1} ( {na1} )\n")
    
    
    def save(self):
        na = self._get_n()
        nb = self._get_n_save()
        self.x_save = self.x.copy()
        self._base_print(".save",na,nb)

    
    def load(self,load_list=None,pos=None):
        na = self._get_n()
        nb = self._get_n_save()
        if load_list == None:
            self.x = self.x_save.copy()
            self._base_print(".load",na,nb,info="all")
            return
        load_list = self.lcal._str2list(load_list)
        labs_list = self.x.columns.tolist()
        slabs_list = self.x_save.columns.tolist()
        nb0 = len(load_list)
        load_list = self._existance_check(load_list,slabs_list)
        load_list = self._existance_check(load_list,labs_list,False)
        nb = len(load_list)
        self.x[load_list] = self.x_save[load_list]
        if pos != None:
            head_list = labs_list[:pos]
            tail_list = labs_list[pos:]
            join_list = head_list + load_list + tail_list 
            self.x = self.x[join_list]
        self._base_print(".load",na,nb)
        
        
    def _get_inf(self):
        df = self.x.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=1)
        labs_list = self.x.columns.tolist()
        inf_list = list(sorted(set(self.x.columns) - set(df.columns),key=labs_list.index))
        return inf_list
        
        
    def _get_const(self):
        zero = 0.0000001 
        con_list = []
        for lab in self.x.columns:
            cstd = self.x[lab].std()
            if cstd <= zero:
                con_list.append(lab)
        return con_list
        

    def remove(self,rem_list=[]):
        na = self._get_n()
        labs_list = self.x.columns.tolist()
        info = None
        if type(rem_list) is str:
            info = rem_list
            if rem_list == "all":
                rem_list = labs_list
            elif rem_list == "inf":
                rem_list = self._get_inf()
                info = info + " " + str(rem_list)
            elif rem_list == "const":
                rem_list = self._get_const()
                info = info + " " + str(rem_list)
            else:
                rem_list = [rem_list]
        rem_list = self._existance_check(rem_list,labs_list)
        nb = len(rem_list)
        self.x = self.x.drop(rem_list,axis=1)
        self._base_print(".remove",na,nb,info)
    

    def add(self,add_list,pos=None):
        # add new descriptors from self.lcal.origin.
        na = self._get_n()
        labs_list = self.x.columns.tolist()
        add_list = self.lcal._str2list(add_list)
        add_list = self.lcal._remove_space(add_list)
        add_list = sorted(set(add_list), key=add_list.index)
        add_list = self._existance_check(add_list,labs_list,False)
        self.lcal._parentheses_check(add_list)
        
        nb = len(add_list)
        df = self.lcal.rabin2df(add_list)
        self.x[add_list] = df[add_list]
        if pos != None:
            head_list = labs_list[:pos]
            tail_list = labs_list[pos:]
            join_list = head_list + add_list + tail_list 
            self.x = self.x[join_list]
        self._base_print(".add",na,nb,add_list)
        
        
    def add_const(self):
        na = self._get_n()
        self.x.insert(0,"const",1.0)
        self.lcal.origin.insert(0,"const",1.0)
        self._base_print(".add_const",na,1)
        

    def add_rh(self,n=1,seed=100):
        # add homogeneous random data.
        na = self._get_n()
        m = len(self.x.index)
        np.random.seed(seed)
        r_arr = np.random.rand(m,n) * 2. - 1.
        r_list = []
        for i in range(n):
            self.x["rh"+str(i)] = r_arr[:,i] 
            self.lcal.origin["rh"+str(i)] = r_arr[:,i]
            r_list.append("rh"+str(i))
        self._base_print(".add_rh",na,n,r_list)


    def add_rg(self,n=1,sigma=1,seed=100):
        na = self._get_n()
        # add gaussian random data.
        m = len(self.x.index)
        np.random.seed(seed)
        r_arr = np.random.normal(0,sigma,(m,n))
        r_list = []
        for i in range(n):
            self.x["rg"+str(i)] = r_arr[:,i]
            self.lcal.origin["rg"+str(i)] = r_arr[:,i]
            r_list.append("rg"+str(i))
        self._base_print(".add_rg",na,n,r_list)
        

# === Labeling ===
    def change_label(self,lab,new_lab):
        print(f".change_label:")
        self.x = self.lcal.change_label(self.x,lab,new_lab)
        self.save()
        
    
    def change_labels(self,symbol="x",num=0):
        print(f".change_labels: from {symbol+str(num)}")
        self.x = self.lcal.change_labels(self.x,symbol,num)
        self.save()
        
        
    def show_labels(self):
        print(f".show_labels:")
        for i, (key,val) in enumerate(self.lcal.label_dic.items()): 
                    print(f"{i+1: >5}   {key}   <---   {val}")
        print()
        
        
    def restore_labels(self):
        print(".restore_labels:")
        labs_list = self.x.columns.tolist()
        for i,lab in enumerate(labs_list):
            res_lab = self.lcal._restore_label(lab)
            self.x.rename(columns={lab: res_lab},inplace=True)
            print(f"{i+1: >5}   {lab}   --->   {res_lab}")
        self.lcal.label_dic = {}
        self.save()    
       
    
# === Plotting ===
    def plot(self,df,se=None,xlab="Index",ylab="Value"):
        plots.preplot(df,se,xlab,ylab)      
    
    
    def vs_plot(self,x1,x2,pmin=None,pmax=None,xlab="y",ylab="hy"):
        plots.vs_plot(x1,x2,pmin,pmax,xlab,ylab)      
            
