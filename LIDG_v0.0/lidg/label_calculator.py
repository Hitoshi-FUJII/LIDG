import numpy as np
import pandas as pd

class LabelCalculator:
    def __init__(self):
        self.label_dic = {}
        self.origin = None
        
# ==============================================
# You can add your original operators in these dictionary 
#  (should be use single str, like, "s" : ( lambda x : np.sqrt(x) ).
# ==============================================

        self.op1_dic = {
            "r" : (lambda x : 1. / x),
            "a" : (lambda x : np.abs(x)),
            "b" : (lambda x : 1. / np.abs(x))}
    
# ==============================================  

        self.op2_dic = {
            "+" : (lambda x1, x2 : x1 + x2),
            "-"  : (lambda x1, x2 : x1 - x2),
            "*"  : (lambda x1, x2 : x1 * x2)}
        
    
    def _label_check(self,lab):
        if lab in self.op1_dic:
            raise LabelError(f"label '{lab}' is invalid:  {list(self.op1_dic.keys())}") 
        for op in self.op2_dic:
            if op in lab:
                raise LabelError(f"label '{lab}' is invalid:  '{op}' ")
        if "(" in lab:
            spl0 = lab.split("(")[0]
            if spl0 in self.op1_dic:
                raise LabelError(f"label '{lab}' is invalid:  '{spl0}' ") 
            
            
    def label_check(self,df):
        # initial label cannot have "+", "-", "*".
        # initial label should not have parentheses except for describing identifiers.
        # blanks (space) in the column names are removed.
        # converts the values of dataframe to numpy float64.
        new_df = df.copy()
        print("  .label_check\n")
        for lab_b in new_df.columns:
            lab = lab_b.replace(" ","")
            self._label_check(lab)
            new_df.rename(columns={lab_b: lab},inplace=True)
        new_df = new_df.astype(np.float64)
        self.origin = new_df.copy()
        return new_df
    

# === Labeling ===
    def change_label(self,df,lab,new_lab):
        new_df = df.copy()
        self._label_check(new_lab)
        if lab not in new_df.columns:
            raise LabelError(f"label '{lab}' is not in the span (.change_label)")
        if new_lab in self.label_dic or new_lab in new_df.columns:
            raise LabelError(f"label '{new_lab}' already exists (.change_label)")
        new_df.rename(columns={lab: new_lab},inplace=True)
        self.label_dic[new_lab] = lab
        print(f"  {new_lab}  <---  {lab}")
        return new_df
    
    
    def change_labels(self,df,symbol="x",num=0):
        new_df = df.copy()
        for i,lab in enumerate(new_df.columns):
            new_lab = symbol+str(i+num)
            if new_lab in self.label_dic or new_lab in new_df.columns:
                raise LabelError(f"label '{new_lab}' already exists (.change_labels)")
            new_df.rename(columns={lab: new_lab},inplace=True)
            self.label_dic[new_lab] = lab
            print(f"{i+1: >5}   {new_lab}   <---   {lab}")
        return new_df
           
    
# === for add method ===
    def _str2list(self,str_or_list):
        if type(str_or_list) == str:
            str_or_list = [str_or_list]
        return str_or_list    
        
        
    def _remove_space(self,slabs_list):
        labs_list = []
        for slab in slabs_list:
            slab = slab.replace(" ","")
            labs_list.append(slab)
        return labs_list


    def _remove_parentheses(self,par):
        if "()" in par or "(+)" in par or "++" in par:
            par = par.replace("()","")
            par = par.replace("(+)","")
            par = par.replace("++","+")
            par = self._remove_parentheses(par)
        return par
    

    def _parentheses_check(self,add_list):
        for lab in add_list:
            lab_str_list = list(lab)
            check = ""
            for s in lab_str_list:
                if s in ["(", ")", "+", "-"]:
                    check += s
            check = check.replace("-","+")
            check = self._remove_parentheses(check)
            if check == "+":
                raise ParenthesesError(f'need outer parentheses for + or - operations: "{lab}"') 
            elif check != "":
                check = check.replace("+","")
                raise ParenthesesError(f'given label (equation) is invalid:  "{check}"  in  "{lab}"')    
        
        
    def _label2list(self,lab):
        lab = lab.replace(" ","")
        op_list = ["("]
        for op1 in self.op1_dic:
            op_list.append(op1+"(")          # op_list = [(, r(, a(, b(]
            
        for op2 in self.op2_dic:
            lab = lab.replace(op2," "+op2+" ")
        lab = lab.replace(")"," ) ")
        lab = lab.replace("(","( ")       #   !!! not " ( " !!!
        lab_spl_list = lab.split()
        stack = []
        for spl in reversed(lab_spl_list):
            if "(" in spl:                             # (, r(, a(, b(, or des(
                if spl not in op_list:            #  des(
                    p1 = stack.pop()                 #  i,j,k,...
                    p2 = stack.pop()                 # )
                    spl += p1 + p2                   # des(i,j,k,...)
                elif spl == "(":                     # (
                    pass
                else:                                  # r(, a(, b(
                    spl = spl.replace("(","")    # r, a, b
                    stack.append("(")           # (
            stack.append(spl) 
        stack.reverse()
        return stack
        
        
    def _restore_label(self,llab):
        llab_spl_list = self._label2list(llab)
        res_lab = ""
        for spl in llab_spl_list:
            if spl in self.label_dic:
                spl = self.label_dic[spl]
                spl = self._restore_label(spl)
            res_lab +=  spl
        return res_lab
    
    
    def _restore_labels(self,llabs_list):
        llabs_list = self._str2list(llabs_list)
        res_labs_list = []
        for llab in llabs_list:
            res_lab = self._restore_label(llab)
            res_labs_list.append(res_lab)
        return res_labs_list
              
        
    def _reverse_shunting_yard(self,rab_lab_list):
        # Reversed shunting yard algorithm. (RAB_infix to prefix notation)
        # list ---> list
        pre_lab_list = []
        stack = []
        for spl in reversed(rab_lab_list):
            if spl == ")":
                stack.append(spl)
            elif spl == "*":
                stack.append(spl)
            elif spl == "+" or spl == "-":
                if len(stack) != 0:
                    if stack[-1] == "*":
                        op = stack.pop()
                        pre_lab_list.insert(0,op)
                stack.append(spl)
            elif spl == "(":
                for i in range(len(stack)):
                    op = stack.pop()
                    if op == ")":
                        break
                    pre_lab_list.insert(0,op)
                #pre_lab_list.insert(0,spl)
            else:
                pre_lab_list.insert(0,spl)
        for op in reversed(stack):
            pre_lab_list.insert(0,op)
        return pre_lab_list

    
    def _pre2arr(self,pre_lab_list):
        stack = []
        for spl in reversed(pre_lab_list):
            #if spl == "(":
            #    pass
            if spl in self.op1_dic:
                x = stack.pop()
                stack.append(self.op1_dic[spl](x))
            elif spl in self.op2_dic:
                x1 = stack.pop()
                x2 = stack.pop()
                stack.append(self.op2_dic[spl](x1,x2))
            elif spl in self.origin.columns:
                stack.append(np.array(self.origin[spl]))
            elif spl.replace(".","").isdecimal():
                stack.append(float(spl) * np.array([1.0]*len(self.origin.index)))
            else:
                raise LabelError(f" '{spl}' is invalid in prefix notation, {pre_lab_list} (._pre2list)") 
        return stack[0]
    
    
    def rabin2df(self,rab_labs_list): 
        # RAB infix notation to dataframe.
        new_df = pd.DataFrame(index=self.origin.index)
        rab_labs_list = self._str2list(rab_labs_list)   
        rab_labs_list = self._remove_space(rab_labs_list)
        res_rab_labs_list = self._restore_labels(rab_labs_list)
        
        for i,res_rab_lab in enumerate(res_rab_labs_list):
            res_rab_lab_list = self._label2list(res_rab_lab)
            pre_lab_list = self._reverse_shunting_yard(res_rab_lab_list)
            new_df[rab_labs_list[i]] = self._pre2arr(pre_lab_list)
        return new_df
       
    
# === Exception ===
class LabelError(Exception):
    def __init__(self,comme):
        self.comme = comme
    def __str__(self):
        return str(self.comme)
    
    
class ParenthesesError(Exception):
    def __init__(self,comme):
        self.comme = comme
    def __str__(self):
        return str(self.comme)
    

    
    
    
