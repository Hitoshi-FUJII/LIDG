import numpy as np
import pandas as pd
import math
import lidg.statistics as st
import time
import lidg.model_selection.plot as pt


# --- For binary permutation problem ---

class GeneticAlgorithm:
    def __init__(self, x_df, y_se, n_len, n_one, score, n_ktop, 
                 n_gen, nmax_pop, n_tou, p_gap, p_crs, p_mut, n_mut):

        self.labs       = x_df.columns.tolist()
        self.x          = x_df.values.copy()
        self.y          = y_se.values.copy()

        self.n_len      = n_len        # the length of gene
        self.n_one      = n_one        # the numper of "1" in the gene
        #self.func       = func         # score calculator
        self.max_flag   = False        # maximum search problem or not
        #self.sw_size    = sw_size      # scale window size (for fitness calculation)
        self.score      = score
        self.n_ktop     = n_ktop       # keep top (in the global ranking) size 
        
        # --- ga parameters ---
        self.n_gen      = n_gen        # the number of generations
        self.nmax_pop   = nmax_pop     # the maximum number of popurations in each generations
        self.n_tou      = n_tou        # tournament size for selection (large: strong selection)
        self.p_gap      = p_gap        # generation gap: (1-p_gap)*nmax_pop is the number of eliets
        self.p_crs      = p_crs        # crossover ratio
        self.p_mut      = p_mut        # mutation ratio
        self.n_mut      = n_mut        # strength of the mutation (the number of swapping)
        
        if score == "e2":
            self.r_or_q = "R2"
        else:
            self.r_or_q = "Q2"

        ncomb = int(st.mCn(self.n_len,self.n_one))
        print(f"Genetic Algorithm for {self.n_one} descriptors:  {self.n_len}C{self.n_one} = {ncomb}")
        
        
    def search(self):
        t0 = time.time()
        # --- make initial individuals ---
        init_indivs_lis = self.make_individuals()
        
        # --- remove duplications and make dictionary ---
        indivs_dic = self.make_dic(init_indivs_lis)
        
        
        appeared_dic = {}       # already calculated individuals
        b_indivs = []           # best individuals in each generation
        b_scores = []           # best scroes in each generation
        w_scores = []           # worst scroes
        a_scores = []           # average scores
        
        print(self.labs)
        print(f"#gen  #cal  ave_score  best_score   best_individual")
        for i in range(self.n_gen):
            # --- Score calculation (only for new individuals) ---
            for indiv in indivs_dic.keys():
                if indiv in appeared_dic.keys():
                    indivs_dic[indiv] = appeared_dic[indiv]
                else:
                    if self.score=="e2":
                        indivs_dic[indiv],_ = self.func(indiv)
                    else:
                        _,indivs_dic[indiv] = self.func(indiv)
                    
            appeared_dic.update(indivs_dic)    
                
            b_score,b_indiv = self.get_best(indivs_dic,best_flag=True)
            w_score,w_indiv = self.get_best(indivs_dic,best_flag=False)
            a_score = np.mean(list(indivs_dic.values())) 

            b_indivs.append(b_indiv)
            b_scores.append(b_score)
            w_scores.append(w_score)
            a_scores.append(a_score)            
            
            # --- fitness calculation ---
            #fitness_dic = self.get_fitness1(indivs_dic,w_scores)
            fitness_dic = self.get_fitness2(indivs_dic,b_score,w_score)
            indivs_lis = list(fitness_dic.keys())
            fitness_lis = list(fitness_dic.values())
            
            # --- selection ---
            selected_indivs = self.selection(indivs_lis,fitness_lis)     # based on the fitness
            elite_indivs = self.elite_selection(indivs_lis,fitness_lis)  # based on the fitness
            
            # --- genetic operations ---
            # crossover
            n_sel = len(selected_indivs)
            n_crs = int(0.5*n_sel)
            child_indivs = []
            for j in range(n_crs):
                p1, p2 = np.random.choice(n_sel,2,replace=False)
                if np.random.rand() < self.p_crs:
                    crs1, crs2 = self.crossover(selected_indivs[p1],selected_indivs[p2])
                else:
                    crs1, crs2 = selected_indivs[p1], selected_indivs[p2]
                child_indivs.extend([crs1,crs2])
            # mutation
            n_mu = int(self.p_mut * n_sel)
            for j in range(n_mu):
                child_indivs[j] = self.mutation(child_indivs[j])
            
            child_indivs.extend(elite_indivs)
            indivs_dic = self.make_dic(child_indivs)
            s_indiv = str(b_indiv)
            s_indiv = s_indiv.replace(", ","")
            print(f"{i: >4}  {len(appeared_dic): >4}   {a_score:6g}   {b_score:.6g}   {s_indiv}")
        
        print()    
        print("Genetic algorithm is finished.")    
        print() 
        t1 = time.time()
        print(f"Elapsed time: {t1-t0} [s]")
        print()  
        des_top_lis,scores_top_lis,cmb_lis = self.get_ktop(appeared_dic)
        #return indivs_top_lis, scores_top_lis, len(appeared_dic),i
     
        df = pd.DataFrame(index=[i for i in range(self.n_ktop)])
        df[self.score] = scores_top_lis
        syy = np.sum((self.y-np.mean(self.y))**2)
        df[self.r_or_q] = [1.-scr/syy for scr in scores_top_lis]
        df["Descriptors"] = des_top_lis
        print(f"Displayed only top {self.n_ktop} (sorted by {self.score})")
        print(df)

        pt.ex_plot(self.x,self.y,cmb_lis,self.r_or_q)
        return des_top_lis
        
    
    def make_individuals(self,n=None):
        if n is None: n = self.nmax_pop
        base_lis = [0]*(self.n_len - self.n_one) + [1]*self.n_one
        indivs_lis = []
        for i in range(n):
            indiv = list(np.random.permutation(base_lis))
            indivs_lis.append(indiv)
        return indivs_lis


    def make_dic(self,lis_or_tup):
        dic = {}
        for l in lis_or_tup:
            dic[tuple(l)] = 0
        return dic
   
    
    def get_best(self,indivs_dic,best_flag=True):
        if self.max_flag is best_flag:
            score = np.max(list(indivs_dic.values()))
        else:
            score = np.min(list(indivs_dic.values()))
        indiv = [k for k, v in indivs_dic.items() if v == score][0]
        return score,indiv
    

    def get_fitness2(self,s_dic,b_score,w_score):
        f_dic = s_dic.copy() 
        for k,v in s_dic.items():
            f_dic[k] = (v - w_score)/(b_score - w_score)
        return f_dic
    
    
    #def get_fitness1(self,s_dic,w_scores):
    #    # use scale window method
    #    f_dic = s_dic.copy() 
    #    u_lis = w_scores[::-1][:self.sw_size]
    #    if self.max_flag:
    #        u_lis.append(min(s_dic.values()))
    #        offset = min(u_lis)
    #        pm = 1
    #    else:
    #        u_lis.append(max(s_dic.values()))
    #        offset = max(u_lis)
    #        pm = -1
    #    for k,v in s_dic.items():
    #        f_dic[k] = pm * (v - offset)
    #    return f_dic
            
    
    def selection(self,indivs_lis,fitness_lis):
        # tournament selection
        n_sel = 2 * int(0.5 * self.p_gap * self.nmax_pop)  # becomes an even number
        n_indivs = len(indivs_lis)
        selected_indivs = []
        for i in range(n_sel):
            cand_ind = np.random.choice(range(n_indivs),self.n_tou,replace=True)
            cand_val = [fitness_lis[i] for i in cand_ind]       
            max_ind = np.argmax(cand_val) # best fitness is always the "maximum" fitness
            selected_indivs.append(indivs_lis[cand_ind[max_ind]])
        return selected_indivs
        
        
    def elite_selection(self,indivs_lis,fitness_lis):
        n_eli = self.nmax_pop - 2 * int(0.5 * self.p_gap * self.nmax_pop)
        sorted_index = np.array(fitness_lis).argsort()[::-1]
        elite_indivs = []
        for ind in sorted_index[:n_eli]:
            elite_indivs.append(indivs_lis[ind])
        return elite_indivs


    def crossover(self,p1,p2):
        # a homogeneous crossover operation for binary permutaion problem
        c1 = np.array(p1)
        c2 = np.array(p2)
        dif = c1 - c2
        dif_ind = np.where( dif != 0 )[0]
        new_dif_ind = np.random.permutation(dif_ind)
        for old,new in zip(dif_ind,new_dif_ind):
            if dif[new] == 1:
                c1[old],c2[old] = 1,0
            else:
                c1[old],c2[old] = 0,1
        return tuple(c1),tuple(c2)


    def mutation(self,p):
        # repeat a swap operation n_mut times
        c = list(p).copy()
        order = range(len(c))
        for i in range(self.n_mut):
            ind1,ind2 = np.random.choice(order,2,replace=False)
            c[ind1],c[ind2] = c[ind2],c[ind1]
        return tuple(c)

    
    def get_ktop(self,appeared_dic):
        k_lis,v_lis = list(appeared_dic.keys()),list(appeared_dic.values())
        if self.max_flag:
            index = np.array(v_lis).argsort()[::-1]
        else:
            index = np.array(v_lis).argsort()
        ktop_indivs,ktop_scores = [],[]
        for i in index[:self.n_ktop]:
            ktop_indivs.append(k_lis[i])
            ktop_scores.append(v_lis[i])

        ktop_des_lis = []
        cmb_lis = []
        for indiv in ktop_indivs:
            des = []
            cmb = []
            for i,j in enumerate(indiv):
                if j == 1:
                    des.append(self.labs[i])
                    cmb.append(i)
            ktop_des_lis.append(des)  
            cmb_lis.append(cmb)  
        return ktop_des_lis,ktop_scores,cmb_lis


# === Score calculation ===
    def func(self,indiv):   
        ind_lis = []
        for i,j in enumerate(indiv):
            if j == 1:
                ind_lis.append(i) 
 
        q,r = np.linalg.qr(self.x[:,ind_lis],mode="reduced")
        qty = np.dot(q.T,self.y)
        e = self.y - np.dot(q,qty)
        q_adm = q*q
        lev = np.sum(q_adm,axis=1) # leverage (diagonal part of hat matrix)
        eq = e / (1. - lev)
        e2 = np.dot(e,e)
        eq2 = np.dot(eq,eq)
        return e2,eq2

