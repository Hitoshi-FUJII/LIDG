import numpy as np
import pandas as pd


class RelationalDataBase():
    def __init__(self):
        self._dic = {}
        self._label_dic = {}
        self.merge_dic = {}
        
        
    def _read_as_list(self,file_name):
        l = []
        with open(file_name,"r") as f:
            for line in f:
                row = line.split()
                l.append(row)
        return l

    
    def _make_label(self,lab,idns,ids):
        label = str(lab)+"("
        for i in idns:
            label += str(ids[i]) + ", "
        label += ",)"
        label = label.replace(", ,","")
        return label
    

    def _divide_label(self,label):
        label = label.replace(" ","")
        label = label.replace(")","")
        lab,ids = label.split("(")
        ids_list = ids.split(",")
        return lab,ids_list
    
    
    def read(self,file_name,*idns_tup,**labs_dic):
        lis = self._read_as_list(file_name)
        print(file_name)
        for lab,val in labs_dic.items():
            self._dic[lab] = {}
            label = self._make_label(lab,idns_tup,lis[0])
            print(label+"  <---  "+str(lis[0]))
            self._label_dic[lab] = label
        print()
        for row in lis[1:]:
            ids_list = []
            for i in idns_tup:
                ids_list.append(row[i])
            ids_tup = tuple(ids_list)
            for lab,val in labs_dic.items():
                self._dic[lab][ids_tup] = float(row[val])
                
    
    def show(self):
        print(".show:")
        for i,(lab,val) in enumerate(self._dic.items()):
            print(str(i)+"  "+lab+"   ("+str(len(val))+")")
        print()
    
    
    def show_label(self):
        print(".show_label:")
        for i,(lab,val) in enumerate(self._label_dic.items()):
            print(str(i)+" "+lab+"  <---  "+str(val))
        print()
        
        
    def divide_dic(self,y_lab,y_ide):
        dic = self._dic[y_lab]
        y_label =  self._label_dic[y_lab]
        y_ides_list = self._divide_label(y_label)[1]
        idn = y_ides_list.index(y_ide)
        ides_list = []
        for ides_tup in dic.keys():
            ides_list.append(ides_tup[idn])
        ides_set = sorted(set(ides_list), key=ides_list.index)
        for ide in ides_set:
            self._dic[y_lab+"_"+ide] = {}
            new_y_label = y_label.replace(y_ide,y_ide+"="+ide)
            self._label_dic[y_lab+"_"+ide] = new_y_label
        for ides_tup,val in dic.items():
            ide = ides_tup[idn]
            self._dic[y_lab+"_"+ide][ides_tup] = val    
        
        
    def merge(self,lab1,labels_list):
        for label in labels_list:
            lab,ides_list = self._divide_label(label)
            self.merge_dic[label] = {}
            for pk_tup in self._dic[lab1].keys():
                pk_list =  []
                for i in ides_list:
                    pk_list.append(pk_tup[int(i)])
                for fk_tup in self._dic[lab].keys():
                    if pk_list == list(fk_tup):
                        self.merge_dic[label][pk_tup] =self._dic[lab][fk_tup]
        merge_df = self.df(self.merge_dic)
        return merge_df
        

    def df(self,dic=None):
        if dic == None:
            dic = self.merge_dic
        col_list = list(dic.keys())
        ind_tups_list = list(dic[col_list[0]].keys())
        
        ind_list = []
        for ind_tup in ind_tups_list:
            ind = str(ind_tup)
            ind = ind.replace(" ","")
            ind = ind.replace("'","")    #  remove quotations
            ind_list.append(ind)

        dic_df = pd.DataFrame(index=ind_list)

        for lab in dic.keys():
            val_list = []
            for val in dic[lab].values():
                val_list.append(val)
            dic_df[lab] = val_list
        return dic_df
        
    
