# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 12:44:51 2017
Get data from PS filies
@author: Inspiron
"""
import numpy as np
import pandas as pd
import os
import csv

features = ["ro","RON", "MON", "OLF","ALK","AROMA","BNZ","KIS", \
            "TLL","MASS","METANOL","ETANOL","MTBE","ETBE","TAME","DIPE","TBS"]
X = []
Y = []
CLs = {}
            
def hello():
    print("PS data module ku")
    
def get_key(d, value):
    for k, v in d.items():
        if v == value:
            return k
#X преобразовать в цифру и заполнить пропуски
def to_float(x):
  
    try:
        return float(str(x).replace(",","."))
    except:
        return np.NAN

        
    
def data_processing(PS, fals = 1):
    global X,Y,CLs
     #Некоторая предобработка значений
    PS["name"] = PS["name"].str.lower()
    PS["provider"] = PS["provider"].str.lower()
    PS["name"] = PS["name"].str.strip()
    PS["provider"] = PS["provider"].str.strip()
    gr = PS.groupby(["name","provider"])
    PS_new = pd.DataFrame()
    Class_min_size = 20
    for i in gr:  
        if len(i[1]) > Class_min_size and i[0][0]!= "бензин" and i[0][1] != "заказчик":
            PS_new =  PS_new.append(i[1], ignore_index = True)
                

    if fals == 1:
        
        X = PS_new[features]
        N = len(PS_new)
        CLs = {}
    
        cl_num = 0
        for i in range(N):
            ind = PS_new.iloc[i][["name", "provider"]]
            list_ind =  list(ind.values)
            if list_ind not in CLs.values():
                cl_num = cl_num+1
                CLs[cl_num] = list_ind
           
    
    
        CLs_ch =  lambda a: get_key(CLs,list(a.values)) 
        Y = PS_new[["name","provider"]].apply(CLs_ch, axis = 1)
        Y.name = "class_num"
    
        Y = pd.DataFrame(Y)
         
        for col in features:    
            X[col] = X[col].apply(to_float)
        borders = {'ro':[650,780], 'RON':[70,130],'MON':[76,130],'OLF':[-1,30],'ALK':[30,90], \
                  'AROMA':[-1,56], 'BNZ': [0,6], 'KIS':[0,20],'TLL':[-1,20],'MASS':[-1,50], \
                  'METANOL':[-1,1.1],'ETANOL':[-1,4],'MTBE':[-1,20],'ETBE':[-1,2],'TAME':[-1,2],'DIPE':[-1,2],\
                  'TBS':[-1,2]}
        i = 0    
        for (k,v)  in borders.items():
            Crit = np.logical_and(X[k]>v[0],X[k]<v[1]) 
            if i == 0:
                Crit_o = Crit.copy()
                i = 1
            else:
                Crit_o = np.logical_and(Crit_o,Crit)
    #   
        X = X[Crit_o]
        Y = Y[Crit_o]
    #Заполнение пустых значений
        for cl in CLs:
            x = X[Y["class_num"] == cl]
            means = (np.max(x) - np.min(x))/2 +np.min(x)
            X[Y["class_num"] == cl] = X[Y["class_num"] == cl].fillna(means)
    if fals == 100:
        PS_fals = PS[PS == PS_new][features]
        X = PS_fals[features]
        Y = [1]
   
        
#%%        
def open_ps_2007():
    global X,Y,CLs
    file = "PS 2007_Р.xlsx"
    xfile = pd.ExcelFile(file)
    df_cols = xfile.parse(sheetname = "columns", header = 3)
    columns = df_cols.columns 
    print(len(columns))
    PS07 = xfile.parse(sheetname = "PetroSpec07", skiprows = [0,1,2,3,4], parse_cols = "A:W", names = columns )
    data_processing(PS)   
    
    
    return (X,Y,CLs)
#%%
def open_ps_2007_fals():
    global X,Y,CLs
    file = "PS 2007_Р.xlsx"
    xfile = pd.ExcelFile(file)
    df_cols = xfile.parse(sheetname = "columns", header = 3)
    columns = df_cols.columns   
   
    PS = xfile.parse(sheetname = "PetroSpec07", skiprows = [0,1,2], parse_cols = "A:V", names = columns )
    data_processing(PS,100)   
    
    
    return (X,Y)
#%%        
def open_ps_2009():
    global X,Y,CLs
    print("--------------------------------",os.getcwd())
    file = "PS 09.xls"
    xfile = pd.ExcelFile(file) 
    df_cols = xfile.parse(sheetname = "columns", header = 3)
    columns = df_cols.columns       
    PS = xfile.parse(sheetname = "PetroCpec 09", skiprows = [0,1,2], parse_cols = "A:W", names = columns )
    data_processing(PS)
     
    
    return (X,Y,CLs)
#%%
def ch_text(x):
    
    if type(x)  is str :
        x = x.lower()
        x = x.strip()
        x = x.replace(",",".")
    try: 
        x = pd.to_numeric(x)
    except Exception: 
        x = np.NaN
        
    return x


#%%
def create_PS_data():
    
    file = "PS 09.xls"
    xfile = pd.ExcelFile(file) 
    df_cols = xfile.parse(sheet_name = "columns", header = 3)
    columns = df_cols.columns       
    PS09 = xfile.parse(sheet_name = "PetroCpec 09", skiprows = [0,1,2], parse_cols = "A:W", names = columns )
    PS09 = PS09.drop("date",axis = 1)
    file = "PS 2007_Р.xlsx"
    xfile = pd.ExcelFile(file)
    df_cols = xfile.parse(sheet_name = "columns", header = 3)
    columns = df_cols.columns.append(pd.Index(["tmp"]))
    PS07 = xfile.parse(sheet_name = "PetroSpec07", skiprows = [0,1,2,3], parse_cols = "A:V", names = columns )
    PS07 = PS07.drop("tmp",axis = 1)
    #data_processing(PS)
    PS = PS09.append(PS07, ignore_index = True)
    PS["name"] = PS["name"].str.lower()
    PS["provider"] = PS["provider"].str.lower()
    PS["name"] = PS["name"].str.strip()
    PS["provider"] = PS["provider"].str.strip()
    gr = PS.groupby(["name","provider"])
    PS_new = pd.DataFrame()
    CLs = {}
    class_number = 0
    Class_min_size = 20
    
    Y = []
    for i in gr:  
        if len(i[1]) > Class_min_size and i[0][0]!= "бензин" and i[0][1] != "заказчик":
            next_data = i[1][features].applymap(ch_text).dropna()
            if next_data.shape[0] > 0:
                PS_new =  PS_new.append(next_data, ignore_index = True)
                class_number += 1
                CLs[class_number] = i[0]
                Y = np.hstack((Y,class_number * np.ones(next_data.shape[0])))
                
    
    #return  np.array(PS_new.values),np.array(Y,int), CLs#(X,Y,CLs)    
    return (PS_new.values,Y.reshape(Y.shape[0],1),CLs)
#%%    
if __name__ == "__main__":
    hello()
   
    #X,Y,CLs = create_PS_data()
    (X,Y,CLs) = create_PS_data()
    myfile = open("PS_X.csv","w")
    with myfile:
        writer = csv.writer(myfile)
        writer.writerows(X)
    myfile = open("PS_Y.csv","w")
    with myfile:
        writer = csv.writer(myfile)
        [writer.writerow(y) for y in Y]
    myfile = open("PS_CLs.csv","w")
    with myfile:
        writer = csv.writer(myfile)
        for i in CLs.items():
            writer.writerow(i)    
    myfile = open("PS_feature.csv","w")
    with myfile:
        writer = csv.writer(myfile)
        writer.writerow(features)        
   