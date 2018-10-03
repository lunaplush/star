import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

from matplotlib.colors import ListedColormap
from sklearn import cross_validation, datasets, linear_model, metrics


np.random.seed(15)
INPUT_DATA = 1 # 1 - Model data
               # 2 - PetroSpec
               
#%% read data from file
if INPUT_DATA == 2:
        
    myfile = open("PS_X.csv","r")
    with myfile:
        reader = csv.reader(myfile)
        i = -1 
        for row in reader:
            
            if  i == -1 : 
                X = np.array([row], float)
                i = 1
            elif i == 1:
                i = 0
            else:
                X = np.insert(X,X.shape[0],row, axis = 0)
                i = 1
        
    myfile = open("PS_Y.csv","r")
    with myfile:
        reader = csv.reader(myfile)
        i = -1 
        for row in reader:
           
            if  i == -1 : 
                Y = np.array([row],float)
                i = 1
            elif i == 1:
                i = 0
            else:
                Y = np.insert(Y,Y.shape[0],row, axis = 0)
                i = 1
    Y = np.array(Y,int).reshape(Y.shape[0],)
#%% model data
if INPUT_DATA == 1:
    
    [X,Y] = datasets.make_blobs(n_samples = 500, n_features = 5, centers = 10, cluster_std = [1.0,2,4,5.5,0.5], random_state = 5)
    
#%%

#Logistic Regression
    
    