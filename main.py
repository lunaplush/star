import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

from matplotlib.colors import ListedColormap
from sklearn import datasets, linear_model, metrics
from sklearn.neighbors import LocalOutlierFactor

%matplotlib inline

np.random.seed(15)
INPUT_DATA = 2 # 1 - Model data
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
    
    [X,Y] = datasets.make_blobs(n_samples = 500, n_features = 2, centers = 4, cluster_std = [0.2,0.2,0.3,0.15,0.15], random_state = 58)
    
#%%
plt.figure(1)
plt.scatter(X[:,2],X[:,10], c = Y, cmap = plt.cm.Paired, s = 13)


#%%

clf = LocalOutlierFactor(n_neighbors = 20,contamination = 0.1 )
y_pred = clf.fit_predict(X)
X_scores = clf.negative_outlier_factor_
plt.figure(2)

plt.scatter(X[:,2],X[:,10], c = Y, cmap = plt.cm.Paired, s = 3)

plt.scatter(X[y_pred == 1][:,2],X[y_pred == 1][:,10], c = Y[y_pred == 1], cmap = plt.cm.Paired, s = 33, marker = "*")

#%%
    
#Logistic Regression

multi_class = "multinomial"
clf = linear_model.LogisticRegression(solver ="sag", penalty= "l2", max_iter= 1000,random_state= 15, multi_class= multi_class).fit(X,Y)
print("training score : %.3f (%s)" % (clf.score(X, Y), multi_class))


