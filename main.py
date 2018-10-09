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
from sklearn.ensemble import IsolationForest


np.random.seed(15)
INPUT_DATA = 2 # 1 - Model data
               # 2 - PetroSpec
               
#%% read data from file
if INPUT_DATA == 2:
    F_with_outlier = False
    if F_with_outlier:    
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
        
        
        clf = LocalOutlierFactor(n_neighbors = 20,contamination = 0.07 )
        y_pred = clf.fit_predict(X)
        X_scores = clf.negative_outlier_factor_
            

    else: 
        myfile = open("PS_X_without_outlier.csv","r")
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
            
        myfile = open("PS_Y_without_outlier.csv","r")
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
#Logistic Regression

multi_class = "multinomial"
clf = linear_model.LogisticRegression(solver ="sag", penalty= "l2", max_iter= 1000,random_state= 15, multi_class= multi_class).fit(X,Y)
print("training score : %.3f (%s)" % (clf.score(X, Y), multi_class))


   
##%%

#I save PS_X_outlier after LocalOutlierFactor with n_neighbors = 20,contamination = 0.07 
#n_neightbors doesnt influence the result, only contaminations
#IsolationForest не подходит. Возможно, потому что ему для обучения нужны правильные данные. 
#Наверое, если предварительную чистку делать LocalOutlierFactor, а потому на почищенных им данных настроить лес, то дальше можно использовать IsolatinForest модел на первом этапе работы алгоритма.
##OUTLIER detection process comparing
#clf = LocalOutlierFactor(n_neighbors = 20,contamination = 0.07 )
#y_pred = clf.fit_predict(X)
#X_scores = clf.negative_outlier_factor_
#
#
#clf2 = IsolationForest( max_samples=100, random_state=rng)
#clf2.fit(X)
#y_pred2 = clf2.predict(X)
#
#features_plot = ([0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[0,16])
#for f1,f2 in features_plot:
#    fig = plt.figure(2,figsize = (23,23))
#    plt.scatter(X[:,f1],X[:,f2], c = Y, cmap = plt.cm.Paired, s = 20, marker = "*",edgecolors='none')
#    plt.scatter(X[y_pred == -1][:,f1],X[y_pred == -1][:,f2], c = Y[y_pred == -1], cmap = plt.cm.Paired, s = 29, marker = "o",  edgecolors='r', facecolors = 'none')
#    #plt.scatter(X[y_pred2 == -1][:,f1],X[y_pred2 == -1][:,f2], c = Y[y_pred2 == -1], cmap = plt.cm.Paired, s = 25, marker = "o",  edgecolors='b', facecolors = 'none')
#    plt.xlim(X[y_pred == 1][:,f1].min(),X[y_pred == 1][:,f1].max())
#    plt.ylim(X[y_pred == 1][:,f2].min(),X[y_pred == 1][:,f2].max())
#    fig.savefig("outliert_detection_res07_{}_{}".format(f1,f2))
#
#X_new = X[y_pred == 1]
#Y_new = Y[y_pred == 1]
#
##save new
#Y_new =Y_new.reshape(Y_new.shape[0],1)
#myfile = open("PS_X_without_outlier.csv","w")
#with myfile:
#    writer = csv.writer(myfile)
#    writer.writerows(X_new)
#myfile = open("PS_Y_without_outlier.csv","w")
#with myfile:
#    writer = csv.writer(myfile)
#    [writer.writerow(y) for y in Y_new]

      

