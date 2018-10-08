import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

from matplotlib.colors import ListedColormap
from sklearn import datasets, linear_model, metrics
from sklearn.neighbors import LocalOutlierFactor, IsolationForest


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

clf = LocalOutlierFactor(n_neighbors = 20,contamination = 0.05 )
y_pred = clf.fit_predict(X)
X_scores = clf.negative_outlier_factor_


clf2 = IsolationForest( max_samples=100, random_state=rng)
clf2.fit(X)
y_pred2 = clf2.predict(X)
plt.figure(2)
f1 = 1
f2 = 3
plt.scatter(X[:,f1],X[:,f2], c = Y, cmap = plt.cm.Paired, s = 7, marker = "*",edgecolors='none')
plt.scatter(X[y_pred == -1][:,f1],X[y_pred == -1][:,f2], c = Y[y_pred == -1], cmap = plt.cm.Paired, s = 6, marker = "o",  edgecolors='r', facecolors = 'none')
plt.scatter(X[y_pred2 == -1][:,f1],X[y_pred2 == -1][:,f2], c = Y[y_pred2 == -1], cmap = plt.cm.Paired, s = 6, marker = "o",  edgecolors='b', facecolors = 'none')
plt.xlim(X[y_pred == 1][:,f1].min(),X[y_pred == 1][:,f1].max())
plt.ylim(X[y_pred == 1][:,f2].min(),X[y_pred == 1][:,f2].max())
#%%

nights = np.arange(3,50,2)
conts = np.arange(0.01,0.5,0.1)

res = np.zeros((len(nights),len(conts)))
n = 0
c = 0
for night in nights :
    for cont in conts:
        clf = LocalOutlierFactor(n_neighbors = night,contamination = cont )
        y_pred = clf.fit_predict(X)
        res[n,c] = len(y_pred[y_pred == -1])
        c +=1
    n += 1 
    c = 0
plt.plot(nights,res[:,1])        
#%%    
#Logistic Regression

multi_class = "multinomial"
clf = linear_model.LogisticRegression(solver ="sag", penalty= "l2", max_iter= 1000,random_state= 15, multi_class= multi_class).fit(X,Y)
print("training score : %.3f (%s)" % (clf.score(X, Y), multi_class))

#%%
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)


# fit the model
clf = IsolationForest( max_samples=100, random_state=rng)
clf.fit(X)
y_pred = clf.predict(X)
plt.scatter(X[y_pred == -1][:,f1],X[y_pred == -1][:,f2], c = Y[y_pred == -1], cmap = plt.cm.Paired, s = 33, marker = "o",  edgecolors='r', facecolors = 'none')

# plot the line, the samples, and the nearest vectors to the plane

xx, yy = np.meshgrid(np.linspace(X[:,f1].min(), X[:,f1].max(), 50), np.linspace(X[:,f2].min(), X[:,f2].max(), 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X[:, f1], X_train[:, f2], c='white',
                 s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((X[:,f1].min(), X[:,f2].min()))
plt.ylim(((X[:,f1].max()),X[:,f2].max()))
plt.legend([b1],
           ["training observations"],
           loc="upper left")
plt.show()
