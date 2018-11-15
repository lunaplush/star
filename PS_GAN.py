# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:08:15 2018

@author: Luna
"""

import torch
import torch.utils.data as torchdata
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix




from matplotlib.colors import ListedColormap
from sklearn import datasets, linear_model, metrics
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from IPython import display

torch.set_num_threads(4)
#%%

# ===========================
# IMPORTANT PARAMETER:
# Number of D updates per G update
# ===========================
# k_d, k_g = 1, 1
PARAMETERS = {"lr":0.001, "momentum":0.5,"epochs": 1000, "batchsize": 64, "batchsize_test": 500, "noise_dim": 20,"k_d":1, "k_g":1, "f1":0,"f2":1}




#%%
def sample_noise(N):
    return np.random.normal(size=(N,PARAMETERS["noise_dim"])).astype(np.float32)

#%%
#VISUALISATION FUNCTIONS
   
import time
np.random.seed(12345)
lims = (-3,3)

def vis_data(data, f1,f2):
    """
        Visualizes data as histogram
    """
    hist = np.histogram2d(data[:, f2], data[:, f1], bins=100, range=[lims, lims])
    plt.pcolormesh(hist[1], hist[2], hist[0], alpha=0.5)

fixed_noise = torch.Tensor(sample_noise(1000))
def vis_g():
    """
        Visualizes generator's samples as circles
    """
    data = generator(fixed_noise).data.numpy()
    if np.isnan(data).any():
        return
    
    plt.scatter(data[:,0], data[:,1], alpha=0.2, c='b')
    plt.xlim(lims)
    plt.ylim(lims)
    
    
def vis_points(data,f1,f2):
    """
        Visualizes the supplied samples as circles
    """
    if np.isnan(data).any():
        return
    
    plt.scatter(data[:,f1], data[:,f2], alpha=0.2, c='b')
    plt.xlim(lims)
    plt.ylim(lims)
    

def get_grid():
    X, Y = np.meshgrid(np.linspace(lims[0], lims[1], 30), np.linspace(lims[0], lims[1], 30))
    X = X.flatten()
    Y = Y.flatten()
        
    grid = torch.from_numpy(np.vstack([X, Y]).astype(np.float32).T)
    grid.requires_grad = True
                            
    return X, Y, grid
              
X_grid, Y_grid, grid = get_grid()
def vis_d():
    """
        Visualizes discriminator's gradient on grid
    """
         
    data_gen = generator(fixed_noise)
#     loss = d_loss(discriminator(data_gen), discriminator(grid))
    loss = g_loss(discriminator(grid))
    loss.backward()
    
    grads = - grid.grad.data.numpy()
    grid.grad.data *= 0 
    plt.quiver(X_grid, Y_grid, grads[:, 0], grads[:, 1], color='black',alpha=0.9)

#%% my funcs
def make_same_length_classes(X,Y):
    len = [ Y[Y==k].shape[0] for k in set(Y)]
    max_len = max(len)
    for k in set(Y):
        k_len = Y[Y == k].shape[0]
        add_len = max_len - k_len
        if add_len!= 0:
            if add_len <= k_len:
                add_x = X[Y==k][:add_len]
            else:
                if  add_len % k_len!= 0:
                    add_x = X[Y==k][: add_len % k_len]
                    one = 0
                else:
                    add_x = X[Y==k]
                    one = 1
                for i in np.arange(add_len // k_len  - one):
                    add_x = np.vstack((add_x, X[Y==k]))
            X = np.vstack((X,add_x))
            Y = np.hstack((Y, np.ones(add_len)*k))
    return (X,Y)

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
        Y = Y-1
#%% model data
if INPUT_DATA == 1:
    
    [X,Y] = datasets.make_blobs(n_samples = 500, n_features = 2, centers = 4, cluster_std = [0.2,0.2,0.3,0.15,0.15], random_state = 58)



#%%
    # PyTorch Dataset
class MyDataSet(torchdata.Dataset):
    def __init__(self, X,Y, transform = None):
        if transform !=None:
            X= transform(X)

        self.X = torch.from_numpy(X)
        self.Y =torch.from_numpy(Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        #sample = (X[idx,:],self.code_to_vec(Y[idx] - 1))


        #sample = (self.X[idx, :], self.Y[idx])
        sample = self.X[idx, :]
        return sample
    
    #https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905
    def code_to_vec(self, class_num):
        y = np.zeros(output_shape, dtype = np.float32)
        y[int(class_num.item())] = 1.
        return y
    
#%%
input_shape = X.shape[1]
output_shape = 16#len(set(Y))

#%%
#CLASSICAL GAN

def get_generator(noise_dim, out_dim, hidden_dim=100):
    layers = [
        nn.Linear(noise_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, out_dim)
    ]
    return nn.Sequential(*layers)

def get_discriminator(in_dim, hidden_dim=100):
    layers = [
        nn.Linear(in_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid()
    ]
        
    return nn.Sequential(*layers)

#%%

generator = get_generator(PARAMETERS["noise_dim"], out_dim = input_shape)
discriminator = get_discriminator(in_dim = input_shape)

lr = PARAMETERS["lr"]
g_optimizer = optim.Adam(generator.parameters(),     lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

#%%
def g_loss(d_scores_fake):
    loss =  - torch.mean(torch.log(d_scores_fake))
    return loss
def d_loss(d_scores_fake, d_scores_real):
    loss = - (torch.mean(torch.log(d_scores_real)) + torch.mean(torch.log(1 - d_scores_fake)))
    return loss

#%%
len = [ Y[Y==k].shape[0] for k in set(Y)]
(X,Y) = make_same_length_classes(X,Y)
print("Make same length classes ok", list(zip(len,[Y[Y==k].shape[0] for k in set(Y)])))




X = X.astype(np.float32)
scaler = StandardScaler().fit(X)
Y = Y.astype(np.int64)

plt.rcParams['figure.figsize'] = (12, 12)
vis_data(scaler.transform(X),PARAMETERS["f1"],PARAMETERS["f2"])
vis_g()
#vis_d()
plt.show()
#s = StratifiedShuffleSplit(n_splits= 1, train_size= 0.7)
#train_index,test_index = next(s.split(X,Y))
PSdataset_train = MyDataSet(X,Y, transform = scaler.transform)
#PSdataset_test = MyDataSet(X[test_index],Y[test_index], transform=scaler.transform)
train_loader = torchdata.DataLoader(PSdataset_train, batch_size= PARAMETERS["batchsize"], shuffle =True )
#test_loader  = torchdata.DataLoader(PSdataset_test,batch_size=  PSdataset_test.Y.shape[0], shuffle = True)

#%%


def train(args, generator, discriminator, train_loader, d_optimizer, g_optimizer, epoch):
    for it, real_data in enumerate(train_loader):
        for _ in range(args["k_d"]):
            d_optimizer.zero_grad()
            noise = torch.Tensor(sample_noise(real_data.shape[0]))
            fake_data = generator(noise)
            loss = d_loss(discriminator(fake_data), discriminator(real_data))            
            loss.backward()
            
            for p in discriminator.parameters():
                p.data.clamp_(-0.01,0.01)
                           
            # Update
            d_optimizer.step()
             
             
        for _ in range(args["k_g"]):
            g_optimizer.zero_grad()
            
            # Sample noise
            noise = torch.Tensor(sample_noise(real_data.shape[0]))

            # Compute gradient
            fake_data = generator(noise)
            loss = g_loss(discriminator(fake_data))
            loss.backward()
            
            # Update
            g_optimizer.step()
       
        if it % 4 == 0:
            plt.clf()
            vis_data(scaler.transform(X),PARAMETERS["f1"],PARAMETERS["f2"])
            
            vis_g()

            #vis_d()
            display.clear_output(wait=True)
            display.display(plt.gcf())
            print(f"Epoch {epoch}; Iteration {it}")    
def test(args,generator, discriminator,test_loader):
    pass    
#%%

for epoch in range(1, PARAMETERS["epochs"] + 1):

    train(PARAMETERS, generator, discriminator, train_loader, d_optimizer,g_optimizer, epoch)

    test(PARAMETERS, generator, discriminator, train_loader)
plt.show()