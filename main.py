import torch
import torch.utils.data as torchdata
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
    # PyTorch Dataset
class MyDataSet(torchdata.Dataset):
    def __init__(self, X,Y):
        self.X =torch.from_numpy(X)
        self.Y =torch.from_numpy(Y)
    def __len__(self):
        return X.shape[0]

    def __getitem__(self, idx):
        sample = (X[idx,:],self.code_to_vec(Y[idx] - 1))
        return sample
    
    #https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905
    def code_to_vec(self, class_num):
        y = np.zeros(output_shape, dtype = np.float32)
        y[int(class_num.item())] = 1.
        return y
    
#%%
input_shape = X.shape[1]
output_shape = len(set(Y))
PARAMETERS = {"lr":0.01, "momentum":0.5,"epochs": 50, "batchsize": 32, "batchsize_test": 500}

#https://github.com/pytorch/examples/blob/master/mnist/main.py

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_shape,50)
        self.fc2 = nn.Linear(50,output_shape)
   
    def forward(self,x):
        #x =  x.view(-1,self.num_flat_features(x))
        x = self.fc1(x).clamp(min = 0)
        x = self.fc2(x)
        return x #F.log_softmax(x,dim = 1)
   
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train(args, model, train_loader, optimizer, epoch):
  # model.train()

   for data,target in train_loader:
       output = model(data)

       loss = MyLossFunc(output,target)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

   print('Train Epoch: {} Loss {:.6f}%'.format(epoch, loss.item()))

def test(args,model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():

        for data, target in test_loader:
            output = model(data)
            test_loss += MyLossFunc(output,target).item()
            pred = output.max(1, keepdim = True) [1]
            #correct =pred.eq(target.view_as(pred)).sum().item()
            #batch_idx +=1
    test_loss /= X.shape[0]//len(test_loader)
    #acc =  100. * correct / (X.shape[0]//len(test_loader))
   # print('Test set: Average loss {:.4f}, Accuracy {} / {} (){:.0f}%'.format(test_loss,correct,X.shape[0]//len(test_loader), acc ))
    print('Test set: Average loss {:.4f}%'.format(test_loss))


model = Net()
#MyLossFunc = nn.NLLLoss

MyLossFunc = torch.nn.MSELoss(reduction='sum')

optimizer = optim.SGD(model.parameters(), lr = PARAMETERS["lr"], momentum = PARAMETERS["momentum"])

X = X.astype(np.float32)
Y = Y.astype(np.float32)
PSdataset = MyDataSet(X,Y)
train_loader = torchdata.DataLoader(PSdataset, batch_size= PARAMETERS["batchsize"], shuffle =True )
test_loader  = torchdata.DataLoader(PSdataset,batch_size= PARAMETERS["batchsize_test"], shuffle = True)
for epoch in range(1, PARAMETERS["epochs"] + 1):

    train(PARAMETERS, model, train_loader, optimizer, epoch)
    test (PARAMETERS, model, test_loader)
    
