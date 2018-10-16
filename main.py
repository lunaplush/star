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
        y = np.zeros(output_shape)
        y[class_num] = 1
        return y
    
#%%
input_shape = X.shape[1]
output_shape = len(set(Y))
PARAMETERS = {"lr":0.01, "momentum":0.5,"epochs": 10, "batchsize": 32, "batchsize_test": 500}

#https://github.com/pytorch/examples/blob/master/mnist/main.py

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_shape,50)
        self.fc2 = nn.Linear(50,output_shape)
   
    def forward(self,x):
        x =  x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x,dim =1)
   
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train(args, model, train_loader, optimizer, epoch):
   model.train()
  # batchsize = args["batchsize"]
  # batch_idx = 0
   #while (batch_idx + 1) * batchsize <= X.shape[0]:
    #   data = X[batch_idx*batchsize:(batch_idx+1)*batchsize,:]
     #  target = Y[batch_idx*batchsize:(batch_idx+1)*batchsize]
   for data,target in train_loader:
       optimizer.zero_grad()
       output = model(data)
       loss = F.nll_loss(output,target)
       loss.backward()
       optimizer.step()
       #batch_idx += 1
       print('Train Epoch: {} Loss {:.6f}%'.format(epoch, loss.item()))

def test(args,model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    #batch_idx = 0
    #batchsize = args["batchsize"]
    with torch.no_grad():
        #while (batch_idx + 1)* batchsize <= X.shape[0]:
         #   data = X[batch_idx*batchsize:(batch_idx+1) * batchsize, :]
          #  target =  Y[batch_idx*batchsize: (batch_idx+1)*batchsize ]

        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output,target,reduction= 'sum').item()
            pred = output.max(1, keepdim = True) [1]
            correct =pred.eq(target.view_as(pred)).sum().item()
            #batch_idx +=1
    test_loss /= X.shape[0]//len(test_loader)
    acc =  100. * correct / (X.shape[0]//len(test_loader))
    print('Test set: Average loss {:.4f}, Accuracy {} / {} (){:.0f}%'.format(test_loss,correct,X.shape[0]//len(test_loader), acc ))
model = Net()
optimizer = optim.SGD(model.parameters(), lr = PARAMETERS["lr"], momentum = PARAMETERS["momentum"])
PSdataset = MyDataSet(X,Y)
train_loader = torchdata.DataLoader(PSdataset, batch_size= PARAMETERS["batchsize"], shuffle =True )
test_loader  = torchdata.DataLoader(PSdataset,batch_size= PARAMETERS["batchsize_test"], shuffle = True)
for epoch in range(1, PARAMETERS["epochs"] + 1):
    
    ZERO GRAD
    train(PARAMETERS, model, train_loader, optimizer, epoch)
    test (PARAMETERS, model, test_loader)
    
#%%    
#Logistic Regression

#multi_class = "multinomial"
#clf = linear_model.LogisticRegression(solver ="sag", penalty= "l2", max_iter= 1000,random_state= 15, multi_class= multi_class).fit(X,Y)
#print("training score : %.3f (%s)" % (clf.score(X, Y), multi_class))


   
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

      

