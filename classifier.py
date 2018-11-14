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
import sklearn
from sklearn import datasets, linear_model, metrics
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
PARAMETERS   = {"lr": 0.01, "momentum": 0.5, "epochs": 100, "batchsize": 64, "batchsize_test": 500}


# %% my funcs
def make_same_length_classes(X, Y):
    len = [Y[Y == k].shape[0] for k in set(Y)]
    max_len = max(len)
    for k in set(Y):
        k_len = Y[Y == k].shape[0]
        add_len = max_len - k_len
        if add_len != 0:
            if add_len <= k_len:
                add_x = X[Y == k][:add_len]
            else:
                if add_len % k_len != 0:
                    add_x = X[Y == k][: add_len % k_len]
                    one = 0
                else:
                    add_x = X[Y == k]
                    one = 1
                for i in np.arange(add_len // k_len - one):
                    add_x = np.vstack((add_x, X[Y == k]))
            X = np.vstack((X, add_x))
            Y = np.hstack((Y, np.ones(add_len) * k))
    return (X, Y)


np.random.seed(15)
INPUT_DATA = 1  # 1 - Model data
# 2 - PetroSpec

# %% read data from file
if INPUT_DATA == 2:
    myfile = open("PS_X_without_outlier.csv", "r")
    with myfile:
        reader = csv.reader(myfile)
        i = -1
        for row in reader:

            if i == -1:
                X = np.array([row], float)
                i = 1
            elif i == 1:
                i = 0
            else:
                X = np.insert(X, X.shape[0], row, axis=0)
                i = 1

    myfile = open("PS_Y_without_outlier.csv", "r")
    with myfile:
        reader = csv.reader(myfile)
        i = -1
        for row in reader:

           if i == -1:
              Y = np.array([row], float)
              i = 1
           elif i == 1:
              i = 0
           else:
              Y = np.insert(Y, Y.shape[0], row, axis=0)
              i = 1
    Y = np.array(Y, int).reshape(Y.shape[0], )
    Y = Y - 1
# %% model data
if INPUT_DATA == 1:
    [X, Y] = datasets.make_blobs(n_samples=15000, n_features=2, centers=4, cluster_std=[0.2, 0.2, 0.3, 0.15, 0.15],
                                 random_state=58)


# %%
# PyTorch Dataset
class MyDataSet(torchdata.Dataset):
    def __init__(self, X, Y, transform=None):
        if transform != None:
            X = transform(X)

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = (self.X[idx, :], self.Y[idx])
        return sample

    # https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905
    #def code_to_vec(self, class_num):
    #    y = np.zeros(output_shape, dtype=np.float32)
    #    y[int(class_num.item())] = 1.
    #    return y


# %%
input_shape  = X.shape[1]
output_shape = len(set(Y))


# https://github.com/pytorch/examples/blob/master/mnist/main.py

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, 50)
        self.fc2 = nn.Linear(50, output_shape)

    def forward(self, x):
        # x =  x.view(-1,self.num_flat_features(x))
        x = self.fc1(x).clamp(min=0)
        x = self.fc2(x)
        return x  # F.log_softmax(x,dim = 1)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train(args, model, train_loader, optimizer, epoch):
    model.train()

    for data, target in train_loader:
        output = model(data)

        loss = MyLossFunc(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} Loss {:.6f}%'.format(epoch, loss.item()))


def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += MyLossFunc(output, target).item()

            pred = output.max(1, keepdim=True)[1]
            tmp = torch.tensor(pred.eq(target.view_as(pred)), dtype=torch.float32)
            correct += int(tmp.sum().item())
            all += target.shape[0]

    C = confusion_matrix(np.array(target), np.array(pred))
    test_loss /= test_loader.__len__()
    acc = 100. * correct / all
    print('Test set: Average loss {:.4f}%  Accuracy {:.2f}% ( {} / {})'.format(test_loss, acc, correct, all))
    return C, np.array(target), np.array(pred)


model = Net()
# model.load_state_dict(torch.load("PS_classifier.pt"))
# model.eval()

MyLossFunc = torch.nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr= PARAMETERS['lr'])
optimizer = optim.SGD(model.parameters(), lr=PARAMETERS["lr"], momentum=PARAMETERS["momentum"])

# %% extend data to same class length
len = [Y[Y == k].shape[0] for k in set(Y)]
(X, Y) = make_same_length_classes(X, Y)
print("Make same length classes ok", list(zip(len, [Y[Y == k].shape[0] for k in set(Y)])))

X = X.astype(np.float32)
scaler = StandardScaler().fit(X)
Y = Y.astype(np.int64)

s = StratifiedShuffleSplit(n_splits=1, train_size=0.7)
train_index, test_index = next(s.split(X, Y))
PSdataset_train = MyDataSet(X[train_index], Y[train_index], transform=scaler.transform)
PSdataset_test = MyDataSet(X[test_index], Y[test_index], transform=scaler.transform)
train_loader = torchdata.DataLoader(PSdataset_train, batch_size=PARAMETERS["batchsize"], shuffle=True)
test_loader = torchdata.DataLoader(PSdataset_test, batch_size=PSdataset_test.Y.shape[0], shuffle=True)
for epoch in range(1, PARAMETERS["epochs"] + 1):
    train(PARAMETERS, model, train_loader, optimizer, epoch)
    C, Y_target, Y_pred = test(PARAMETERS, model, test_loader)
F_score = sklearn.metrics.f1_score(Y_target, Y_pred, average=None)
F_score_avarage = sklearn.metrics.f1_score(Y_target, Y_pred, average='micro')
print(C)
print(F_score_avarage)
ILLUSTRATE = True
if True:
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

    Y_net = model(torch.from_numpy(scaler.transform(X).astype(np.float32))).max(1, keepdim=True)[1]
    plt.scatter(X[:, 0], X[:, 1], s=4, marker="s", c=Y_net.numpy().reshape(Y_net.shape[0], ), cmap=plt.cm.Paired)
# print(model(torch.from_numpy(scaler.transform([X[1122,:]]).astype(np.float32))).max(1, keepdim = True)[1]+1)


# print(list(zip(model(torch.tensor(X[1:10,:])).argmax(1, keepdim = True),torch.tensor(Y[1:10]))))


# illustrate

# load PS_classifier
# torch.save(model.state_dict(), "PATH name") - SAVE

# model = Net()
# model.load_state_dict(torch.load("PS_classifier.pt"))
# model.eval()


