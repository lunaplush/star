import numpy as np
import csv
from sklearn.preprocessing import StandardScaler

def getPS():
    np.random.seed(12345)
    lims = (-3,3)

    myfile = open("PS_X_without_outlier.csv", "r")
    with myfile:
        reader = csv.reader(myfile)
        i = -1
        for row in reader:
           if i == -1:
               X = np.array([row], np.float32)
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

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return X,Y,scaler
