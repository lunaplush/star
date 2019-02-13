import torch
import torch.nn as nn
import PS_data_read
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



torch.set_num_threads(4)
# ==========================)=
# IMPORTANT PARAMETER:
# Number of D updates per G update
# ===========================
# k_d, k_g = 1, 1
PARAMETERS = {"lr":0.001, "momentum":0.5,"epochs": 1, "batchsize": 64, "batchsize_test": 128,\
               , "f1":0,"f2":1,"hidden_g":20,"hidden_d":50}




#input







#Generator

class Generator(nn.Module):
    def __init__(self, noise_shape = 2,output_shape = 2, hidden_shape = 40):
        super(Generator,self).__init__()
        self.fc1 = nn.Linear(noise_shape,hidden_shape)
        self.fc2 = nn.Linear(hidden_shape,output_shape)
    def forward(self, x):
        x = self.fc1(x).clamp(min = 0)
        x = self.fc2(x)
        return  x


#Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_shape = 2, output_shape = 1, hidden_shape = 50 ):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_shape)
        self.fc2 = nn.Linear(hidden_shape,hidden_shape)
        self.fc3 = nn.Linear(hidden_shape, output_shape)

    def forward(self, x):
        x = self.fc1(x).clamp(min = 0)
        x = self.fc2(x).clamp(min = 0)
        x = self.fc3(x)
        return x  # F.log_softmax(x,dim = 1)
X,Y,scaler = PS_data_read.getPS()
M = X.shape[1]
generator = Generator(PARAMETERS["noise_dim"],M )



