import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(15)



#-----------------------GENERATOR----------------------------
def get_generator(nois_dim, out_dim, hidden_dim = 100):
    layers = [
        nn.Linear(nois_dim,hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim,hidden_dim)
        nn.LeakyReLU()
        nn.Linear(hidden_dim,out_dim)
    ]
    return nn.Sequential(*layers)

#---------------------DISCRIMINATOR--------------------------
def get_discriminator(in_dim, hidden_dim = 100):
    layesr = [
        nn.Linear(in_dim,hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim,hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim,hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim,1),
        nn.Sigmoid()
    ]
    return nn.Sequential(*layesr)
#--------------------OPTIMIZATION---------------------------
P_DATA_DIM = 2
NOISE_DIM = 20
generator =     get_generator(NOISE_DIM,out_dim = P_DATA_DIM)
discriminator = get_discriminator(in_dim = P_DATA_DIM)

lr = 0.001
g_optimizer = optim.Adam(generator.parameters(),    lr = lr, betas = (0.5,0.999))
d_optimizer = optim.Adam(discriminator.parameters(),lr = lr, betas = (0.5,0.999))
