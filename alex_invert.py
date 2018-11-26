import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np


torch.set_num_threads(4)


#%%INPUT DATA

#load or generate X and y
np.random.seed(12345)

lims = (-5,-5)

from scipy.stats import rv_discrete

MEANS = np.array(
        [[-1,-3],
         [1,3],
         [-2,0],
        ])
COVS = np.array(
        [[[1,0.8],[0.8,1]],
        [[1,-0.5],[-0.5,1]],
        [[1,0],[0,1]],
        ])
PROBS = np.array([
        0.2,
        0.5,
        0.3
        ])
assert len(MEANS) == len(COVS) == len(PROBS), "number of components mismatch"
COMPONENTS = len(MEANS)

comps_dist = rv_discrete(values=(range(COMPONENTS), PROBS))

def sample_true(N):
    comps = comps_dist.rvs(size=N)
    conds = np.arange(COMPONENTS)[:,None] == comps[None,:]
    arr = np.array([np.random.multivariate_normal(MEANS[c], COVS[c], size=N)
                     for c in range(COMPONENTS)])
    return np.select(conds[:,:,None], arr).astype(np.float32)

NOISE_DIM = 2
def sample_noise_good(N):
    return np.random.normal(size=(N,NOISE_DIM),scale = [[0.3,0],[0,0.3]]).astype(np.float32)
def sample_noise_bad(N):
    return np.random.normal(size=(N,NOISE_DIM),mean =[-2,3], scale = [[0.3,0],[0,0.3]]).astype(np.float32)


#%% NETS
#make models Generator and Discriminator
class Generator(nn.Module):
    def __init__(self, input_shape = 2, hidden_shape = 50, output_shape = 2):
        super(Generator, self).__init__()
        self.ln1 = nn.Linear(input_shape,hidden_shape)
        self.ln2 = nn.Linear(hidden_shape,hidden_shape)
        self.ln3 = nn.Linear(hidden_shape, output_shape)
    def forward(self, x):
        x = self.ln1(x)
        x = nn.LeakyReLU(x)
        x = self.ln2(x)
        x = nn.LeakyReLU(x)
        x = nn.ln3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_shape, hidden_shape = 80, output_shape =  1):
        super(Discriminator,self).__init__()
        self.ln1 = nn.Linear(input_shape, hidden_shape)
        self.ln2 = nn.Linear(hidden_shape, hidden_shape)
        self.ln3 = nn.Linear(hidden_shape,hidden_shape)
        self.ln4 = nn.Linear(hidden_shape, output_shape)
    def forward(self, x):
        x = self.ln1(x)
        x = nn.LeakyReLU(x)
        x = self.ln2(x)
        x = nn.LeakyReLU(x)
        x = nn.ln3(x)
        x = nn.LeakyReLU(x)
        x = nn.ln4(x)
        x = nn.Sigmoid(x)
        return x


#%% TRAIN AND ACCURACY

#%% make DataSet and DataLoader

#%% make Train and Test function

generator = Generator()
