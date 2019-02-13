import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as torchdata
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import argparse

import time

parser = argparse.ArgumentParser(description='Alex GAN')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                    help='learning rate (default: 0.003)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='CUDA training')
parser.add_argument('--seed', type=int, default=123, metavar='S',
                    help='random seed (default: 123)')
parser.add_argument('--batch-size-test', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')



args = parser.parse_args()
#torch.set_num_threads(4)
torch.manual_seed(args.seed)
lims = (-5,5)
#%% PARAMETERS
VISUALISATION = True


#N - Количество элементов сгенерированных данных
#M - #Размерность данных 
#PARAMETERS = {"lr":0.001,
              #"momentum":0.5,\
              #"epochs": 100,\
              #"batchsize": 64,\
              "batchsize_test": 120,\
              "noise_dim": 2,\
              "k_d":1,\
              "k_g":1,\
              "f1":0,\
              "f2":1,\
              "hidden_g":50,\
              "hidden_d":70,\
              "N":6400,\
              "M":2,\
              "N_noise":1000,\
              "epoch_visualisation":10,\
              "accuracy_learing_rate":0.00001 }


Load_Model = False


#%% VISUALISATION
def vis_data(data):
    """
        Visualizes data as histogram
    """
    hist = np.histogram2d(data[:, 1], data[:, 0], bins=100, range=[lims, lims])
    plt.pcolormesh(hist[1], hist[2], hist[0], alpha=0.5)


def vis_g():
    """
        Visualizes generator's samples as circles
    """
    data_good = generator(fixed_noise_good).data.numpy()
    data_bad = generator(fixed_noise_bad).data.numpy()
    if np.isnan(data_good).any() or np.isnan(data_bad).any():
        return
    
    plt.scatter(data_good[:,0], data_good[:,1], alpha=0.2, c='b')
    plt.scatter(data_bad[:,0], data_bad[:,1], alpha=0.2, c='tomato')
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
         
    data_gen_good = generator(fixed_noise_good)
    data_gen_bad = generator(fixed_noise_bad)
#     loss = d_loss(discriminator(data_gen), discriminator(grid))
    loss = g_loss(discriminator(grid),discriminator(grid))
    loss.backward()
    
    grads = - grid.grad.data.numpy()
    grid.grad.data *= 0 
    plt.quiver(X_grid, Y_grid, grads[:, 0], grads[:, 1], color='black',alpha=0.9)
#%%INPUT DATA

#load or generate X 
np.random.seed(12345)



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
    return torch.from_numpy(np.select(conds[:,:,None], arr).astype(np.float32))

NOISE_DIM = PARAMETERS["noise_dim"]
def sample_noise_good(N):
    return torch.from_numpy(np.random.normal(size=(N,NOISE_DIM),scale = [0.2,0.2]).astype(np.float32))
def sample_noise_bad(N):
    return torch.from_numpy(np.random.normal(size=(N,NOISE_DIM),loc =[-2,3], scale = [0.2,0.2]).astype(np.float32))

fixed_noise_good = torch.Tensor(sample_noise_good(PARAMETERS["N_noise"]))
fixed_noise_bad = torch.Tensor(sample_noise_bad(PARAMETERS["N_noise"]))

class MyDataSet(torchdata.Dataset):
    def __init__(self,X):
        self.X = torch.Tensor(X)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self,idx):
        return self.X[idx,:]
    
    
X_data = sample_true(PARAMETERS["N"])
dataSetX = MyDataSet(X_data)

#%% NETS
#make models Generator and Discriminator
class Generator(nn.Module):
    def __init__(self, input_shape = 2, hidden_shape = 50, output_shape = 2):
        super(Generator, self).__init__()
        self.ln1 = nn.Linear(input_shape,hidden_shape)
        self.fn1 = nn.LeakyReLU()
        self.ln2 = nn.Linear(hidden_shape,hidden_shape)
        self.fn2 = nn.LeakyReLU()
        self.ln3 = nn.Linear(hidden_shape, output_shape)
    def forward(self, x):
        x = self.ln1(x)
        x = self.fn1(x)
        #x = self.ln2(x)
        #x = self.fn2(x)
        x = self.ln3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_shape = 2, hidden_shape = 80, output_shape =  1):
        super(Discriminator,self).__init__()
        self.ln1 = nn.Linear(input_shape, hidden_shape)
        self.fn1 = nn.LeakyReLU()
        self.ln2 = nn.Linear(hidden_shape, hidden_shape)
        self.fn2 = nn.LeakyReLU()
        self.ln3 = nn.Linear(hidden_shape,hidden_shape)
        self.fn3 = nn.LeakyReLU()
        self.ln4 = nn.Linear(hidden_shape, output_shape)
        self.fn4 = nn.Sigmoid()
    def forward(self, x):

        x = self.ln1(x)
        x = self.fn1(x)
        x = self.ln2(x)
        x = self.fn2(x)
        x = self.ln3(x)
        x = self.fn3(x)
        x = self.ln4(x)
        x = self.fn4(x)
        return x

generator =  Generator(PARAMETERS["noise_dim"],PARAMETERS["hidden_g"])
discriminator = Discriminator(hidden_shape = PARAMETERS["hidden_d"])


if Load_Model:
    generator.load_state_dict(torch.load(""))
    discriminator.load_state_dict(torch.load(""))


#%% TRAIN AND ACCURACY
def d_loss(d_scores_real, d_scores_fake):

    loss = - (torch.mean(torch.log(d_scores_real)) + \
               torch.mean(torch.log(1 - d_scores_fake)))
    return loss
def g_loss(d_scores_fake_good, d_scores_fake_bad):
    loss = torch.mean(torch.log(1 - d_scores_fake_good)) + \
            torch.mean(torch.log(d_scores_fake_bad))
    return loss

lr = PARAMETERS["lr"]
g_optimizer = optim.Adam(generator.parameters(),     lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discmd
riminator.parameters(), lr=lr, betas=(0.5, 0.999))

train_loader = torchdata.DataLoader(dataSetX, batch_size = PARAMETERS["batchsize"])
test_loader = torchdata.DataLoader(dataSetX,batch_size=dataSetX.__len__())

def train(generator, discriminator, loader, epoch):
    m = 0
    for it, real_data in enumerate(loader):
        if m == 0:
            m = real_data.shape[0]
        for _ in range(PARAMETERS["k_d"]):
            d_optimizer.zero_grad()
             
            noise_data_good = sample_noise_good(m)
            fake_data = generator(noise_data_good)
            lossD = d_loss(discriminator(real_data), discriminator(fake_data))
            # if lossD is None:
            #     print("LossD is None")
            #     break
            lossD.backward()

            #for p in discriminator.parameters():
             #   p.data.clamp_(-0.01,0.01)
            d_optimizer.step()
        for _ in range(PARAMETERS["k_g"]):
            g_optimizer.zero_grad()
                    
            noise_data_good = sample_noise_good(m)
            fake_data_good = generator(noise_data_good)
            
            noise_data_bad = sample_noise_bad(m)
            fake_data_bad = generator(noise_data_bad)
            
            lossG = g_loss(discriminator(fake_data_good), discriminator(fake_data_bad))
            lossG.backward()
            # if lossG is None:
            #     print("LossG is None")
            #     break
            g_optimizer.step()
    if epoch % 10 == 0:
        print("EPOCH: {}. Loss generator {:6f}, loss discriminator {:6f}".format(epoch,lossG,lossD))
    if VISUALISATION   and epoch % PARAMETERS["epoch_visualisation"] == 0:
        plt.figure()
        plt.rcParams['figure.figsize'] = (12, 12)
        vis_data(np.array(X_data))
        vis_g()
        vis_d()
    return lossD,lossG
def test (loader,generator,discriminator,epoch):
      m = 0
      with torch.no_grad():
          for it, real_data in enumerate(loader):
              if m == 0:
                  m = real_data.shape[0]
              noise_data_good = sample_noise_good(m)
              fake_data_good = generator(noise_data_good)
              noise_data_bad = sample_noise_bad(m)
              fake_data_bad = generator(noise_data_bad)
              
              lossD_on_good_noise = discriminator(noise_data_good)
              lossD_on_bad_noise = discriminator(noise_data_bad)
              lossD_on_real = discriminator(real_data)
      if epoch % PARAMETERS["epoch_visualisation"]== 0:
          plt.savefig("Alex_GAN_results{:d}_{:4d}_{:2}_{:2d}_{}h{}m{}s.jpg".format(epoch, time.localtime()[0],\
                        time.localtime()[1],time.localtime()[2],time.localtime()[3],time.localtime()[4],time.localtime()[5]))

      return lossD_on_good_noise,lossD_on_bad_noise,lossD_on_real
              
              
             
GoodAccuracy = False         

for epoch in range(PARAMETERS["epochs"]+1):
    #LEARN NETS 
    train(generator,discriminator,train_loader, epoch)
    
    #CHECK ACCURACY
    lossD_on_good_noise,lossD_on_bad_noise,lossD_on_real = \
                test(dataSetX,generator,discriminator,epoch)
    l1_GN = lossD_on_good_noise.mean()
    l2_BN = lossD_on_bad_noise.mean()
    l3_RD = lossD_on_real.mean()
    if np.abs(l1_GN-0.5) < PARAMETERS["accuracy_learing_rate"]  and \
       np.abs(l2_BN-0.5) < PARAMETERS["accuracy_learing_rate"]  and \
       np.abs(l3_RD-0.5) < PARAMETERS["accuracy_learing_rate"]:
        GoodAccuracy = True  
    if GoodAccuracy:
        break

torch.save(generator.state_dict(),"generator_{:d}_{:4d}_{:2}_{:2d}_{}h{}m{}s.pht"\
           .format(epoch, time.localtime()[0],time.localtime()[1],time.localtime()[2],time.localtime()[3],time.localtime()[4],time.localtime()[5]))
torch.save(discriminator.state_dict(),"discriminator_{:d}_{:4d}_{:2}_{:2d}_{}h{}m{}s.pht"\
           .format(epoch, time.localtime()[0],time.localtime()[1],time.localtime()[2],time.localtime()[3],time.localtime()[4],time.localtime()[5]))
