import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def ExtraInit(layer):
    inp = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(inp)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, stateSize, actionSize, seed, FC1Units = 128, FC2Units = 64):
        super(Actor, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.FC1 = nn.Linear(stateSize, FC1Units)
        self.BN1 = nn.BatchNorm1d(FC1Units)
        self.FC2 = nn.Linear(FC1Units, FC2Units)
        self.BN2 = nn.BatchNorm1d(FC2Units)
        self.FC3 = nn.Linear(FC2Units, actionSize)
        self.BN3 = nn.BatchNorm1d(actionSize)
        self.initParameters()

    def initParameters(self):
        self.FC1.weight.data.uniform_(*ExtraInit(self.FC1))
        self.FC2.weight.data.uniform_(*ExtraInit(self.FC2))
        self.FC3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        x = self.FC1(state)
        x = F.relu(x)
        x = self.BN1(x)
        x = self.FC2(x)
        x = F.relu(x)
        x = self.BN2(x)
        x = self.FC3(x)
        x = self.BN3(x)

        return torch.tanh(x)

class Critic(nn.Module):
    def __init__(self, stateSize, actionSize, seed, FCS1Units = 128, FC2Units = 64):
        super(Critic, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.BN0  = nn.BatchNorm1d(stateSize)
        self.FCS1 = nn.Linear(stateSize, FCS1Units)
        self.BN1  = nn.BatchNorm1d(FCS1Units)
        self.FC2  = nn.Linear(FCS1Units + actionSize, FC2Units)
        self.BN2  = nn.BatchNorm1d(FC2Units)
        self.FC3  = nn.Linear(FC2Units, 1)
        self.initParameters()

    def initParameters(self):
        self.FCS1.weight.data.uniform_(*ExtraInit(self.FCS1))
        self.FC2.weight.data.uniform_(*ExtraInit(self.FC2))
        self.FC3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = self.BN0(state)
        x = self.FCS1(x)
        x = F.relu(x)
        x = torch.cat((x, action), dim = 1)
        x = self.FC2(x)
        x = F.relu(x)
        x = self.FC3(x)
        
        return x