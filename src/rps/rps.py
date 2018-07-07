import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import *

from ..core.game import ZeroSumGame
from ..core.solve import GRELogit
from ..core.paynet import ZSGSolver

class RPSNet_weights(nn.Module):
    def __init__(self, size, nfeatures, scale=1.0, tanh=False, softmax=False):
        assert size==3, 'Only implemented for 3x3 matrices'
        super(RPSNet_weights, self).__init__()
        self.usize, self.vsize = size, size
        self.fc1 = nn.Linear(nfeatures, 3, bias=False)
        # self.fc2 = nn.Linear(3, 3)
        self.scale = scale
        self.softmax = softmax
        if self.softmax:
            self.softmax_layer = nn.Softmax()
        self.nfeatures = nfeatures
        self.fc1.weight.data = torch.Tensor(np.zeros([3,2]))
    

    def forward(self, x):
        fullsize = x.size()
        nbatchsize = x.size()[0]
        temp = Variable(torch.DoubleTensor(np.zeros((nbatchsize, 3, 3))))

        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        if self.softmax: x = self.softmax_layer(x)
        # x = self.fc2(x)
        x = x.view(-1, 3) 
        temp[:, 0, 1], temp[:, 1, 0] = x[:, 0], -x[:, 0]
        temp[:, 0, 2], temp[:, 2, 0] = -x[:, 1], x[:, 1]
        temp[:, 1, 2], temp[:, 2, 1] = x[:, 2], -x[:, 2]

        temp = temp * self.scale
        solver = ZSGSolver(self.usize, self.vsize)
        u, v = solver(temp)
        
        return u, v, temp, self.fc1.weight.unsqueeze(0).expand((nbatchsize, 3, self.nfeatures))


