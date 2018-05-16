import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import *

from game import ZeroSumGame
from solve import GRELogit

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


class ZSGSolver(torch.autograd.Function):
    # Naive way of iterating through each item in the mini-batch
    # TODO: do in parallel? How to solve for inverse? Should cache from forward pass?
    def __init__(self, usize, vsize):
        self.usize, self.vsize = usize, vsize
        super(ZSGSolver, self).__init__()

    def forward(self, input):
        input_np = input.numpy()
        batchsize = input_np.shape[0]
        U, V = np.zeros([batchsize, self.usize], dtype=np.float64), np.zeros([batchsize, self.vsize], dtype=np.float64)
        self.dev2s = []
        for i in range(batchsize):
            p = np.squeeze(input_np[i, :, :])
            game = ZeroSumGame(p)
            logitSolver = GRELogit(game)
            u, v, dev2 = logitSolver.Solve(alpha=0.15, beta=0.90, tol=10**-15, epsilon=0, min_t=10**-20, max_it=1000)
            U[i, :] , V[i, :] = u, v
            self.dev2s.append(dev2)

        U, V = torch.DoubleTensor(U), torch.DoubleTensor(V)
        self.save_for_backward(input, U, V)
        return U, V


    def backward(self, grad_u, grad_v):
        batchsize = grad_u.shape[0]
        P, U, V = tuple([x.data.numpy() for x in self.saved_variables])
        dP = np.zeros([batchsize, self.usize, self.vsize])
        for i in range(batchsize):
            u, v = U[i, :], V[i, :]
            p = P[i, :, :]
            
            gu, gv = grad_u[i, :].numpy(), grad_v[i, :].numpy()
            d = np.linalg.solve(self.dev2s[i], -np.concatenate([gu, gv, [0], [0]], axis=0))
            du, dv = d[:self.usize], d[self.usize:(self.usize+self.vsize)]
            dp = np.outer(du, v) + np.outer(u, dv)
            dP[i, :, :] = dp
        return torch.DoubleTensor(dP)
            

if __name__ == '__main__':
    print('All tests for old architectures removed')
    
