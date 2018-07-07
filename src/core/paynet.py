import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import *

from .game import ZeroSumGame
from .solve import GRELogit

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
    
