import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import *

from .game import ZeroSumSequenceGame 
from .solve import QRESeqLogit


class ZSGSeqSolver(torch.autograd.Function):
    # Naive way of iterating through each item in the mini-batch
    def __init__(self, Iu, Iv, Au, Av, verify='kkt', single_feature=False):
        self.Iu, self.Iv = Iu, Iv
        self.Au, self.Av = Au, Av
        self.usize, self.vsize = len(Au), len(Av)
        self.uinfosize, self.vinfosize = len(Iu), len(Iv)
        self.verify = verify
        self.single_feature = single_feature
        super(ZSGSeqSolver, self).__init__()

    def forward(self, input):
        input_np = input.numpy()
        batchsize = input_np.shape[0]
        U, V = np.zeros([batchsize, self.usize], dtype=np.float64), np.zeros([batchsize, self.vsize], dtype=np.float64)
        self.dev2s = []
        for i in range(batchsize):
            if not self.single_feature or i == 0:
                p = np.squeeze(input_np[i, :, :])
                game = ZeroSumSequenceGame(self.Iu, self.Iv, self.Au, self.Av, p)
                logitSolver = QRESeqLogit(game)
                u, v, dev2 = logitSolver.Solve(alpha=0.15, beta=0.50, tol=10**-7, epsilon=0., min_t=10**-20, max_it=1000, verify=self.verify)
                U[i, :] , V[i, :] = np.squeeze(u), np.squeeze(v)
                self.dev2s.append(dev2)
            else: # reuse old result when input features are guaranteed to be the same
                U[i, :] , V[i, :] = np.squeeze(u), np.squeeze(v)
                self.dev2s.append(dev2)

        Ut, Vt = torch.DoubleTensor(U), torch.DoubleTensor(V)
        self.input, self.U, self.V = input_np, U, V
        # IMPORTANT NOTE: save for backward does not work unless static method! Memory accumulates and leaks
        # self.save_for_backward(input, U, V)
        return Ut, Vt

    def backward(self, grad_u, grad_v):
        batchsize = grad_u.shape[0]
        # P, U, V = tuple([x.data.numpy() for x in self.saved_variables])
        P, U, V = self.input, self.U, self.V
        dP = np.zeros([batchsize, self.usize, self.vsize])
        for i in range(batchsize):
            u, v = U[i, :], V[i, :]
            p = P[i, :, :]
            
            gu, gv = grad_u[i, :].numpy(), grad_v[i, :].numpy()
            solve_rhs = -np.concatenate([gu, gv, [0] * (self.uinfosize + self.vinfosize)], axis=0)
            if not self.single_feature:
            # if True:
                d = np.linalg.solve(self.dev2s[i], solve_rhs)
            elif self.single_feature and i == 0:
                lu_and_piv = scipy.linalg.lu_factor(self.dev2s[0], check_finite=False)
                d = scipy.linalg.lu_solve(lu_and_piv, solve_rhs, check_finite=False)
            else:
                d = scipy.linalg.lu_solve(lu_and_piv, solve_rhs, check_finite=False)

            du, dv = d[:self.usize], d[self.usize:(self.usize+self.vsize)]
            dp = np.outer(du, v) + np.outer(u, dv)
            dP[i, :, :] = dp
        return torch.DoubleTensor(dP)


if __name__ == '__main__':
    pass
