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
    # Training set (uniform training set)
    print('Testing for uniform card distribution')
    nCards=13
    P, _, _ = OneCardPokerWNHDComputeP(nCards=nCards, initial=1., raiseval=1.)
    print('-------')
    print('Verify that as the lambda parameter grows large, we get the correct result')
    fac = 1000.
    P *= fac
    Iu, Iv, Au, Av = OneCardPokerComputeSets(nCards)
    zssg = ZeroSumSequenceGame(Iu, Iv, Au, Av, P)
    solver = QRESeqLogit(zssg)
    ugt, vgt, _ = solver.Solve(alpha=0.1, beta=0.5, tol=10**-7, epsilon=0., min_t=10**-25, max_it=10000, verify='kkt')
    print('Ground truths u:\n', ugt)
    print('Ground truths v:\n', vgt)
    print('Payoff contributed after dividing again by the addiitional lambda parameter')
    print('This should be the same as the theoretical result (e.g. from Geoff Gordon\' webpage, i.e. -0.64 if lambda is large enough')
    print(zssg.ComputePayoff(ugt, vgt)[0]/fac)
    print('--------')


    print('We now sample actions randomly from these ground truths')
    nSamples = 100
    U = np.zeros((nSamples, )) # Action distributions
    V = np.zeros((nSamples, ))
    for i in range(nSamples):
        U[i], V[i] = OneCardPokerWNHDSeqSample(nCards, ugt, vgt)
    U = np.array(U, dtype=int)
    V = np.array(V, dtype=int)

    print('--------------------------------')
    print('Testing network, gradient and backprop')
    net = OneCardPokerPaynet(nCards, uniform_dist=False)
    net = net.double()
    print(net)
    net.zero_grad()
    target_u = Variable(torch.LongTensor(U), requires_grad=False)
    target_v= Variable(torch.LongTensor(V), requires_grad=False)
    test_in = Variable(torch.DoubleTensor(np.ones([nSamples, 1])))
    out_u, out_v, out_P, out_bets, out_probs = net(test_in)
    
    print('NLL loss:')
    criterion = nn.NLLLoss()
    lossu = criterion(torch.log(out_u), target_u)
    lossv = criterion(torch.log(out_v), target_v)
    print('lossu %s, lossv %s' % (lossu, lossv))

    lossu.backward()
    
    print('Press any key to cont.')
    input()
    print('-------------------------------')
    print('All basic tests done')
