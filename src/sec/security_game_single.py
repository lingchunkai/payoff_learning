import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import *

from ..core.game import ZeroSumSequenceGame 
from ..core.solve import QRESeqLogit
from ..core.seq_paynet import ZSGSeqSolver

'''
1 stage game reimplemented
'''

class SecurityGameOnePaynet(nn.Module):
    def __init__(self, nDef, S1, max_val=1., nFeats=1, single_feature=False, verify='kkt'):
        '''
        :param nDef number of defensive resources
        :param static_rewards if rewards to be learnt are fixed over time
        '''

        super(SecurityGameOnePaynet, self).__init__()
        self.nDef = nDef
        self.nFeats  = nFeats
        self.max_val = max_val
        self.S1 = S1

        self.fc = nn.Linear(nFeats, 2, bias=False)

        self.Iu, self.Iv, self.Au, self.Av = ComputeSecurityGameOneSets(nDef)
        self.solver = ZSGSeqSolver(self.Iu, self.Iv, self.Au, self.Av, verify=verify, single_feature=single_feature)
        self.transform = SecurityGameOneTransform(nDef, S1=S1)


    def forward(self, x):
        nbatchsize = x.size()[0]
        j = -(torch.tanh(self.fc(x)) +1.) * self.max_val # 2-vector in initial and raising values
        P = self.transform(j)
        u, v = self.solver(P)

        return u, v, P, j


class SecurityGameOneTransform(nn.Module):

    def __init__(self, nDef, S1):
        '''
        :param nCard number of unique cards
        '''
        super(SecurityGameOneTransform, self).__init__()
        self.nDef = nDef
        self.S1 = S1
        self.P11 = Variable(torch.DoubleTensor(ComputeSuccessProbs(nDef, S1[0])), requires_grad=False)
        self.P12 = Variable(torch.DoubleTensor(list(reversed(ComputeSuccessProbs(nDef, S1[1])))), requires_grad=False)


    def forward(self, param):
        '''
        :param param[0, 1]: R1
        :param param[2, 3]: R2
        '''
        batchsize = param.size()[0]
        R1 = param[:, 0:2]

        P = Variable(torch.DoubleTensor(np.zeros([batchsize, 2, self.nDef+1])))

        P11x = self.P11.expand([batchsize, self.nDef+1])
        P12x = self.P12.expand([batchsize, self.nDef+1])

        R1x = [R1[:, i].unsqueeze(1).expand([batchsize, self.nDef+1]) for i in range(2)]
        
        E11 = P11x * R1x[0]
        E12 = P12x * R1x[1]        

        P = torch.stack(
                [E11,
                 E12], dim=1)

        return P


def ComputeSecurityGameOnePayoffMatrix(nDef, R1, S1):
    '''
    :param num defenders
    :param R1 2-tuple of reward for target 1, target 2 respectively
    :param S1 2-tuple of attacker uccess probability of getting through a single defender
    '''

    P11 = np.array(ComputeSuccessProbs(nDef, S1[0]))
    P12 = np.array(list(reversed(ComputeSuccessProbs(nDef, S1[1]))))
    E11 = P11 * R1[0]
    E12 = P12 * R1[1]

    M = np.array(
            [E11,
             E12])

    return M


def ComputeSuccessProbs(nDef, S):
    return [S ** i for i in range(nDef+1)]


def ComputeSecurityGameOneSets(nDefs):
    Iu = np.array([[0, 1]])
    Au = np.array([[], []])
    Iv = np.array([list(range(nDefs+1))])
    Av = np.array([[] for i in range(nDefs+1)])
    
    return Iu, Iv, Au, Av


def SecurityGameOneSample(nDef, U, V, S1):
    
    SuccMat = ComputeSuccessProbs(nDef, S1[0])

    defStrat = np.random.choice(range(nDef+1), p=np.squeeze(V))
    atkFirstMove = np.random.choice(range(2), p=np.squeeze(U))
    
    return atkFirstMove, defStrat


if __name__ == '__main__':
    import logging
    # logging.basicConfig(level=logging.DEBUG)
    from game import ZeroSumSequenceGame 
    from solve import QRESeqLogit

    nDef = 5
    np.set_printoptions(precision=3)
    # S1=[0.75, 0.75]
    # P = ComputeSecurityGamePayoffMatrix(nDef, R1=[-1, -2], S1=S1, R2=None, S2=None)
    S1=[0.90, 0.90]
    P = ComputeSecurityGameOnePayoffMatrix(nDef, R1=[-5, -20], S1=S1)
    # P = ComputePayoffMatrix(nDef, R1=[-200,-200], S1=[0.5, 0.5], R2=None, S2=None)
    print(P)
    Iu, Iv, Au, Av = ComputeSecurityGameOneSets(nDef)
    zssg = ZeroSumSequenceGame(Iu, Iv, Au, Av, P)
    solver = QRESeqLogit(zssg)
    ugt, vgt, _ = solver.Solve(alpha=0.1, beta=0.5, tol=10**-7, epsilon=0., min_t=10**-25, max_it=10000, verify='kkt')
    print('ugt\n', ugt)
    print('vgt\n', vgt)

    print(np.dot(P, vgt))
    print(np.dot(ugt.T, np.dot(P, vgt)))
    import copy
    tmp = copy.deepcopy(ugt)
    tmp[2:4] = ugt[4:6]
    tmp[4:6] = ugt[2:4]
    print(np.dot(tmp.T, np.dot(P, vgt)))
    print(np.dot(ugt.T, P))

    print('-------------------------------')
    print('Testing Sampling')
    nSamples = 5
    U = np.zeros((nSamples, )) # Action distributions
    V = np.zeros((nSamples, ))
    for i in range(nSamples):
        U[i], V[i] = SecurityGameOneSample(nDef, np.squeeze(ugt), np.squeeze(vgt), S1)
    U = np.array(U, dtype=int)
    V = np.array(V, dtype=int)

    print('-------------------------------')
    print('Testing security net, gradient and backprop')
    nDef = 5
    net = SecurityGameOnePaynet(nDef, S1=[0.5, 0.5])
    net = net.double()
    print(net)
    net.zero_grad()
    target_u = Variable(torch.LongTensor(U), requires_grad=False)
    target_v= Variable(torch.LongTensor(V), requires_grad=False)
    test_in = Variable(torch.DoubleTensor(np.ones([nSamples, 1])))
    out_u, out_v, out_P, out_f = net(test_in)
    print(out_u)
    print(out_v)
