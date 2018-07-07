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
2 stage game
'''

class SecurityGamePaynet(nn.Module):
    def __init__(self, nDef, S1, max_val=1., S2=None, nFeats=1, single_feature=False, static_rewards=True, verify='kkt'):
        '''
        :param nDef number of defensive resources
        :param static_rewards if rewards to be learnt are fixed over time
        '''

        super(SecurityGamePaynet, self).__init__()
        self.nDef = nDef
        self.static_rewards = static_rewards 
        self.nFeats  = nFeats
        self.max_val = max_val
        if S2 is None: S2 = S1
        self.S1, self.S2 = S1, S2

        if static_rewards:
            self.fc = nn.Linear(nFeats, 2, bias=False)
        else:
            self.fc = nn.Linear(nFeats, 2*2,  bias=False)

        self.Iu, self.Iv, self.Au, self.Av = ComputeSecurityGameSets(nDef)
        self.solver = ZSGSeqSolver(self.Iu, self.Iv, self.Au, self.Av, verify=verify, single_feature=single_feature)
        self.transform = SecurityGameTransform(nDef, S1=S1, S2=S2)


    def forward(self, x):
        nbatchsize = x.size()[0]
        j = -(torch.tanh(self.fc(x)) +1.) * self.max_val # 2-vector in initial and raising values
        if self.static_rewards:
            r = torch.cat([j, j], dim=1)
        else: r = j

        P = self.transform(r)
        u, v = self.solver(P)

        return u, v, P, j if self.static_rewards else r


class SecurityGameTransform(nn.Module):

    def __init__(self, nDef, S1, S2=None):
        '''
        :param nCard number of unique cards
        '''
        super(SecurityGameTransform, self).__init__()
        self.nDef = nDef
        if S2 is None: S2 = S1
        self.S1, self.S2 = S1, S2 
        self.P11 = Variable(torch.DoubleTensor(ComputeSuccessProbs(nDef, S1[0])), requires_grad=False)
        self.P12 = Variable(torch.DoubleTensor(list(reversed(ComputeSuccessProbs(nDef, S1[1])))), requires_grad=False)
        self.P21 = Variable(torch.DoubleTensor(ComputeSuccessProbs(nDef, S2[0])), requires_grad=False)
        self.P22 = Variable(torch.DoubleTensor(list(reversed(ComputeSuccessProbs(nDef, S2[1])))), requires_grad=False)


    def forward(self, param):
        '''
        :param param[0, 1]: R1
        :param param[2, 3]: R2
        '''
        batchsize = param.size()[0]
        R1, R2 = param[:, 0:2], param[:, 2:4]

        P = Variable(torch.DoubleTensor(np.zeros([batchsize, 10, self.nDef+1])))

        P11x = self.P11.expand([batchsize, self.nDef+1])
        P12x = self.P12.expand([batchsize, self.nDef+1])
        P21x = self.P21.expand([batchsize, self.nDef+1])
        P22x = self.P22.expand([batchsize, self.nDef+1])

        R1x = [R1[:, i].unsqueeze(1).expand([batchsize, self.nDef+1]) for i in range(2)]
        R2x = [R2[:, i].unsqueeze(1).expand([batchsize, self.nDef+1]) for i in range(2)]
        
        E11 = P11x * R1x[0]
        E12 = P12x * R1x[1]        

        E21 = P21x * R2x[0]
        E22 = P22x * R2x[1]

        P = torch.stack(
                [E11,
                 E12,
                 P11x * E21,
                 P11x * E22,
                 (1.-P11x) * E21,
                 (1.-P11x) * E22,
                 P12x * E21,
                 P12x * E22,
                 (1.-P12x) * E21,
                 (1.-P12x) * E22], dim=1)

        return P


def ComputeSecurityGamePayoffMatrix(nDef, R1, S1, R2=None, S2=None):
    '''
    :param num defenders
    :param R1 2-tuple of reward for target 1, target 2 respectively
    :param S1 2-tuple of attacker uccess probability of getting through a single defender
    :param R2 Same as R1, but for day 2. Defaults to R1
    :param S2 Same as S1, but for day 2. Defaults to R2
    '''

    if R2 is None: R2 = R1
    if S2 is None: S2 = S1

    P11 = np.array(ComputeSuccessProbs(nDef, S1[0]))
    P12 = np.array(list(reversed(ComputeSuccessProbs(nDef, S1[1]))))
    E11 = P11 * R1[0]
    E12 = P12 * R1[1]

    P21 = np.array(ComputeSuccessProbs(nDef, S2[0]))
    P22 = np.array(list(reversed(ComputeSuccessProbs(nDef, S2[1]))))
    E21 = P21 * R2[0]
    E22 = P22 * R2[1]

    M = np.array(
            [E11,
             E12,
             P11 * (E21),
             P11 * (E22),
             (1.-P11) * E21,
             (1.-P11) * E22,
             P12 * (E21),
             P12 * (E22),
             (1.-P12) * E21,
             (1.-P12) * E22])

    return M


def ComputeSuccessProbs(nDef, S):
    return [S ** i for i in range(nDef+1)]


def ComputeSecurityGameSets(nDefs):
    Iu = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    Au = np.array([[1, 2], [3, 4], [], [], [], [], [], [], [], []])
    Iv = np.array([list(range(nDefs+1))])
    Av = np.array([[] for i in range(nDefs+1)])
    
    return Iu, Iv, Au, Av


def SecurityGameSample(nDef, U, V, S1, S2=None):
    '''
    '''
    if S2 is None: S2 = S1
    
    SuccMat = [ComputeSuccessProbs(nDef, S1[0]), ComputeSuccessProbs(nDef, S1[1])]

    defStrat = np.random.choice(range(nDef+1), p=np.squeeze(V))
    atkFirstMove = np.random.choice(range(2), p=np.squeeze(U)[0:2])

    pFirstSuccess = SuccMat[atkFirstMove][defStrat]
    firstSuccess = np.random.choice([True, False], p=[pFirstSuccess, 1.0-pFirstSuccess])

    if atkFirstMove == 0:
        if firstSuccess: # attack successful 
            finalAtkStrat = np.random.choice([2, 3], p=np.squeeze(U)[2:4]/U[0])
        else:
            finalAtkStrat = np.random.choice([4, 5], p=np.squeeze(U)[4:6]/U[0])
    else:
        if firstSuccess: # attack successful 
            finalAtkStrat = np.random.choice([6, 7], p=np.squeeze(U)[6:8]/U[1])
        else:
            finalAtkStrat = np.random.choice([8, 9], p=np.squeeze(U)[8:10]/U[1])
    
    return finalAtkStrat, defStrat


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
    P = ComputeSecurityGamePayoffMatrix(nDef, R1=[-5, -20], S1=S1, R2=None, S2=None)
    # P = ComputePayoffMatrix(nDef, R1=[-200,-200], S1=[0.5, 0.5], R2=None, S2=None)
    print(P)
    Iu, Iv, Au, Av = ComputeSecurityGameSets(nDef)
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
        U[i], V[i] = SecurityGameSample(nDef, np.squeeze(ugt), np.squeeze(vgt), S1)
    U = np.array(U, dtype=int)
    V = np.array(V, dtype=int)

    print('-------------------------------')
    print('Testing security net, gradient and backprop')
    nDef = 5
    net = SecurityGamePaynet(nDef, S1=[0.5, 0.5])
    net = net.double()
    print(net)
    net.zero_grad()
    target_u = Variable(torch.LongTensor(U), requires_grad=False)
    target_v= Variable(torch.LongTensor(V), requires_grad=False)
    test_in = Variable(torch.DoubleTensor(np.ones([nSamples, 1])))
    out_u, out_v, out_P, out_probs = net(test_in)
    print(out_u)
    print(out_v)
