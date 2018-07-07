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

class OneCardPokerTransform(nn.Module):
    '''
    Transform parameters into payoffs for one-card-poker
    Rules: uniform distribution with *KNOWN* ordinal ranking
    Parameters: 
        - initial: amount that players are forced to place in the pot before 
            you even get the cards 
        - raiseval: amount raised for min (1st) player in first decision 
            (and the second player needs to meet if he does not fold).
            If raiseval is constrained to be equal to initial, then this 
            is equivalent to one-card poker with learning the lambda parameter.
    '''

    def __init__(self, nCards):
        '''
        :param nCard number of unique cards
        '''
        super(OneCardPokerTransform, self).__init__()
        self.nCards = nCards
        _, self.imat, self.rmat = OneCardPokerComputeP(nCards)
        self.imat = Variable(torch.DoubleTensor(self.imat), requires_grad=False)
        self.rmat = Variable(torch.DoubleTensor(self.rmat), requires_grad=False)
    

    def forward(self, param):
        '''
        :param param[0]: initial, param[1]: raiseval
        '''
        batchsize = param.size()[0]
        initial, raiseval = param[:, 0], param[:, 1]
        
        normalizing_factor = 1./self.nCards/(self.nCards-1) # Probability (for chance nodes of each path)

        P = Variable(torch.DoubleTensor(np.zeros([batchsize, self.nCards*4, self.nCards*4])))

        for i in range(batchsize):
            P[i, :, :] = self.imat * initial[i].expand_as(self.imat) + self.rmat * (initial[i]+raiseval[i]).expand_as(self.rmat)
        
        P *= normalizing_factor
       
        return P


class OneCardPokerPaynet(nn.Module):
    def __init__(self, nCards, 
                tie_initial_raiseval=False, 
                uniform_dist=True, 
                initial_params=None, 
                fixbets=None, 
                verify='kkt', 
                single_feature=False, 
                dist_type='WNHD'):
        '''
        :param dist_type affects what type of distribution transform we use. 
            -WNHD for hypergeometric
            For dist_type='WNHD', we apply a softmax after the linear transform to ensure a valid distribution
        '''

        super(OneCardPokerPaynet, self).__init__()
        self.nCards = nCards
        self.tie_initial_raiseval = tie_initial_raiseval 
        self.uniform_dist = uniform_dist
        self.dist_type = dist_type 
        if tie_initial_raiseval:
            self.fc = nn.Linear(1, 1, bias=False)
        else:
            self.fc = nn.Linear(1, 2, bias=False)
        if not uniform_dist:
            self.fc_dist = nn.Linear(1, self.nCards, bias=False)

        # Initial parameters
        if initial_params is not None:
            if self.tie_initial_raiseval:
                self.fc_dist.weight = torch.nn.Parameter(torch.from_numpy(np.atleast_2d(initial_params[1:]).T))
                self.fc.weight = torch.nn.Parameter(torch.from_numpy(np.atleast_2d(initial_params[0:1]).T)) 
            else: # both ante and bets are to be learnt, these are parameters 0 and 1
                self.fc_dist.weight = torch.nn.Parameter(torch.from_numpy(np.atleast_2d(initial_params)[2:].T))
                self.fc.weight = torch.nn.Parameter(torch.from_numpy(np.atleast_2d(initial_params)[:2].T)) 

        # initialize and fix bets. Note this OVERWRITES what is in initial_params
        if fixbets is not None:
            if self.tie_initial_raiseval:
                self.fc.weight = torch.nn.Parameter(torch.from_numpy(np.atleast_2d([fixbets]).T)) 
            else: # both ante and bets are to be learnt, these are parameters 0 and 1
                self.fc.weight = torch.nn.Parameter(torch.from_numpy(np.atleast_2d([fixbets, fixbets]).T)) 

            for param in self.fc.parameters(): param.requires_grad = False

        self.Iu, self.Iv, self.Au, self.Av = OneCardPokerComputeSets(nCards)
        self.solver = ZSGSeqSolver(self.Iu, self.Iv, self.Au, self.Av, verify=verify, single_feature=single_feature)
        if  self.dist_type == 'WNHD':
            self.transform = OneCardPokerWNHDTransform(nCards)
            self.softmax = torch.nn.Softmax()
        else: assert False, 'Invalid dist_type'

    def forward(self, x):
        nbatchsize = x.size()[0]
        j = self.fc(x) # 2-vector in initial and raising values
        if self.tie_initial_raiseval:
            j = torch.cat([j, j], dim=1)

        if self.dist_type=='WNHD':
            r = self.softmax(self.fc_dist(x))
        else: assert False, 'Invalid dist_type'

        P = self.transform(torch.cat([j, r], dim=1))
        u, v = self.solver(P)

        return u, v, P, j, r

class OneCardPokerWNHDTransform(nn.Module):
    '''
    Transform parameters into payoffs for one-card-poker.
    Transformation is done similar to the Wallenius' noncentral hypergeometric distribution.
    Rules: *KNOWN* ordinal ranking
    Parameters: 
        - initial: amount that players are forced to place in the pot before 
            you even get the cards 
        - raiseval: amount raised for min (1st) player in first decision 
            (and the second player needs to meet if he does not fold).
            If raiseval is constrained to be equal to initial, then this 
            is equivalent to one-card poker with learning the lambda parameter.
        - carddist: vector of *normalized* card distributions
    '''

    def __init__(self, nCards):
        '''
        :param nCard number of unique cards
        '''
        super(OneCardPokerWNHDTransform, self).__init__()
        self.nCards = nCards
    
        self.cmpmat = np.zeros([nCards, nCards])
        self.cmpmat[np.triu_indices(nCards)] += 1.
        self.cmpmat[np.tril_indices(nCards)] -= 1.


    def forward(self, param):
        batchsize = param.size()[0]
        initial, raiseval = param[:, 0], param[:, 1]
        traiseval = initial + raiseval
        initial_m = initial.unsqueeze(-1).unsqueeze(-1).expand([batchsize, self.nCards, self.nCards])
        traiseval_m = traiseval.unsqueeze(-1).unsqueeze(-1).expand([batchsize, self.nCards, self.nCards])
        carddist = param[:, 2:]

        # Unnormalized probability matrices
        outer = torch.bmm(carddist.unsqueeze(2), carddist.unsqueeze(1))
        correction = Variable(torch.DoubleTensor(np.zeros([batchsize, self.nCards, self.nCards])))
        normalizing_constant = Variable(torch.DoubleTensor(np.zeros([batchsize, self.nCards, self.nCards])))
        for i in range(batchsize): # TODO: vectorize. As of Jan 2018, no block version of diag exists
            correction[i, :, :] = torch.diag(carddist[i, :]) ** 2
            normalizing_constant[i, :, :] = torch.diag(1./(1.0 - carddist[i, :]))

        probability_matrix = torch.bmm(normalizing_constant, outer-correction)
        
        P = Variable(torch.DoubleTensor(np.zeros([batchsize, self.nCards*4, self.nCards*4])))
        
        vcmpmat = Variable(torch.DoubleTensor(self.cmpmat).unsqueeze(0).expand([batchsize, self.nCards, self.nCards]), requires_grad=False)
        P[:, slice(0,self.nCards*2,2), slice(0, self.nCards*2,2)] = vcmpmat * probability_matrix * initial_m
        P[:, slice(self.nCards*2, self.nCards*4, 2), slice(1, self.nCards*2, 2)] = probability_matrix * initial_m
        P[:, slice(1, self.nCards*2,2), slice(self.nCards*2, self.nCards*4,2)] = -probability_matrix * initial_m 
        P[:, slice(self.nCards*2+1, self.nCards*4,2),slice(1, self.nCards*2,2)] = vcmpmat * probability_matrix * traiseval_m
        P[:, slice(1, self.nCards*2,2), slice(self.nCards*2+1, self.nCards*4,2)] = vcmpmat * probability_matrix * traiseval_m

        return P


def OneCardPokerWNHDComputeP(nCards, initial=1., raiseval=1., card_distribution=None):
    '''
    :param dist array-like distribution of cards. Should sum to 1
    '''
    if card_distribution is None:
        card_distribution = [1./nCards for i in range(nCards)] # uniform by default
    dist = np.array(card_distribution)
    total_cards = np.sum(dist)
    probability_matrix = np.dot(np.diag(1./(1.-dist)), np.outer(dist, dist) - np.diag(dist**2))

    cmpmat = np.zeros([nCards, nCards])
    cmpmat[np.triu_indices(nCards)] += 1.
    cmpmat[np.tril_indices(nCards)] -= 1.

    # Partial payoff matrix for portions where payoff is only the initial
    initial_mat = np.zeros([nCards*4, nCards*4])
    # 1st player waits, second player waits -- compare cards
    initial_mat[np.ix_(range(0,nCards*2,2), range(0, nCards*2,2))] = cmpmat * probability_matrix
    # 1st player waits, second player raises, first player waits(folds) (1)
    initial_mat[np.ix_(range(nCards*2, nCards*4, 2), range(1, nCards*2, 2))] = np.ones([nCards, nCards]) * probability_matrix
    # First player raises, 2nd player forfeits (-1)
    initial_mat[np.ix_(range(1, nCards*2,2), range(nCards*2, nCards*4,2))] = -np.ones([nCards, nCards]) * probability_matrix # 2nd player forfeits after first player leaves
    
    # Partial payoff matrix for portions where payoff is initial+raiseval
    raise_mat = np.zeros([nCards*4, nCards*4])
    # 1st player waits, second player raises, first follows (+-2)
    raise_mat[np.ix_(range(nCards*2+1, nCards*4,2),range(1, nCards*2,2))] = cmpmat * probability_matrix
    # First player raises, 2nd player fights (+-2). 
    raise_mat[np.ix_(range(1, nCards*2,2), range(nCards*2+1, nCards*4,2))] = cmpmat * probability_matrix
    
    full_mat = initial_mat * initial + raise_mat * (raiseval + initial)
    return full_mat, initial_mat, raise_mat

def OneCardPokerComputeSets(nCards):
    # Actions: 0-7: Even is for no raise, odd is for raise
    # Second stage: Next 4 information states (the previous action was to bet 0, second player raises)
    # Actions: 8-15: Even is for fold, odd is for raise
    # Iu = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]])
    Iu = np.array([[i*2, i*2+1] for i in range(nCards*2)])
    # 8 information states, 4 \times 2 (whether 1st player betted)
    # First 4 information states are for 1st player passing, next 4 if first player raise
    # Infostates 0-3 are for cards 0-3 respectively, 4-7 similarly
    # Even action is for pass, odd for raise
    # Iv = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]) 
    Iv = np.array([[i*2, i*2+1] for i in range(nCards*2)])
    # Au = np.array([[4], [], [5], [], [6], [], [7], [], [], [], [], [], [], [], [], []])
    Au = [[] for i in range(nCards*4)]
    for i in range(nCards): Au[i*2] = [i + nCards]
    Au = np.array(Au)
    Av = np.array([[] for x in range(nCards*4)])

    return Iu, Iv, Au, Av
    

def OneCardPokerWNHDSeqSample(nCards, U, V, dist=None):
    '''
    :param dist unormalized distribution of cards
    '''
    if dist is None:
        dist = np.array([1./nCards for i in range(nCards)])

    Iu, Iv, Au, Av = OneCardPokerComputeSets(nCards)
    uCard = np.random.choice(range(nCards), p=dist)
    newdist = np.array(dist)
    newdist[uCard] = 0.
    vCard = np.random.choice(range(nCards), p=newdist/np.sum(newdist))

    uRet, vRet = np.array([0] * (nCards * 4)), np.array([0] * (nCards * 4))
    # Naively simulate random actions according to the behavioral strategy induced by sequence form
    uInfoSet = uCard
    uActionsAvail = Iu[uInfoSet]
    uFirstAction = np.random.choice(uActionsAvail, p=np.squeeze(U[uActionsAvail]))
    uFirstRaise = True if uFirstAction % 2 == 1 else False

    vInfoSet = nCards * (1 if uFirstRaise else 0) + vCard
    vActionsAvail = Iv[vInfoSet]
    vFirstAction = np.random.choice(vActionsAvail, p=np.squeeze(V[vActionsAvail]))
    vFirstRaise = True if vFirstAction % 2 == 1 else False

    if uFirstRaise == vFirstRaise: # Game is over, either both fold or both raise
        uRet[uFirstAction], vRet[vFirstAction] = 1.0, 1.0
        return uFirstAction, vFirstAction

    if uFirstRaise == True and vFirstRaise == False: # Game is over, first player raise, second fold
        uRet[uFirstAction], vRet[vFirstAction] = 1.0, 1.0
        return uFirstAction, vFirstAction

    # At this stage, first player did not raise by second raised.
    uInfoSet = nCards + uCard
    uActionsAvail = Iu[uInfoSet]
    uProbs = U[uActionsAvail]
    uProbs = uProbs / np.sum(uProbs) # Normalize to probability vector
    uSecondAction = np.random.choice(uActionsAvail, p=np.squeeze(uProbs))
    uRet[uSecondAction], vRet[vFirstAction] = 1.0, 1.0
    return uSecondAction, vFirstAction


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
