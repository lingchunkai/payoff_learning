import numpy as np
import argparse
import h5py, scipy
from itertools import *
from ..core.game import ZeroSumSequenceGame 
from ..core.solve import QRESeqLogit
from .datagen import DataGen, dict_to_h5

class SeqDataGen(object):
    def __init__(self, Iu, Iv, Au, Av, simulator, seed=0):
        self.usize, self.vsize = len(Au), len(Av)
        self.r = np.random.RandomState(seed)
        self.simulator = simulator

    
    def GenData(self, nSamples):
        assert False, 'Base class, not implemented!'


    def GetActionSet(self, P):
        solver = QRESeqLogit(ZeroSumSequenceGame(self.Iu, self.Iv, self.Au, self.Av, P))
        
        # Ground truth distributions
        U, V, _ = solver.Solve(alpha=0.1, beta=0.5, tol=10**-7, epsilon=0., min_t=10**-25, max_it=10000, verify=False)

        # Sample actions from ground truths
        Au, Av = self.simulator(U, V)

        return U, V, Au, Av


class OneCardPokerWNHDGen(SeqDataGen):
    '''
    One card poker with WNHD distributed card draws
    '''
    def __init__(self, nCards, initial=1., raiseval=1., dist=None, stdev=10.0, seed=0, data_sampling_dist='same_as_cards'):
        '''
        :param dist normalized (e.g. via softmax) card distribution
        :param data_sampling_dist card distribution the *data* is sampled from. 
               Note this *can* be different from the card distribution assumed by the players.
               defaults to the same as dist (i.e. data distribution is the same as card distribution) 
        '''
        self.nCards = nCards
        self.nFeats = 1
        self.initial, self.raiseval = initial, raiseval
        self.Iu, self.Iv, self.Au, self.Av = OneCardPokerComputeSets(nCards)

        r = np.random.RandomState(seed)
        if dist is None:
            self.dist = np.array([1./nCards for i in range(nCards)])
        elif dist == 'dirichlet':
            # NOTE: RATE PARAMETER is stdev 
            print('dirichlet with parameter', stdev)
            self.dist = r.dirichlet(np.ones([nCards]) * stdev)
        else:
            assert False, 'Invalid card playing distribution'

        if data_sampling_dist is not 'same_as_cards':
            if data_sampling_dist == 'uniform': # Almost uniform...
                self.data_sampling_dist = [1./nCards] * self.nCards 
                print('sampling from uniform')
            else:
                assert False, 'Unknown data sampling distribution'
        else: # same as card distribution
            self.data_sampling_dist = self.dist

        print(self.dist)
        self.sim = lambda ugt, vgt, nCards=nCards, dist=self.dist: OneCardPokerWNHDSeqSample(nCards, ugt, vgt, self.data_sampling_dist)
        super(OneCardPokerWNHDGen, self).__init__(self.Iu, self.Iv, self.Au, self.Av, self.sim, seed)

    
    def GenData(self, nSamples):
        P = np.zeros((nSamples, self.nCards*4, self.nCards*4))
        F = np.ones((nSamples, self.nFeats))
        U = np.zeros((nSamples, self.nCards*4)) # Action distributions
        V = np.zeros((nSamples, self.nCards*4))
        Au = np.zeros((nSamples,)) # Sampled actions taken by u-player
        Av = np.zeros((nSamples,))
        D = np.zeros((nSamples, self.nCards)) # Distribution of cards

        # Create payoff matrices and solve for them
        for i in range(nSamples):
            P[i, :], _, _ = OneCardPokerWNHDComputeP(self.nCards, initial=self.initial, raiseval=self.raiseval, card_distribution=self.dist)
            U[i, :], V[i, :], Au[i], Av[i] = map(np.squeeze, self.GetActionSet(P[i, :]))
            D[i, :] = self.dist

        d = {'P': P, 'D': D, 'F': F, 'U': U, 'V': V, 'Au': Au, 'Av': Av}
        return d, (P, F, U, V, Au, Av)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='''Generate extensive form dataset (Sequence Form).\n 
            Currently supports poker and security games (t=1 or t=2)''',
            epilog='''Sample usage:\n
            python seq_datagen.py --type OneCardPokerWNHD --nCards 13 --initial_bet 1.0 --raiseval 1.0 --nSamples 500 --savePath ./seq_datagen_default.h5\n
            python seq_datagen.py --type OneCardPokerWNHD --nCards 13 --initial_bet 1.0 --raiseval 1.0 --dist exponential --stdev 10.0 --nSamples 500 --savePath ./seq_datagen_default.h5''')
    parser.add_argument('--type', type=str, default='OneCardPoker', 
                    help='{OneCardPokerWNHD, SecurityGame, SecurityGameOne}') 
    parser.add_argument('--nCards', type=int,
                    help='nCards if type is OneCardPoker')
    parser.add_argument('--nDef', type=int,
                    help='nDef if type is SecurityGame')
    parser.add_argument('--SecurityGameScale', type=float, default='2.5',
                    help='Scale of data distribution')
    parser.add_argument('--SecurityGameStaticRewards', type=int,
                    help='1 if static rewards')
    parser.add_argument('--SecurityGameDefProbs', type=str, default='0.5,0.5,0.5,0.5',
                    help='4-floats describing success probabilities.')
    parser.add_argument('--dist', type=str, default=None,
                    help='poker dist, {dirichlet}')
    parser.add_argument('--sampling_dist', type=str, default='same_as_cards',
                    help='{same_as_cards, uniform}')
    parser.add_argument('--stdev', type=float, default=10.,
                    help='Rate parameter for distribution for each card if dist=dirichlet')
    parser.add_argument('--initial_bet', type=float, default=1.0,
                    help='initial bet if type is OneCardPoker')
    parser.add_argument('--raiseval', type=float, default=1.0,
                    help='raising value if type is OneCardPoker')
    parser.add_argument('--nSamples', type=int, required=True,
                    help='Number of samples to generate.')
    parser.add_argument('--seed', type=int, default=0,
                    help='Randomization seed.')
    parser.add_argument('--savePath', type=str, default='./data/default_name.h5',
                    help='Path to save generated dataset.')

    args = parser.parse_args()

    f = h5py.File(args.savePath)
    if args.type=='OneCardPokerWNHD':
        gen = OneCardPokerWNHDGen(nCards=args.nCards, dist=args.dist, stdev=args.stdev, initial=args.initial_bet, raiseval=args.raiseval, seed=args.seed)
        d, _ = gen.GenData(args.nSamples)
        dict_to_h5(d, f)
    elif args.type=='SecurityGame':
        S = list(map(float, args.SecurityGameDefProbs.split(',')))
        gen = SecurityGameGen(args.nDef, scale=args.SecurityGameScale, seed=args.seed, static_rewards=True, S=S)
        d, _ = gen.GenData(args.nSamples)
        dict_to_h5(d, f)
    elif args.type=='SecurityGameOne':
        S = list(map(float, args.SecurityGameDefProbs.split(',')))
        gen = SecurityGameOneGen(args.nDef, scale=args.SecurityGameScale, seed=args.seed, S=S)
        d, _ = gen.GenData(args.nSamples)
        dict_to_h5(d, f)
    else:
        assert False, 'Invalid type'



