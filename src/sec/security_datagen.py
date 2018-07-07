import numpy as np
import argparse
import h5py, scipy
from itertools import *

from ..core.game import ZeroSumSequenceGame 
from ..core.solve import QRESeqLogit

from ..experiments.datagen import DataGen, dict_to_h5
from ..experiments.seq_datagen import SeqDataGen

from .security_game import ComputeSecurityGamePayoffMatrix, ComputeSecurityGameSets, SecurityGameSample
from .security_game_single import ComputeSecurityGameOnePayoffMatrix, ComputeSecurityGameOneSets, SecurityGameOneSample

class SecurityGameGen(SeqDataGen):
    '''
    Security Game
    '''
    def __init__(self, nDef, scale=2.5, seed=0, static_rewards=True, S=[0.5, 0.5, 0.5, 0.5]):
        '''
        '''
        self.nDef = nDef
        self.nFeats = 1
        self.Iu, self.Iv, self.Au, self.Av = ComputeSecurityGameSets(nDef)
        self.S = S

        r = np.random.RandomState(seed)
        self.dist = r.uniform(low=-scale, high=0., size=2)
        self.static_rewards = static_rewards
        self.R1 = self.dist
        if static_rewards: self.R2 = self.R1
        else:
            self.R2 = r.uniform(low=-scale, high=0., size=2)
        self.S = S

        self.sim = lambda ugt, vgt, nDef=nDef, R1=self.R1, R2=self.R2: SecurityGameSample(nDef, ugt, vgt, S[0:2], S[2:4])
        super(SecurityGameGen, self).__init__(self.Iu, self.Iv, self.Au, self.Av, self.sim, seed)

    
    def GenData(self, nSamples):
        P = np.zeros((nSamples, 10, self.nDef+1))
        F = np.ones((nSamples, self.nFeats))
        U = np.zeros((nSamples, 10)) # Action distributions
        V = np.zeros((nSamples, self.nDef+1))
        Au = np.zeros((nSamples,)) # Sampled actions taken by u-player
        Av = np.zeros((nSamples,))
        D = np.zeros((nSamples, 2 if self.static_rewards else 4)) # Distribution of cards

        # Create payoff matrices and solve for them
        for i in range(nSamples):
            P[i, :] = ComputeSecurityGamePayoffMatrix(self.nDef, self.R1, self.S[:2], self.R2, self.S[2:])
            U[i, :], V[i, :], Au[i], Av[i] = map(np.squeeze, self.GetActionSet(P[i, :]))
            if self.static_rewards:
                D[i, :] = self.R1 
            else: D[i, :] = np.concatenate([self.R1, self.R2])

        d = {'P': P, 'D': D, 'F': F, 'U': U, 'V': V, 'Au': Au, 'Av': Av}
        return d, (P, F, U, V, Au, Av)


class SecurityGameOneGen(SeqDataGen):
    '''
    Security Game
    '''
    def __init__(self, nDef, scale=2.5, seed=0, S=[0.5, 0.5]):
        '''
        '''
        self.nDef = nDef
        self.nFeats = 1
        self.Iu, self.Iv, self.Au, self.Av = ComputeSecurityGameOneSets(nDef)
        self.S = S

        r = np.random.RandomState(seed)
        self.dist = r.uniform(low=-scale, high=0., size=2)
        self.R1 = self.dist
        self.S = S

        self.sim = lambda ugt, vgt, nDef=nDef, R1=self.R1: SecurityGameOneSample(nDef, ugt, vgt, S[0:2])
        super(SecurityGameOneGen, self).__init__(self.Iu, self.Iv, self.Au, self.Av, self.sim, seed)

    
    def GenData(self, nSamples):
        P = np.zeros((nSamples, 2, self.nDef+1))
        F = np.ones((nSamples, self.nFeats))
        U = np.zeros((nSamples, 2)) # Action distributions
        V = np.zeros((nSamples, self.nDef+1))
        Au = np.zeros((nSamples,)) # Sampled actions taken by u-player
        Av = np.zeros((nSamples,))
        D = np.zeros((nSamples, 2)) # Distribution of cards

        # Create payoff matrices and solve for them
        for i in range(nSamples):
            P[i, :] = ComputeSecurityGameOnePayoffMatrix(self.nDef, self.R1, self.S[:2])
            U[i, :], V[i, :], Au[i], Av[i] = map(np.squeeze, self.GetActionSet(P[i, :]))
            D[i, :] = self.R1 

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

    if args.type=='SecurityGame':
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



