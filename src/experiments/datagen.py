import numpy as np
import argparse
import h5py, scipy
from itertools import *
from ..core.game import PayoffMatrixGenerator, ZeroSumGame
from ..core.solve import GRELogit

# TODO: parse using kwargs
class DataGen(object):
    def __init__(self, usize, vsize, seed=0):
        self.usize, self.vsize = usize, vsize
        self.r = np.random.RandomState(seed)

    
    def GenData(self, nSamples):
        assert False, 'Base class, not implemented!'


    def GetActionSet(self, P):
        solver = GRELogit(ZeroSumGame(P))
        
        # Ground truth distributions
        U, V, _ = solver.Solve(alpha=0.1, beta=0.95, tol=10**-15, epsilon=10**-16, min_t=10**-20, max_it=100)

        # Sample actions from ground truths
        Au = self.r.choice(range(self.usize), 1, replace=True, p=U)
        Av = self.r.choice(range(self.vsize), 1, replace=True, p=V)

        return U, V, Au, Av


class RCP_weights(DataGen):
    def __init__(self, usize, vsize, nFeats, seed=0, scale=1.0, softmax=False):
        self.scale = scale
        assert usize==3 and vsize==3, 'Only implemented when number of actions is 3'
        self.size = usize
        self.nFeats = nFeats
        self.softmax=softmax
        super(RCP_weights, self).__init__(usize, vsize, seed)

    
    def GenData(self, nSamples):
        P = np.zeros((nSamples, self.size, self.size))
        F = np.zeros((nSamples, self.nFeats))
        U = np.zeros((nSamples, self.size)) # Action distributions
        V = np.zeros((nSamples, self.size))
        Au = np.zeros((nSamples,)) # Sampled actions taken by u-player
        Av = np.zeros((nSamples,))
        D = np.zeros((nSamples, self.size, self.nFeats)) # Distribution of cards
        weights = self.r.uniform(low=0., high=1.0, size=(self.size, self.nFeats))

        # Create payoff matrices and solve for them
        for i in range(nSamples):
            feats = self.r.uniform(low=0, high=1.0, size=self.nFeats)
            pays = np.dot(weights, feats)
            if self.softmax: pays=np.exp(pays)/np.sum(np.exp(pays))
            pays = pays * self.scale
            P[i, 0, 1], P[i, 1, 0] = pays[0], -pays[0]
            P[i, 0, 2], P[i, 2, 0] = -pays[1], pays[1]
            P[i, 1, 2], P[i, 2, 1] = pays[2], -pays[2]
            F[i, :] = feats
            D[i, :] = weights
            
            U[i, :], V[i, :], Au[i], Av[i] = self.GetActionSet(P[i, :, :])

        d = {'P': P, 'D': D, 'F': F, 'U': U, 'V': V, 'Au': Au, 'Av': Av}
        return d, (P, F, U, V, Au, Av)

         
def dict_to_h5(d, f):
    for k, v in d.items():
        f.create_dataset(k, data=v)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Generate dataset.',
            epilog='''Sample usage:\n
            python datagen.py --type RCP_weights --nFeats 2 --size 3 3 --scale 10 --nSamples 20000 --savePath ./data/rcp_weights_default.h5''')
    parser.add_argument('--type', type=str, default='RCP_weights', 
                    help='{RCP_weights}') 
    parser.add_argument('--size', type=int, required=True, nargs=2,
                    help='2 numbers for usize, vsize respectively. (nTargets, nResources) if type is AttackerDefender')
    parser.add_argument('--nFeats', type=int, default=2,
                    help='Number of features if we are generating payoffs from features')
    parser.add_argument('--scale', type=float, default=1.,
                    help='How much to scale game matrix by.')
    parser.add_argument('--nSamples', type=int, required=True,
                    help='Number of samples to generate.')
    parser.add_argument('--seed', type=int, default=0,
                    help='Randomization seed.')
    parser.add_argument('--savePath', type=str, default='./data/default_name.h5',
                    help='Path to save generated dataset.')
    parser.add_argument('--softmax', type=int, default=0,
                    help='1 if applying softmax prior to P.')

    args = parser.parse_args()

    f = h5py.File(args.savePath)
    if args.type=='RCP_weights':
        gen = RCP_weights(usize=args.size[0], vsize=args.size[1], nFeats=args.nFeats, seed=args.seed, scale=args.scale, softmax=True if args.softmax==1 else False)
        d, _ = gen.GenData(args.nSamples)
        dict_to_h5(d, f)
    else:
        assert False, 'Invalid type'

