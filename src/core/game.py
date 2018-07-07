'''
Define two-player games
'''
import numpy as np
import scipy.stats
import itertools

class Game(object):
    '''
    Normal form game
    '''
    
    def __init__(self, Pu, Pv):
        self.P = [Pu, Pv]


    def ComputePayoff(self, u, v):
        '''
        Compute payoffs u^T Pu v and u^T Pv v.
        :return payoffs for player 1 and 2 (who supplied u and v, respectively)
        '''
        pay_u = np.dot(np.dot(u.T, self.P[0]), v)
        pay_v = np.dot(np.dot(u.T, self.P[1]), v)

        return pay_u, pay_v

class ZeroSumGame(Game): 
    def __init__(self, P): 
        '''
        Construct a 2-player game with P being the (n x m) payoff matrix.
        The vaue of the game is u^T P v, where u is the minimizing player.
        :param payoff matrix defined by (n x m) numpy array 
        '''
        self.Pz = P
        super(ZeroSumGame, self).__init__(-P, P)
    
    def SetP(self, p):
        self.Pz = p


class ZeroSumSequenceGame(ZeroSumGame):
    def __init__(self, Iu, Iv, Au, Av, P):
        '''
        :param Iu, Iv, list of list of actions for each information set, for min/max player respectively
        :param Au, Av, list of list of possible information sets following each action, for min/max player respectively
        :param P payoff matrix, *already* taking into account the chance nodes.
        '''
        self.Iu, self.Iv = Iu, Iv
        self.Au, self.Av = Au, Av

        # True if leaf action else False
        self.Lu = [len(Au[i]) == 0 for i in range(len(Au))]
        self.Lv = [len(Av[i]) == 0 for i in range(len(Av))]

        # index of parent (action) of given information set
        self.Par_inf_u = self.GetParentAction(Au, len(Iu))
        self.Par_inf_v = self.GetParentAction(Av, len(Iv))

        # index of parent (information set) of given action
        self.Par_act_u = self.GetParentInf(Iu, len(Au))
        self.Par_act_v = self.GetParentInf(Iv, len(Av))

        # index of parent parent (cached)
        self.ParPar_act_u = [None] * len(Au)
        self.ParPar_act_v = [None] * len(Av)
        for a in range(len(self.Au)): self.ParPar_act_u[a] = self.GetParParAction(a, 'u')
        for a in range(len(self.Av)): self.ParPar_act_v[a] = self.GetParParAction(a, 'v')

        self.ParPreprocess_u = np.ix_([x for x in self.ParPar_act_u if x is not None])
        self.ParPreprocess_v = np.ix_([x for x in self.ParPar_act_v if x is not None])
        self.ParIndices_u = np.ix_([a for a in range(len(Au)) if self.ParPar_act_u[a] is not None])
        self.ParIndices_v = np.ix_([a for a in range(len(Av)) if self.ParPar_act_v[a] is not None])

        # Construct E and F's
        self.Cu, self.cu = self.MakeConstraintMatrices(Iu, self.Par_inf_u, len(Au))
        self.Cv, self.cv = self.MakeConstraintMatrices(Iv, self.Par_inf_v, len(Av))
        
        # Checks on whether payoff matrices conform to requirements.
        # assert self.CheckPMatrix(self.Lu, self.Lv, P)
        # assert self.CheckTree(Iu, Au)
        # assert self.CheckTree(Iv, Av)

        super(ZeroSumSequenceGame, self).__init__(P)


    def ComputePayoff(self, u, v, tol=10.**-5):
        assert np.linalg.norm(np.dot(self.Cu, u).squeeze() - self.cu) < tol, 'dot=%s cu=%s diff=%s' % (np.dot(self.Cu, u), self.cu, np.dot(self.Cu, u) - self.cu)
        assert np.linalg.norm(np.dot(self.Cv, v).squeeze() - self.cv) < tol, 'dot=%s cv=%s diff=%s' % (np.dot(self.Cv, v), self.cv, np.dot(self.Cv, v) - self.cv)
        return super(ZeroSumSequenceGame, self).ComputePayoff(u, v)


    def ComputeProbs(self, player, uv):
        z = np.squeeze(uv)
        if player == 'u' or player == 0:
            ret = np.ones([uv.size])
            if self.ParIndices_u[0].size > 0:
                ret[self.ParIndices_u] = 1./z[self.ParPreprocess_u]
            ret *= z
        else:
            ret = np.ones([uv.size])
            if self.ParIndices_v[0].size > 0:
                ret[self.ParIndices_v] = 1./z[self.ParPreprocess_v]
            ret *= z

        return ret
        ret = np.zeros([uv.size])
        for a in range(uv.size):
            para = self.GetParParAction(a, player)
            if para is None: 
                ret[a] = uv[a]
            else:
                ret[a] = uv[a]/uv[para]
        return ret

    def MakeConstraintMatrices(self, I, Par_inf, nActions):
        C = np.zeros([len(I), nActions])
        c = np.zeros([len(I)])
        for i in range(len(I)):
            if Par_inf[i] is None: 
                c[i] = 1.
            else:
                C[i, Par_inf[i]] = -1.
            for a in I[i]:
                C[i, a] = 1.
        
        return C, c

        
    def GetParentAction(self, A, nInfoset):
        ret = [None] * nInfoset
        for idx, a in enumerate(A):
            for next_i in a:
                ret[next_i] = idx

        return ret

    
    def GetParentInf(self, I, nActions):
        ret = [None] * nActions
        for idx, i in enumerate(I):
            for next_a in i:
                ret[next_a] = idx

        return ret


    def CheckPMatrix(self, Lu, Lv, P):
        '''
        Check if payoff matrix has rewards delayed till leafs. True if alright.
        '''
        for i, j in itertools.product(range(P.shape[0]), range(P.shape[1])):
            if P[i][j] != 0.0 and not Lu[i] and not Lv[j]:
                return False

        return True


    def CheckTree(self, I, A, Par_inf=None):
        '''
        Check that single player game (fixed other player) 
        does not have any cycles in information sets
        '''

        if Par_inf is None: 
            Par_inf = self.GetParentAction(A, len(I))
        
        visited = [False] * len(I)
        for i in range(len(I)):
            if visited[i] is not None: continue # continue if not a root info set
            if visited[i]: return False
            to_visit = [i]
            visited[i] = True

            while(len(to_visit) > 0):
                i = to_visit.pop()
                for a in I[i]:
                    for next_i in A[a]:
                        if visited[next_i]: return False
                        visited[next_i] = True
                
        return True

        
    def GetParParAction(self, a, player):
        if player == 'u' or player == 0:
            if self.ParPar_act_u[a] is not None: 
                return self.ParPar_act_u[a]
            par = self.Par_act_u[a]
            para = self.Par_inf_u[par]
            self.ParPar_act_u[a] = para
        else:
            if self.ParPar_act_v[a] is not None: 
                return self.ParPar_act_v[a]
            par = self.Par_act_v[a]
            para = self.Par_inf_v[par]
            self.ParPar_act_v[a] = para
        return para


    def GetParParInfo(self, i, player):
        if player == 0 or player == 'u':
            par = self.Par_inf_u(i)
            if par == None: return None
            pari = self.Par_act_u(par)
            return pari
        else:
            par = self.Par_inf_v(i)
            if par == None: return None
            pari = self.Par_act_v(par)
            return pari


    def ConvertReducedNormalForm(self):
        '''
        Each 1-0 sequence maps to an action in reduced normal form
        Only use on small games!
        '''
        
        def DFSGetStrategies(i, player):
            if player == 'u': 
                Iuv = self.Iu
                Auv = self.Au
            elif player == 'v':
                Iuv = self.Iv
                Auv = self.Av
            else: assert False, 'Invalid Player'
            ret = []
            for a in Iuv[i]:
                r = []
                for next_i in Auv[a]: # Careful of parallel information set!
                    r.append(DFSGetStrategies(next_i, player))
                if len(r) == 0: # Base case at the end of the dfs
                    x = np.zeros([len(Auv)])
                    x[a] = 1.
                    ret.append(x)
                else: # 
                    for X in itertools.product(*tuple(r)):
                        x = sum(X)
                        x[a] = 1.
                        ret.append(x)
            return ret

        # Run DFS on all root infosets and take cross products
        ru, U = [], []
        for root_u in [i for i in range(len(self.Iu)) if self.Par_inf_u[i] is None]:
            ru.append(DFSGetStrategies(root_u, 'u'))
        for u_cross_item in itertools.product(*ru):
            U.append(sum(u_cross_item))

        rv, V = [], []
        for root_v in [i for i in range(len(self.Iv)) if self.Par_inf_v[i] is None]:
            rv.append(DFSGetStrategies(root_v, 'v'))
        for v_cross_item in itertools.product(*rv):
            V.append(sum(v_cross_item))

        # Construct P_normal matrix
        P_normal = np.zeros([len(U), len(V)])
        for u_idx, rnf_strategy_u in enumerate(U):
            for v_idx, rnf_strategy_v in enumerate(V):
                P_normal[u_idx, v_idx] = self.ComputePayoff(rnf_strategy_u, rnf_strategy_v)[1] # Use max player

        return P_normal, U, V


    def ConvertStrategyToSeqForm(self, uv, uvRep):
        ret = np.zeros(uvRep[0].shape)
        for k in range(len(uv)):
            ret += uv[k] * uvRep[k]
        return ret

    # ========================================================================================================
    
    # Game specific stuff, TODO: refactor...

    # ========================================================================================================
    

    def MatchingPennies(left=1., right=1.):
        Iu = np.array([[0, 1]])
        Iv = np.array([[0, 1]])
        Au = np.array([[], []]) # all actions are terminal (from the player's perspective)
        Av = np.array([[], []])
        P = np.array([[left, 0.], [0., right]]) 
        zssg = ZeroSumSequenceGame(Iu, Iv, Au, Av, P)
        return zssg

    
    def PerfectInformationMatchingPennies(left=1., right=1.):
        Iu = np.array([[0, 1]]) # '0' is LEFT
        Iv = np.array([[0, 1], [2, 3]]) # Action's '0' and '2' are left
        Au = np.array([[], []])
        Av = np.array([[], [], [], []])
        P = np.array([[left, 0., 0., 0.], [0., 0., 0., right]])
        zssg = ZeroSumSequenceGame(Iu, Iv, Au, Av, P)
        return zssg


    def MaxSearchExample():
        Iu = np.array([[0]])
        Iv = np.array([[0, 1], [2, 3, 4], [5, 6], [7], [8], [9]])
        Au = np.array([[]])
        Av = np.array([[1], [2], [3], [4], [5], [], [], [], [], []])
        P = np.array([[0, 0, 0, 0, 0, 5., 10., 3., 3., 3.]])
        zssg = ZeroSumSequenceGame(Iu, Iv, Au, Av, P)
        return zssg


    def MinSearchExample():
        ''' Same as above '''
        Iv = np.array([[0]])
        Iu = np.array([[0, 1], [2, 3, 4], [5, 6], [7], [8], [9]])
        Av = np.array([[]])
        Au = np.array([[1], [2], [3], [4], [5], [], [], [], [], []])
        P = np.array([[0, 0, 0, 0, 0, 5., 10., 3., 3., 3.]]).T
        zssg = ZeroSumSequenceGame(Iu, Iv, Au, Av, P)
        return zssg


    def OneCardPoker(numCards = 4, paymentScale=1.):
        if numCards == 4:
            normalizing_factor = 1./4./3. # Probability (for chance nodes of each path)
            # Actions: 0-7: Even is for no raise, odd is for raise
            # Second stage: Next 4 information states (the previous action was to bet 0, second player raises)
            # Actions: 8-15: Even is for fold, odd is for raise
            Iu = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]])
            # 8 information states, 4 \times 2 (whether 1st player betted)
            # First 4 information states are for 1st player passing, next 4 if first player raise
            # Infostates 0-3 are for cards 0-3 respectively, 4-7 similarly
            # Even action is for pass, odd for raise
            Iv = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]) 
            Au = np.array([[4], [], [5], [], [6], [], [7], [], [], [], [], [], [], [], [], []])
            Av = np.array([[] for x in range(16)])
            P = np.zeros([16, 16])
            cmpmat =np.array([[0., 1., 1., 1.],
                            [-1., 0., 1., 1.],
                            [-1., -1., 0., 1.],             
                            [-1., -1., -1., 0.]])         
            # 1st player waits, second player waits -- compare cards
            P[np.ix_(range(0,8,2), range(0,8,2))] = cmpmat
            # 1st player waits, second player raises, first player waits(folds) (1)
            P[np.ix_([8, 10, 12, 14], [1, 3, 5, 7])] = np.ones([numCards, numCards]) - np.eye(numCards)
            # 1st player waits, second player raises, first follows (+-2)
            P[np.ix_(range(9,16,2),range(1,8,2))] = cmpmat * 2
            # First player raises, 2nd player either forfeits (-1) or fights (+-2). 0's are for impossible scenarios
            P[np.ix_(range(1,8,2), range(8,16,2))] = -(np.ones([numCards, numCards]) - np.eye(numCards)) # 2nd player forfeits after first player leaves
            P[np.ix_(range(1,8,2), range(9,16,2))] = cmpmat * 2
            P *= normalizing_factor * paymentScale 
            zssg = ZeroSumSequenceGame(Iu, Iv, Au, Av, P)
            return zssg

class PayoffMatrixGenerator(object):
    def __init__(self, usize, vsize, seed=0):
        self.r = np.random.RandomState(seed)
        self.usize = usize
        self.vsize = vsize
        

    def Mixture(self, nbasis, rng=(-1, 1)):
        '''
        Creates payoff of a convex combination of nbasis usize x vsize payoff matrices.
        '''
        proportion = self.r.uniform(0, high=rng[1], size=nbasis)
        proportion /= np.sum(proportion)

        basis = self.r.uniform(low=rng[0], high=rng[1], size=(self.usize, self.vsize, nbasis))
        P = np.sum(basis * proportion, axis=2)
        return P


    def Gram(self, rank, max_sum_sv=None, max_1_norm=None):
        '''
        Create low-rank matrix with number of singular values equal to rank.
        Constraints: 
        :param max_sum_sv Singular values will sum to max_sum_sv.
        :param max_1_norm Elementwise 1-norm
        '''
        assert max_sum_sv is None or max_1_norm is None,  \
                'Limitation cannot be simultaneously set on both 1-norm of singular values and element-wise 1-norm'

        if max_sum_sv is None: max_sum_sv = 1.0

        def ortho_matrix(n):
            return scipy.stats.ortho_group.rvs(n)

        U_ortho, V_ortho = ortho_matrix(self.usize), ortho_matrix(self.vsize)
        svs = self.r.uniform(low=0, high=1, size=rank)
        svs = svs / np.sum(svs) * max_sum_sv
        P = np.zeros([self.usize, self.vsize])
        for i in range(rank):
            P += np.outer(U_ortho[:, i], V_ortho[i, :]) * svs[i]

        # Scale to elementwise max-norm
        if max_1_norm is not None:
            scale = np.max(np.abs(P))
            P  = P / scale * max_1_norm

        return P
        

    def Uniform(self, rng=(-1, 1)):
        '''
        '''
        return self.r.uniform(low=rng[0], high=rng[1], size=(self.usize, self.vsize))
    

if __name__ == '__main__':
    P = np.array([[5, 1], [3, 4]])
    zsg = ZeroSumGame(P)
    print(zsg.Pz)
    print(zsg.P[0], zsg.P[1])

    print('Generated matrices')
    PMG = PayoffMatrixGenerator(5, 5, seed=0)
    print('Uniform:', PMG.Uniform())
    print('Mixture:', PMG.Mixture(nbasis=2, rng=(-1, 1)))
    print('Gram:', PMG.Gram(rank=3, max_sum_sv=1))
    print('Gram:', PMG.Gram(rank=3, max_1_norm=1.0))

    print('---- Sequence Form----')
    print('Matching pennies')
    zssg = ZeroSumSequenceGame.MatchingPennies()
    print('leaf_u:', zssg.Lu, 'leaf_v:', zssg.Lv)
    print('Par_inf_u:', zssg.Par_inf_u, 'Par_inf_v:', zssg.Par_inf_v)
    print('Par_act_u:', zssg.Par_act_u, 'Par_act_v:', zssg.Par_act_v)
    print('Constraints_u:', zssg.Cu, zssg.cu) 
    print('Constraints_v:', zssg.Cv, zssg.cv)
    u, v  = np.array([0.2, 0.8]), np.array([0.4, 0.6])
    print('Evaluating', u, v, zssg.ComputePayoff(u, v))
    print('Reduced normal form:', zssg.ConvertReducedNormalForm())
    print('')

    print('Perfect Information Matching Pennies (second player knows first players action)')
    zssg = ZeroSumSequenceGame.PerfectInformationMatchingPennies()
    print('leaf_u:', zssg.Lu, 'leaf_v:', zssg.Lv)
    print('Par_inf_u:', zssg.Par_inf_u, 'Par_inf_v:', zssg.Par_inf_v)
    print('Par_act_u:', zssg.Par_act_u, 'Par_act_v:', zssg.Par_act_v)
    print('Constraints_u:', zssg.Cu, zssg.cu)
    print('Constraints_v:', zssg.Cv, zssg.cv)
    u, v  = np.array([0.2, 0.8]), np.array([0.3, 0.7, 0.1, 0.9])
    print('Evaluating', u, v, zssg.ComputePayoff(u, v))
    print('Reduced normal form:', zssg.ConvertReducedNormalForm())
    print('')

    print('Max Search Example')
    zssg = ZeroSumSequenceGame.MaxSearchExample()
    print('leaf_u:', zssg.Lu, 'leaf_v:', zssg.Lv)
    print('Par_inf_u:', zssg.Par_inf_u, 'Par_inf_v:', zssg.Par_inf_v)
    print('Par_act_u:', zssg.Par_act_u, 'Par_act_v:', zssg.Par_act_v)
    print('Constraints_u:\n', zssg.Cu, zssg.cu)
    print('Constraints_v:\n', zssg.Cv, zssg.cv)
    u, v = np.array([1.]), np.array([0.5, 0.5, 0.25, 0.125, 0.125, 0.25, 0.25, 0.25, 0.125, 0.125])
    print(np.dot(zssg.Cv,  v))
    print('Evaluating', u, v, zssg.ComputePayoff(u, v))
    print('Reduced normal form:', zssg.ConvertReducedNormalForm())
    print('')

    print('One-card poker with just 4 cards')
    zssg = ZeroSumSequenceGame.OneCardPoker(4)
    print('leaf_u:', zssg.Lu, 'leaf_v:', zssg.Lv)
    print('Par_inf_u:', zssg.Par_inf_u, 'Par_inf_v:', zssg.Par_inf_v)
    print('Par_act_u:', zssg.Par_act_u, 'Par_act_v:', zssg.Par_act_v)
    print('Constraints_u:', zssg.Cu, zssg.cu) 
    print('Constraints_v:', zssg.Cv, zssg.cv)
    print('')

