'''
QRE solver for *zero sum* 2 player games
'''

import numpy as np
import scipy.sparse
from .game import ZeroSumGame

import logging
logger = logging.getLogger(__name__)

class Solver(object):
    def __init__(self, g):
        self.g = g
        self.usize = g.Pz.shape[0]
        self.vsize = g.Pz.shape[1]

    def CheckZeroSum():
        pass

    def Solve():
        pass


class GRELogit(Solver):
    
    def __init__(self, g):
        super(GRELogit, self).__init__(g)

    def Solve(self, alpha=0.1, beta=0.95, tol=10**-3, epsilon=10**-8, min_t=10**-20, max_it=50):
        '''
        Solve for QRE using Newton's method.
        :param alpha parameter for line search
        :param beta parameter for line search
        :param tol for termination condition
        :param epsilon minimum requirement for log barrier
        :param min_t minimum factor for line-search
        :param max_it maximum number of allowable iterations 
        :return u, v mixed strategy for QRE
        :return dev2, second derivative of KKT matrix
        '''

        def line_search(x, r, nstep):
            t = 0.9999
            while (np.any((x + t * nstep)[:-2] < epsilon) or # log-barrier requires special treatment when probabilities are near-zero
                    np.linalg.norm(residual(x + t * nstep)) > (1.0-alpha*t) * np.linalg.norm(r)):
                # print(np.linalg.norm(residual(x + t * nstep)), (1.0-alpha*t) * np.linalg.norm(r))
                t *= beta
                if t < min_t: 
                    break
            return t

        def term_cond(r):
            return True if np.linalg.norm(r) < tol else False

        def residual(x):
            return residual_(*unpack(x))

        def residual_(u, v, mu, vu):
            ru = np.atleast_2d(np.dot(self.g.Pz, v) + np.log(u) + 1. + mu * np.ones([self.usize])).T
            rv = np.atleast_2d(np.dot(self.g.Pz.T, u) - np.log(v) - 1 + vu * np.ones([self.vsize])).T
            rmu = [[np.sum(u) - 1.]]
            rvu = [[np.sum(v) - 1.]]
            r = np.concatenate([ru, rv, rmu, rvu], axis=0)
            return r

        def unpack(x):
            '''
            Break concatenated x into u, v, mu, vu
            :return u, v, mu, vu
            '''
            return np.split(x, np.cumsum([self.usize, self.vsize, 1]))

        # Initialize using all ones
        u = np.ones([self.usize]) / self.usize
        v = np.ones([self.vsize]) / self.vsize
        mu, vu = np.array([0.]), np.array([0.])
        
        for it in range(max_it):

            logger.debug('Iteration number %d', it)
            logger.debug('(u, v, mu, vu): %s %s %s %s', u, v, mu, vu)

            # Second derivative of KKT conditions
            matu = np.concatenate([np.diag(1./u), self.g.Pz, np.ones([self.usize, 1]), np.zeros([self.usize, 1])], axis=1)
            matv = np.concatenate([self.g.Pz.T, -np.diag(1./v), np.zeros([self.vsize, 1]), np.ones([self.vsize, 1])], axis=1)
            matmu = np.concatenate([np.ones([1, self.usize]), np.zeros([1, self.vsize]), np.array([[0]]), np.array([[0]])], axis=1)
            matvu = np.concatenate([np.zeros([1, self.usize]), np.ones([1, self.vsize]), np.array([[0]]), np.array([[0]])], axis=1)
            mat = np.concatenate([matu, matv, matmu, matvu], axis=0)
            
            # Constants in Newton's method
            r = residual_(u, v, mu, vu)
            logger.debug('Residual norm: %s', np.linalg.norm(r))
            if term_cond(r):
                logger.debug('Residual norm below tol at iteration %d. Terminating', it)
                break

            # Newton step
            nstep = np.squeeze(np.linalg.solve(mat, -r))
            logger.debug('Newton Step (u, v, mu, vu): %s %s %s %s', *unpack(nstep))

            # Line search
            x = np.concatenate([u, v, mu, vu], axis=0)
            t = line_search(x, r, nstep)
            x = x + t * nstep
            u, v, mu, vu = unpack(x)
            
            logger.debug('(u, v, mu, vu): %s %s %s %s', u, v, mu, vu)

            if np.all(np.abs(t * nstep) < tol): break
            
        if not self.VerifySolution(u, v, epsilon=0.0001):
            logger.warning('Solution verification failed! (u, v): %s %s', u, v)

        return u, v, mat
            

    def VerifySolution(self, u, v, epsilon=0.01):
        '''
        Verify if (u, v) is a fixed point of logit response
        :param numpy array, mixed strategy of min player
        :param numpy array, mixed strategy of max player
        :param Tolerance
        '''
        U = np.exp(-np.dot(self.g.Pz, v))
        V = np.exp(np.dot(self.g.Pz.T, u))
        ucheck = U/np.sum(U)
        vcheck = V/np.sum(V)
        checkPass = not (np.linalg.norm(ucheck - u) > epsilon or np.linalg.norm(vcheck - v) > epsilon)
        if checkPass:
            logger.debug('Verify solution (u, uCheck, v, vCheck): %s %s %s %s, checkPass: %s', u, ucheck, v, vcheck, checkPass)
        if not checkPass:
            logger.warning('Verify solution (u, uCheck, v, vCheck): %s %s %s %s, checkPass: %s', u, ucheck, v, vcheck, checkPass)
            logger.warning('P: %s', self.g.Pz)

        return checkPass


class QRESeqLogit(Solver):
    '''
    QRE for sequence form based on our proposed regularization.
    This is a generalization of the normal form representation.
    '''
    def __init__(self, g):
        self.uinfsize = len(g.Iu)
        self.vinfsize = len(g.Iv)

        self.Ju = np.array([float(len(a)) for a in g.Au])
        self.Jv = np.array([float(len(a)) for a in g.Av]) 

        super(QRESeqLogit, self).__init__(g)

        self.Mu = np.zeros([self.usize, self.usize])
        for a in range(self.usize):
            par = self.g.ParPar_act_u[a]
            if par is not None: self.Mu[par, a] = -1.
        self.Mu += np.diag((1.+self.Ju))/2.

        self.Mv = np.zeros([self.vsize, self.vsize])
        for a in range(self.vsize):
            par = self.g.ParPar_act_v[a]
            if par is not None: self.Mv[par, a] = -1.
        self.Mv += np.diag((1.+self.Jv))/2

        # for splitting x into u, v, mu, nu
        self.cuts = np.cumsum([self.usize, self.vsize, self.uinfsize])

    def GetFeasibleSolution(self, player):
        # DFS to get `uniform' solution 
        # TODO: refactor 

        if player == 'u':
            ret = [None] * self.usize
            stk = []
            root_infosets = [i for i in range(self.uinfsize) if self.g.Par_inf_u[i] is None]
            for i in root_infosets:
                avg = 1./float(len(self.g.Iu[i]))
                for a in self.g.Iu[i]:
                    stk.append((a, avg))

            while(len(stk) > 0):
                a, val = stk.pop()
                ret[a] = val
                for next_i in self.g.Au[a]:
                    avg = val/float(len(self.g.Iu[i]))
                    for next_a in self.g.Iu[next_i]:
                        stk.append((next_a, avg))
        
        else:
            ret = [None] * self.vsize
            stk = []
            root_infosets = [i for i in range(self.vinfsize) if self.g.Par_inf_v[i] is None]
            for i in root_infosets:
                avg = 1./float(len(self.g.Iv[i]))
                for a in self.g.Iv[i]:
                    stk.append((a, avg))

            while(len(stk) > 0):
                a, val = stk.pop()
                ret[a] = val
                for next_i in self.g.Av[a]:
                    avg = val/float(len(self.g.Iv[i]))
                    for next_a in self.g.Iv[next_i]:
                        stk.append((next_a, avg))

        return np.atleast_2d(ret).T

    def residual2_(self, u, v, mu, vu):
        uprob = self.g.ComputeProbs('u', u)
        vprob = self.g.ComputeProbs('v', v)

        #print('res', uprob)
        #print('res', vprob)

        total = 0.
        total += np.linalg.norm(np.squeeze(np.dot(self.g.Pz, v)) + np.log(uprob) + 1. \
            - self.Ju \
            + np.squeeze(np.dot(self.g.Cu.T, mu)), ord=2) ** 2
        # print(total)
        total += np.linalg.norm(np.squeeze(np.dot(self.g.Pz.T, u)) - np.log(vprob) - 1. \
            + self.Jv \
            + np.squeeze(np.dot(self.g.Cv.T, vu)), ord=2) ** 2
        # print(total)

        # If we start with feasible set, we could just assume these are 0...
        # total += np.linalg.norm(np.squeeze(np.dot(self.g.Cu, u)) - self.g.cu, ord=2) ** 2
        # total += np.linalg.norm(np.squeeze(np.dot(self.g.Cv, v)) - self.g.cv, ord=2) ** 2
        # print(total)
        return np.sqrt(total)

    def residual_(self, u, v, mu, vu):
        '''
        uprob = np.atleast_2d(self.g.ComputeProbs('u', u)).T
        vprob = np.atleast_2d(self.g.ComputeProbs('v', v)).T
        ru = np.atleast_2d(np.dot(self.g.Pz, v) + np.log(uprob) + 1. \
            - np.atleast_2d([len(a) for a in self.g.Au]).T \
            + np.dot(self.g.Cu.T, mu))
        rv = np.atleast_2d(np.dot(self.g.Pz.T, u) - np.log(vprob) - 1. \
            + np.atleast_2d([len(a) for a in self.g.Av]).T \
            + np.dot(self.g.Cv.T, vu))
        rmu = np.atleast_2d(np.squeeze(np.dot(self.g.Cu, u)) - self.g.cu).T
        rvu = np.atleast_2d(np.squeeze(np.dot(self.g.Cv, v)) - self.g.cv).T
        r = np.concatenate([ru, rv, rmu, rvu], axis=0)
        return r
        '''

        r = np.zeros([u.size + v.size + mu.size + vu.size])
        uprob = self.g.ComputeProbs('u', u)
        vprob = self.g.ComputeProbs('v', v)

        r[:self.cuts[0]] = np.squeeze(np.dot(self.g.Pz, v)) + np.log(uprob) + 1. \
            - self.Ju \
            + np.squeeze(np.dot(self.g.Cu.T, mu))

        r[self.cuts[0]:self.cuts[1]] = np.squeeze(np.dot(self.g.Pz.T, u)) - np.log(vprob) - 1. \
            + self.Jv \
            + np.squeeze(np.dot(self.g.Cv.T, vu))

        r[self.cuts[1]:self.cuts[2]] = np.squeeze(np.dot(self.g.Cu, u)) - self.g.cu
        r[self.cuts[2]:] = np.squeeze(np.dot(self.g.Cv, v)) - self.g.cv

        return r

    def Solve(self, alpha=0.1, beta=0.5, tol=10**-3, epsilon=10**-8, min_t=10**-20, max_it=50, verify='kkt'):
        '''
        Solve for QRE using Newton's method.
        :param alpha parameter for line search
        :param beta parameter for line search
        :param tol for termination condition
        :param epsilon minimum requirement for log barrier
        :param min_t minimum factor for line-search
        :param max_it maximum number of allowable iterations 
        :param verify {'kkt', 'rnf', 'both', 'none'}. Default: kkt
        :return u, v mixed strategy for QRE
        :return dev2, second derivative of KKT matrix
        '''

        def line_search(x, r, nstep):
            t = 0.9999
            nstep = np.atleast_2d(nstep).T
            rnorm2 = np.linalg.norm(r)
            # rnorm2 = residual2(x)
            # print(np.linalg.norm(r))
            while (np.any((x + t * nstep)[:self.cuts[1]] < epsilon) or # log-barrier requires special treatment when probabilities are near-zero
                   residual2(x + t * nstep) > (1.0-alpha*t) * rnorm2):
                   # np.linalg.norm(residual(x + t * nstep)) > (1.0-alpha*t) * np.linalg.norm(r)):
            # while (np.linalg.norm(residual(x + t * nstep)) > (1.0-alpha*t) * np.linalg.norm(r)):
                t *= beta
                if t < min_t: 
                    logger.warning('Line search terminated due to t < min_t (%s < %s)', t, min_t)
                    break
            return t

        def term_cond(r):
            return True if np.linalg.norm(r) < tol else False

        def residual(x):
            return self.residual_(*unpack(x))

        def residual2(x):
            ''' Same as above but only care about square of norm of r '''
            return self.residual2_(*unpack(x))

        def unpack(x):
            '''
            Break concatenated x into u, v, mu, vu
            :return u, v, mu, vu
            '''

            return x[:self.cuts[0]], x[self.cuts[0]:self.cuts[1]], x[self.cuts[1]:self.cuts[2]], x[self.cuts[2]:]
            # return np.split(x, np.cumsum([self.usize, self.vsize, self.uinfsize]))

        """
        def rho(uv, player):
            '''
            '''
            if player == 'u': size = self.usize
            else: size = self.vsize

            ret = np.zeros([size, size])
            for a in range(size):
                para = self.g.GetParParAction(a, player)
                if para is not None:
                    ret[a, self.g.GetParParAction(a, player)] = 1./(uv[para])
            
            return ret
           
        def kappa(uv, player):
            if player == 'u': size = self.usize
            else: size = self.vsize
            
            ret = np.zeros([size, size])
            for a in range(size):
                pa = self.g.GetParParAction(a, player)
                if pa is not None:
                    ret[pa, a] += -1./(uv[pa])

            return ret
        """
        """
        def DiagRhoKappaU(u):
            ''' Optimized version
            '''
            ret = np.diag(1./np.squeeze(u) * (1.+self.Ju))
            # ret = np.zeros([self.usize, self.usize])
            for a in range(self.usize):
                # ret[a,a] = 1./u[a] * (1. + self.Ju[a])

                pa = self.g.GetParParAction(a, 'u')
                if pa is not None:
                    ret[pa, a] += -1./u[pa]
                    ret[a, pa] += -1./u[pa]

            return ret

        """

        
        def DiagRhoKappaU(u):
            X = np.dot(np.diag(1./np.atleast_1d(np.squeeze(u))), self.Mu)
            return X + X.T


        def DiagRhoKappaV(v):
            X = np.dot(np.diag(1./np.atleast_1d(np.squeeze(v))), self.Mv)
            return -X - X.T

        """
        def DiagRhoKappaV(v):
            ret = np.diag(1./np.squeeze(v) * (1.+self.Jv))
            # ret = np.zeros([self.vsize, self.vsize])
            for a in range(self.vsize):
                #ret[a,a] = 1./v[a] * (1. + self.Jv[a])

                pa = self.g.GetParParAction(a, 'v')
                if pa is not None:
                    ret[pa, a] += -1./v[pa]
                    ret[a, pa] += -1./v[pa]

            return -ret # opposite from above
        """



        u = self.GetFeasibleSolution('u')
        v = self.GetFeasibleSolution('v')
        mu = np.zeros([self.uinfsize, 1])
        vu = np.zeros([self.vinfsize, 1])

        # Iniitalize Hessian
        matu = np.concatenate([np.zeros([self.usize, self.usize]), 
                                self.g.Pz, 
                                self.g.Cu.T, 
                                np.zeros([self.usize, self.vinfsize])], 
                                axis=1)
        matv = np.concatenate([self.g.Pz.T, 
                                np.zeros([self.vsize, self.vsize]), 
                                np.zeros([self.vsize, self.uinfsize]),
                                self.g.Cv.T], 
                                axis=1)
        matmu = np.concatenate([self.g.Cu, 
                                np.zeros([self.uinfsize, self.vsize]), 
                                np.zeros([self.uinfsize, self.uinfsize]),
                                np.zeros([self.uinfsize, self.vinfsize])], 
                                axis=1)
        matvu = np.concatenate([np.zeros([self.vinfsize, self.usize]), 
                                self.g.Cv, 
                                np.zeros([self.vinfsize, self.uinfsize]), 
                                np.zeros([self.vinfsize, self.vinfsize])], 
                                axis=1)
        mat = np.concatenate([matu, matv, matmu, matvu], axis=0)
       
        for it in range(max_it):
            logger.debug('Iteration number %d', it)
            logger.debug('(u, v, mu, vu): %s %s %s %s', u.T, v.T, mu.T, vu.T)
            mat[:self.usize, :self.usize] = DiagRhoKappaU(u) # \
                # np.diag((1./np.squeeze(u)) * (1. + self.Ju)) - rho(u, 'u') + kappa(u, 'u')
            mat[self.usize: self.usize+self.vsize, self.usize:self.usize+self.vsize] = DiagRhoKappaV(v) # \
                # -np.diag((1./np.squeeze(v)) * (1. + self.Jv)) + rho(v, 'v') - kappa(v, 'v')
            
            # Constants in Newton's method
            r = self.residual_(u, v, mu, vu)
            logger.debug('Residual norm: %s', np.linalg.norm(np.squeeze(r)))
            if term_cond(r):
                logger.debug('Residual norm below tol at iteration %d. Terminating', it)
                break

            # Newton step
            nstep = np.squeeze(np.linalg.solve(mat, -r))
            # nstep = np.squeeze(scipy.sparse.linalg.bicg(mat, -r, tol=tol**2)[0])
            logger.debug('Newton Step (u, v, mu, vu): %s %s %s %s', *unpack(nstep))

            # Line search
            x = np.concatenate([u, v, mu, vu], axis=0)
            t = line_search(x, r, nstep)
            x = x + t * np.atleast_2d(nstep).T
            u, v, mu, vu = unpack(x)

            if it > 100:
                logger.warning('Iteration number %s>100, t=%s, old_r=%s', it, t, r)

            '''
            rnew = residual_(u, v, mu, vu)
            logger.debug('r_old %s r_new %s' % (np.linalg.norm(np.squeeze(r)), np.linalg.norm(np.squeeze(rnew))))
            '''
            '''
            logger.debug('(u, v, mu, vu): %s %s %s %s', u.T, v.T, mu.T, vu.T)
            if np.all(np.abs(t * nstep) < tol): 
                logger.warning('Step size too small, terminating at iter %s!'%(it))
                break
            '''

        if (verify == 'rnf' or verify=='both') and not self.VerifySolutionRNF(u, v, epsilon=0.0001):
            logger.warning('RNF solution verification failed! (u, v): %s %s', u, v)
        elif (verify == 'kkt' or verify=='both') and not self.VerifySolutionKKT(u, mu, v, vu, epsilon=0.0001):
            logger.warning('KKT solution verification failed! (u, v): %s %s', u, v)
            
        return u, v, mat
            

    def VerifySolutionKKT(self, u, mu, v, nu, epsilon=0.01):
        
        def VerifyOnePlayer(p, lamb, x, player, epsilon=epsilon):
            '''
            :param p sequence form stategy
            :param lambda dual (lagrange multiplier), i.e. mu or nu
            :param player we are interested in
            :param epsilon tolerace

            By default, we are using the derivation for the min player (hence extra negative signs).
            *** IMPORTANT *** 
            Dual variable here may suffer from a sign change.
            *** IMPORTANT ***
            '''
            size = x.size
            Iuv = self.g.Iu if player=='u' or player==0 else self.g.Iv
            Auv = self.g.Au if player=='u' or player==0 else self.g.Av
            par_act = self.g.Par_act_u if player=='u' or player == 0 else self.g.Par_act_v

            checkPass = True
            # Check probabilities OK
            for i in range(size):
                par = self.g.GetParParAction(i, player)
                if par is None: pval = p[i] # No parent
                else: pval = p[i]/p[par]

                if player=='u' or player ==0:
                    checkProb = np.exp(-x[i] + len(Auv[i]) + np.sum(lamb[np.array(Auv[i]).astype(np.int64)]) - 1 - lamb[par_act[i]])
                else:
                    checkProb = np.exp(x[i] + len(Auv[i]) - np.sum(lamb[np.array(Auv[i]).astype(np.int64)]) - 1 + lamb[par_act[i]])

                if np.linalg.norm(pval - checkProb) < epsilon:
                    logger.debug('Verify action %s (%s, %sCheck): %s %s, result: %s', i, player, player, pval, checkProb, True)
                else:
                    logger.warning('Verify action %s (%s, %sCheck): %s %s, result: %s', i, player, player, pval, checkProb, False)
                    checkPass = False

            # Check duals OK
            for i in range(len(Iuv)):
                v = 0.
                if player=='u' or player==0:
                    for a_next in Iuv[i]:
                        v += np.exp(-x[a_next] + len(Auv[a_next]) + np.sum(lamb[np.array(Auv[a_next]).astype(np.int64)]))
                    checklamb = np.log(v) - 1
                else:
                    for a_next in Iuv[i]:
                        v += np.exp(x[a_next] + len(Auv[a_next]) - np.sum(lamb[np.array(Auv[a_next]).astype(np.int64)]))
                    checklamb = 1. - np.log(v)

                if np.linalg.norm(lamb[i] - checklamb) < epsilon:
                    logger.debug('Verify duals %s (%s, %sCheck): %s %s, result: %s', i, player, player, lamb[i], checklamb, True)
                else:
                    logger.warning('Verify duals %s (%s, %sCheck): %s %s, result: %s', i, player, player, lamb[i], checklamb, False)
                    checkPass = False

            return checkPass
        
        checkPass = True
        checkPass = checkPass and VerifyOnePlayer(u, mu, np.dot(self.g.Pz, v), player='u', epsilon=epsilon)
        checkPass = checkPass and VerifyOnePlayer(v, nu, np.dot(self.g.Pz.T, u), player='v', epsilon=epsilon)

        return checkPass


    def VerifySolutionRNF(self, u, v, epsilon=0.01):
        '''
        Verify if (u, v) is a fixed point of logit response 
        (our formulation, i.e. reduced normal form)
        :param numpy array, mixed strategy of min player
        :param numpy array, mixed strategy of max player
        :param Tolerance
        '''
        
        # Compute RNF QRE
        logger.debug('===================== Solving Reduced Normal Form Game Directly =========================')
        RNF_P, RNF_urep, RNF_vrep = self.g.ConvertReducedNormalForm()
        NormalFormSolver = GRELogit(game.ZeroSumGame(RNF_P))
        RNF_u, RNF_v, _ = NormalFormSolver.Solve(tol=10**-15, epsilon=10**-15, max_it=500)
        logger.debug('===================== End of Reduced Normal Form Solution  =========================')
        ucheck = self.g.ConvertStrategyToSeqForm(RNF_u, RNF_urep)
        vcheck = self.g.ConvertStrategyToSeqForm(RNF_v, RNF_vrep)

        checkPass = not (np.linalg.norm(np.squeeze(ucheck) - np.squeeze(u)) > epsilon \
                        or np.linalg.norm(np.squeeze(vcheck) - np.squeeze(v)) > epsilon)
        if checkPass:
            logger.debug('Verify RNF solution (u, uCheck, v, vCheck): %s %s %s %s, checkPass: %s', u.squeeze(), ucheck, v.squeeze(), vcheck, checkPass)
        if not checkPass:
            logger.warning('Verify RNF solution (u, uCheck, v, vCheck): %s %s %s %s, checkPass: %s', u.squeeze(), ucheck, v.squeeze(), vcheck, checkPass)

        return checkPass


if __name__ == '__main__':
    # Log to both file and stdout
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler('solver.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Simple Solver
    P = np.array([[5, 1], [3, 4]])
    zsg = game.ZeroSumGame(P)
    Solver = GRELogit(zsg)
    Solver.Solve()

    print('Any key to continue')
    input()

    # Bigger Solver
    P = np.array([[-10, 1, 2], [5, 2, 8], [3, 4, 3]])
    zsg = game.ZeroSumGame(P)
    Solver = GRELogit(zsg)
    Solver.Solve()

    # Translated payoff matrix
    P += 50
    zsg = game.ZeroSumGame(P)
    Solver = GRELogit(zsg)
    Solver.Solve()

    print('Any key to continue. Sequence form tests next')
    input()

    print('---- Sequence Form -----')
    zssg = game.ZeroSumSequenceGame.MatchingPennies(left=0.75, right=0.25)
    Solver = QRESeqLogit(zssg)
    Solver.Solve(max_it=100, verify='both')

    print('Any key to continue')
    input()

    zssg = game.ZeroSumSequenceGame.PerfectInformationMatchingPennies(left=3., right=1.)
    Solver = QRESeqLogit(zssg)
    Solver.Solve(max_it=1000, verify='both')

    print('Any key to continue')
    input()

    zssg = game.ZeroSumSequenceGame.MinSearchExample()
    Solver = QRESeqLogit(zssg)
    Solver.Solve(max_it=1000, tol=10**-7, verify='both')

    print('Any key to continue')
    input()

    zssg = game.ZeroSumSequenceGame.MaxSearchExample()
    Solver = QRESeqLogit(zssg)
    Solver.Solve(max_it=1000, tol=10**-7, verify='both')

    print('Any key to continue')
    input()

    zssg = game.ZeroSumSequenceGame.OneCardPoker(4)
    Solver = QRESeqLogit(zssg)
    ugt, vgt, _ = Solver.Solve(max_it=1000, tol=10**-7, verify='both')
    print(zssg.ComputePayoff(ugt, vgt))

    print('Any key to continue')
    input()

    zssg = game.ZeroSumSequenceGame.OneCardPoker(4, 10.0)
    Solver = QRESeqLogit(zssg)
    ugt, vgt, _ = Solver.Solve(max_it=1000, tol=10**-7, verify='both')
    print(zssg.ComputePayoff(ugt, vgt))
