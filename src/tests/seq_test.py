from ..core.game import ZeroSumSequenceGame 
from ..core.solve import QRESeqLogit
from ..core.seq_paynet import ZSGSeqSolver

from ..poker.poker import OneCardPokerWNHDComputeP, OneCardPokerComputeSets, OneCardPokerWNHDSeqSample
from ..poker.poker import OneCardPokerPaynet

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
