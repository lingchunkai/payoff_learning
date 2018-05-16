import numpy as np
import torch.optim as optim
import h5py
import argparse
import torch
import torch.nn as nn
import copy
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class GameDataset(Dataset):
    def __init__(self, loadpath, size=-1):
        '''
        :param size size of dataset to read -1 if we use the full dataset
        '''
        self.loadpath = loadpath
        f = h5py.File(loadpath, 'r')
        self.f = f
        self.feats, self.Au, self.Av, self.GTu, self.GTv, self.GTP = f['F'][:], f['Au'][:], f['Av'][:], f['U'][:], f['V'][:], f['P'][:]
        self.P = f['P']
        self.others = dict()
        if 'D' in f: 
            self.others['D'] = f['D']

        # By default, we use the full dataset
        self.offset, self.size = 0, self.feats.shape[0] if size == -1 else size

        super(Dataset, self).__init__()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx > self.size:
            raise IndexError('idx > size')
        i = idx + self.offset

        if len(self.others) == 0:
            return self.feats[i, :], self.Au[i], self.Av[i], self.GTu[i, :], self.GTv[i, :], self.GTP[i, :, :]
        else:
            return self.feats[i, :], self.Au[i], self.Av[i], self.GTu[i, :], self.GTv[i, :], self.GTP[i, :, :], self.others['D'][i, :] # TODO: fix

    def GetGameDimensions(self):
        return tuple(self.P.shape[1:])

    def GetFeatureDimensions(self):
        return self.feats.shape[-1]

    def Split(self, frac=0.3, random=False):
        '''
        :param fraction of dataset reserved for the other side
        :param shuffle range of data
        return new dataset carved of size self.size * frac
        '''
        if random == True:
            pass #TODO: randomly shuffle data. Make sure only to shuffle stuff within the range!

        cutoff = int(float(self.size) * (1.0-frac))
        if cutoff == 0 or cutoff >= self.size:
            raise ValueError('Split set is empty or entire set')

        other = GameDataset(self.loadpath)
        other.offset = self.offset + cutoff
        other.size = self.size - cutoff
        self.size = cutoff
        
        return other
   
class AttackerDefenderDataset(GameDataset):
    def __init__(self, loadpath, size=-1):
        super(AttackerDefenderDataset, self).__init__(loadpath, size)


    def GetNumResources(self):
        try: nRes = self.f['nDef'].value
        except KeyError as e:
            print('Error in accessing dataset, potentially not an attacker-defender game?')
            raise
            
        return nRes


def Evaluate(net, sample_batched, loss_type, optimizer):
    nll = nn.NLLLoss()
    mse = nn.MSELoss()
    KLDivLoss = nn.KLDivLoss()

    optimizer.zero_grad()
    
    batched_data = tuple(sample_batched)
    # TODO: less hacky way of extracting parameters
    if len(batched_data) == 6:
        feats, Au, Av, GTu, GTv, GTP = batched_data
    elif len(batched_data) == 7:
        feats, Au, Av, GTu, GTv, GTP, others = batched_data

    feats = Variable(feats.double())
    Av = Variable(Av.long(), requires_grad=False)
    Au = Variable(Au.long(), requires_grad=False)
    GTu = Variable(GTu.double(), requires_grad=False)
    GTv = Variable(GTv.double(), requires_grad=False)
    GTP = Variable(GTP.double(), requires_grad=False)

    ret = net(feats)
    # TODO: less hacky way of extracting parameters
    if len(ret) == 3:
        U, V, P = ret
    elif len(ret) == 4: # for security game
        U, V, P, target_rewards = ret
    elif len(ret) == 5:
        U, V, P, betVal, cardProbs = ret # for OneCardPoker (TODO: refactor)

    if loss_type == 'mse':
        lossu = mse(U, GTu)
        lossv = mse(V, GTv)
        loss = lossu + lossv

    elif loss_type == 'logloss':
        lossv = nll(torch.log(V), Av)
        lossu = nll(torch.log(U), Au)
        loss = lossu + lossv

    elif loss_type == 'ulogloss':
        lossu = nll(torch.log(U), Au)
        lossv = lossu
        loss = lossu

    elif loss_type == 'vlogloss':
        lossv = nll(torch.log(V), Av)
        lossu = lossv
        loss = lossv

    elif loss_type == 'paymatrixloss':
        loss, lossu, lossv = mse(P, GTP), None, None

    elif loss_type == 'optimallogloss':
        GTlossu = nll(torch.log(GTu), Au)
        GTlossv = nll(torch.log(GTv), Av)
        lossu, lossv, loss = GTlossu, GTlossv, GTlossu + GTlossv

    elif loss_type == 'pokerparams_probs':
        GTcardprobs = Variable(others.double(), requires_grad=False)
        loss, lossu, lossv = mse(cardProbs, GTcardprobs), None, None

    elif loss_type == 'pokerparams_probs_joint_KLDiv':
        batchsize=others.size()[0]
        ncards=others.size()[1]

        GTcardprobs = Variable(others.double(), requires_grad=False)
        GTtotal_cards = GTcardprobs.sum(1)
        GTnormalizing_constant = GTtotal_cards * (GTtotal_cards - 1)
        GTouter = torch.bmm(GTcardprobs.unsqueeze(2), GTcardprobs.unsqueeze(1))
        GTcorrection = Variable(torch.DoubleTensor(np.zeros([batchsize, ncards, ncards])))
        for i in range(batchsize): # TODO: vectorize. As of Jan 2018, no block version of diag exists
            GTcorrection[i, :, :] = torch.diag(GTcardprobs[i, :])
        GTprobmat = (GTouter-GTcorrection)/GTnormalizing_constant.expand([ncards, ncards, batchsize]).transpose(0,2)
        GTprobmat_flat = GTprobmat.view(batchsize, -1) # technically no need for this

        total_cards = cardProbs.sum(1)
        normalizing_constant = total_cards * (total_cards - 1)
        outer = torch.bmm(cardProbs.unsqueeze(2), cardProbs.unsqueeze(1))
        correction = Variable(torch.DoubleTensor(np.zeros([batchsize, ncards, ncards])))
        for i in range(batchsize): # TODO: vectorize. As of Jan 2018, no block version of diag exists
            correction[i, :, :] = torch.diag(cardProbs[i, :])
        probmat = (outer-correction)/normalizing_constant.expand([ncards, ncards, batchsize]).transpose(0,2)
        probmat_flat = probmat.view(batchsize, -1) # technically no need for this

        loss, lossu, lossv = KLDivLoss(torch.log(probmat_flat), GTprobmat_flat), None, None

    elif loss_type == 'pokerparams_WNHD_probs_joint_KLDiv':
        
        batchsize=others.size()[0]
        ncards=others.size()[1]

        GTcardprobs = Variable(others.double(), requires_grad=False)
        GTouter = torch.bmm(GTcardprobs.unsqueeze(2), GTcardprobs.unsqueeze(1))
        GTcorrection = Variable(torch.DoubleTensor(np.zeros([batchsize, ncards, ncards])))
        GTnormalizing_constant = Variable(torch.DoubleTensor(np.zeros([batchsize, ncards, ncards])))
        for i in range(batchsize): # TODO: vectorize. As of Jan 2018, no block version of diag exists
            GTcorrection[i, :, :] = torch.diag(GTcardprobs[i, :]) ** 2
            GTnormalizing_constant[i, :, :] = torch.diag(1./(1.0 - GTcardprobs[i, :]))
        GTprobmat = torch.bmm(GTnormalizing_constant, GTouter-GTcorrection)
        GTprobmat_flat = GTprobmat.view(batchsize, -1) # technically no need for this

        outer = torch.bmm(cardProbs.unsqueeze(2), cardProbs.unsqueeze(1))
        correction = Variable(torch.DoubleTensor(np.zeros([batchsize, ncards, ncards])))
        normalizing_constant = Variable(torch.DoubleTensor(np.zeros([batchsize, ncards, ncards])))
        for i in range(batchsize): # TODO: vectorize. As of Jan 2018, no block version of diag exists
            correction[i, :, :] = torch.diag(cardProbs[i, :]) ** 2
            normalizing_constant[i, :, :] = torch.diag(1./(1.0 - cardProbs[i, :]))
        probmat = torch.bmm(normalizing_constant, outer-correction)
        probmat_flat = probmat.view(batchsize, -1) # technically no need for this

        loss, lossu, lossv = KLDivLoss(torch.log(probmat_flat), GTprobmat_flat), None, None

    elif loss_type == 'target_rewards':
        GTtarget_rewards = Variable(others.double(), requires_grad=False)
        loss, lossu, lossv = mse(target_rewards, GTtarget_rewards), None, None

    else: assert False, 'Invalid loss type specified'

    return loss, lossu, lossv

if __name__ == '__main__':
    print('Old tests removed.')    
