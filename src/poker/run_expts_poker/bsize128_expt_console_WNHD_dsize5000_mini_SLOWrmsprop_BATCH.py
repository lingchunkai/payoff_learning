import logging
logging.basicConfig(level=logging.INFO)
from ..experiments_onecardpoker import Train
import numpy as np
import os

import types

args=types.SimpleNamespace()

args.split_settings = 'const_val'
args.val_size = 2000

args.loss='logloss'
args.datasize=5000
args.lr=0.002
args.nEpochs=2500
args.monitorFreq=20
args.fracval=0.3
args.batchsize=128
args.tie_initial_raiseval=True
args.uniform_dist=False
args.single_feature=1
args.dist_type='WNHD'
save_path_template = './results/onecardpoker/WNHD_size5000_SLOWrmsprop_bsize128/%d_%d/'
args.initial_weights=[10.,0.25,0.25,0.25,0.25]

args.momentum=0.1
args.optimizer='rmsprop'
args.fixbets=10

nTests = 3
nTrials = 10 # trials per test
trainlosstrends, payofftrends, msetrends, cardprobstrends, loglosstrends, monitor_ts, nets = tuple([[None]*nTests for x in range(7)])
for i in range(nTests):
    # for j in range(nTrials):
    for j in [0,1,2,3,4]:
        save_folder = save_path_template % (i, j)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        args.save_path = save_folder + 'e'
        args.loadpath='./data/SeqOneCardPokerWNHDMiniExp_10_10_1/data%d-%d.h5' % (i+1, j+1)
        print('Training argument', args.loadpath)
        trainlosstrends[i], monitor, monitor_ts[i], nets[i] = Train(args)
        payofftrends[i], msetrends[i], loglosstrends[i], cardprobstrends[i] = monitor['payofftrend'], monitor['msetrend'], monitor['loglosstrend'], monitor['vcardprobstrend']


