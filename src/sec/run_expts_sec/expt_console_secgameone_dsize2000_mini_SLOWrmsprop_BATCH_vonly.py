import logging
logging.basicConfig(level=logging.INFO)
from ..experiments_security_game_one import Train
# import matplotlib.pyplot as plt
import numpy as np
import os

import types
args=types.SimpleNamespace()

args.split_settings = 'const_val'
args.val_size = 2000

args.loss='vlogloss'
args.datasize=2000
args.lr=0.005
args.nEpochs=2000
args.monitorFreq=5
args.fracval=0.3
args.batchsize=128
args.tie_initial_raiseval=True
args.uniform_dist=False
args.single_feature=1
save_path_template = './results/secgame/secgameone_size2000_SLOWrmsprop_bsize_vonly/%d_%d/'
args.max_val=2.
args.success_rates='0.5,0.5'
args.static_rewards=1

args.momentum=0.1
args.optimizer='rmsprop'
args.fixbets=10

nTests = 3
nTrials = 10 # trials per test
for i in range(nTests):
    for j in range(nTrials):
        save_folder = save_path_template % (i, j)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        args.save_path = save_folder + 'e'
        args.loadpath='./data/SeqSecurityGameOne_scale2_unifsuc/data%d-%d.h5' % (i+1, j+1)
        print('Training argument', args.loadpath)
        Train(args)


