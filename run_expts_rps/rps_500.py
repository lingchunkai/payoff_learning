import logging
logging.basicConfig(level=logging.INFO)
from experiments_rps_weights import Train
import numpy as np
import os

import types
args=types.SimpleNamespace()
args.loss='logloss'
args.datasize=500
args.lr=0.01
args.nEpochs=1000
args.monitorFreq=5
args.fracval=0.3
args.batchsize=128
args.scale=10.0
save_path_template = './results/rps/rps1_weights_big500/%d_%d/'
args.optimizer='adam'
args.softmax=False

nTests = 3
nTrials = 5 # trials per test
for i in range(nTests):
    for j in range(nTrials):
        save_folder = save_path_template % (i, j)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        args.save_path = save_folder + 'e'
        args.loadpath='./data/rps1_weights/data%d-%d.h5' % (i+1, j+1)
        print('Training argument', args.loadpath)
        Train(args)

