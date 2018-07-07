import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .security_game import SecurityGamePaynet 
from ..experiments.experiments import GameDataset, Evaluate
import logging, sys, pickle, copy
logger = logging.getLogger(__name__)

def Train(args):
    if not hasattr(args, 'split_settings') or args.split_settings == 'old':
        # 70-30 split, old settings. Data size includes both 70+30
        data = GameDataset(args.loadpath, size=args.datasize)
        usize, vsize = data.GetGameDimensions()
        nfeatures = data.GetFeatureDimensions()

        train_data, test_data = data, data.Split(frac=args.fracval, random=False)
        train_dl = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=1)
        test_dl = DataLoader(test_data, batch_size=test_data.__len__(), shuffle=False, num_workers=1)
    elif args.split_settings == 'const_val':
        # Fixed holdout set. datasize 
        train_data = GameDataset(args.loadpath, size=args.datasize)
        usize, vsize = train_data.GetGameDimensions()
        nfeatures = train_data.GetFeatureDimensions()

        test_data = GameDataset(args.loadpath, offset=-args.val_size, size=args.val_size)

        train_dl = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=1)
        test_dl = DataLoader(test_data, batch_size=test_data.__len__(), shuffle=False, num_workers=1)
    
    # Setup net,  optimizers, and loss
    lr = args.lr
    if hasattr(args, 'single_feature') and args.single_feature > 0: single_feature = True 
    else: single_feature=False
    if hasattr(args, 'static_rewards') and args.static_rewards > 0: static_rewards = True 
    else: static_rewards=False
    if hasattr(args, 'success_rates'): S = list(map(float, args.success_rates.split(',')))
    else: assert False, 'No success rate specified.'
    if hasattr(args, 'max_val'): max_val = args.max_val
    else: assert False, 'No max val specified'

    net = SecurityGamePaynet(int(vsize)-1, 
            S,
            max_val=max_val,
            verify='none', 
            single_feature=single_feature,
            static_rewards=static_rewards)

    net = net.double()
    print(net)
    # require_grad filter is a hack for freezing bet sizes
    print(args)
    if args.optimizer == 'rmsprop':
        optimizer=optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    elif args.optimizer == 'adam':
        optimizer=optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    elif args.optimizer == 'sgd':
        optimizer=optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.momentum, nesterov=True)

    from time import gmtime, strftime
    curdatetime = strftime("%Y-%m-%d_%H-%M-%S___", gmtime())
    args.save_path = args.save_path + curdatetime

    # Training proper
    losstrend_u, losstrend_v, losstrend_total = [], [], []
    monitor_t = []
    vpayofftrend, vmsetrend, vloglosstrend, vpayofftrend_train, vtarget_rewards = [], [], [], [], []

    # from pympler.tracker import SummaryTracker

    for i_epoch in range(args.nEpochs):
        t_lossu, t_lossv, t_loss = [], [], []
        for i_batch, sample_batched in enumerate(train_dl):
            loss, lossu, lossv = Evaluate(net, sample_batched, args.loss, optimizer)

            loss.backward(retain_graph=True)
            optimizer.step()

            t_lossu.append(lossu.data.numpy())
            t_lossv.append(lossv.data.numpy())
            t_loss.append(loss.data.numpy())

            sys.stdout.write('|')
            sys.stdout.flush()
        
        losstrend_u.append(np.mean(np.array(t_lossu)))
        losstrend_v.append(np.mean(np.array(t_lossv)))
        losstrend_total.append(np.mean(np.array(t_loss)))

        logger.info('Average train loss: %s, U loss: %s, V loss: %s ' % (losstrend_total[-1], losstrend_u[-1], losstrend_v[-1]))

        if i_epoch % args.monitorFreq == 0:

            for i_batch, sample_batched in enumerate(test_dl):

                loss, lossu, lossv = Evaluate(net, sample_batched, args.loss, optimizer)
                vloglosstrend.append(loss.data.numpy())
                logger.info('Total validation loss: %s, U loss: %s, V loss: %s' % (loss.data.numpy(), lossu.data.numpy(), lossv.data.numpy()))

                loss, _, _ = Evaluate(net, sample_batched, 'paymatrixloss', optimizer)
                vpayofftrend.append(loss.data.numpy())
                logger.info('Payoff loss: %s' % (loss.data.numpy()) )

                loss, _, _ = Evaluate(net, sample_batched, 'mse', optimizer)
                vmsetrend.append(loss.data.numpy())
                logger.info('Mse loss: %s' % (loss.data.numpy()) )

                loss, _, _ = Evaluate(net, sample_batched, 'target_rewards', optimizer)
                vtarget_rewards.append(loss.data.numpy())
                logger.info('Target Rewards: %s' % (loss.data.numpy()) )

                monitor_t.append(i_epoch)

            tloss = 0.0
            for i_batch, sample_batched in enumerate(train_dl):
                loss, _, _ = Evaluate(net, sample_batched, 'paymatrixloss', optimizer)
                tloss += loss.data.numpy()

            tloss /= i_batch+1
            vpayofftrend_train.append(tloss)
            logger.info('Payoff loss_train: %s' % (tloss) )

            # temporary save
            val_monitor = {'payofftrend': vpayofftrend, 
                            'msetrend': vmsetrend,
                            'loglosstrend': vloglosstrend,
                            'vtarget_rewards': vtarget_rewards}
            sto_dict = {'val_monitor': val_monitor,
                        'trainlosstrends': losstrend_total, 
                        'monitor_t': monitor_t}
            #            # 'net': net}
            fname = args.save_path + '%06d' % (i_epoch) + '.p'
            pickle.dump(sto_dict, open( fname, 'wb'))

        val_monitor = {'payofftrend': vpayofftrend, 
                        'msetrend': vmsetrend,
                        'loglosstrend': vloglosstrend,
                        'vtarget_rewards': vtarget_rewards}

    return losstrend_total, val_monitor, monitor_t, net

if __name__  == '__main__':
    print('Old tests removed. Run through Train() instead')
