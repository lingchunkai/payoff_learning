import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .poker import OneCardPokerPaynet
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
    
    # Set weight initialization if we have to
    if hasattr(args, 'initial_params') and len(args.initial_params) > 0:
        print(len(args.initial_params))
        # parse parameter splits
        x = args.initial_params.split(',')
        initial_params = list(map(float, x))
    else: initial_params = None

    if args.fixbets is None or args.fixbets < 0:
        if args.fixbets < 0.:
            print('args.fixbets < 0, Setting to no fix bets!')
        fixbets = None
    else: fixbets = args.fixbets

    # Setup net,  optimizers, and loss
    lr = args.lr
    if hasattr(args, 'single_feature') and args.single_feature > 0: single_feature = True 
    else: single_feature=False
    if not hasattr(args, 'dist_type'): args.dist_type='original'
    net = OneCardPokerPaynet(int(usize/4), 
            tie_initial_raiseval=True, 
            uniform_dist=False, 
            initial_params=initial_params, 
            fixbets=args.fixbets, 
            verify='none', 
            single_feature=single_feature,
            dist_type=args.dist_type)

    net = net.double()
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
    vpayofftrend, vmsetrend, vloglosstrend, vpayofftrend_train, vcardprobstrend, vcardprobs_joint_KL_trend = [], [], [], [], [], []

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

            # tracker = SummaryTracker()
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

                loss, _, _ = Evaluate(net, sample_batched, 'pokerparams_probs', optimizer)
                vcardprobstrend.append(loss.data.numpy())
                logger.info('Car probs loss: %s' % (loss.data.numpy()) )

                loss, _, _ = Evaluate(net, sample_batched, 'pokerparams_WNHD_probs_joint_KLDiv', optimizer)
                vcardprobs_joint_KL_trend.append(loss.data.numpy())
                logger.info('KL Div Loss: %s' % (loss.data.numpy()) )

                monitor_t.append(i_epoch)
            
            # Hack to extract (ONLY FOR PRINTING) probabilities and bets learnt
            from torch.autograd import Variable # TODO: delete...
            z = Variable(sample_batched[0].double(), requires_grad=False)
            _, _, _, bets, probs = net(z)
            print(probs[0, :].data.numpy().squeeze())
            print(bets[0, :].data.numpy().squeeze())

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
                            'vcardprobstrend': vcardprobstrend,
                            'vcardprobs_joint_KL_trend': vcardprobs_joint_KL_trend}
            sto_dict = {'val_monitor': val_monitor,
                        'trainlosstrends': losstrend_total, 
                        'monitor_t': monitor_t}
            #            'net': net}
            fname = args.save_path + '%06d' % (i_epoch) + '.p'
            pickle.dump(sto_dict, open( fname, 'wb'))

            # tracker.print_diff()
        val_monitor = {'payofftrend': vpayofftrend, 
                        'msetrend': vmsetrend,
                        'loglosstrend': vloglosstrend,
                        'vcardprobstrend': vcardprobstrend,
                        'vcardprobs_joint_KL_trend': vcardprobs_joint_KL_trend}

    return losstrend_total, val_monitor, monitor_t, net

if __name__  == '__main__':
    print('Old tests have been removed. Run by calling Train()')

