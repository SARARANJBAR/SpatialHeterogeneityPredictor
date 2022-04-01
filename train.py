#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 08:02:21 2022

@author: Sara Ranjbar
Last updated: March 20th, 2022
"""

import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from torch.nn import BCELoss
from torch.autograd import Variable
from util.saveutil import save_model, save_plots, SaveBestModel
from models import networks
from sklearn.model_selection import StratifiedKFold

torch.backends.cudnn.enabled = False

def validate_model(model, validationLoader, criterion):
    """Validate the model on the separate validation set."""
    valid_running_loss = 0.0
    counter = 0

    # reduction is needed for validatio set because we have no class weights
    if criterion.reduction == 'none':
        criterion.reduction = 'mean'

    # define final result softmax
    m = nn.Softmax(dim=1)

    with torch.no_grad():

        for i, data in enumerate(validationLoader):

            counter += opt.batchSize

            inputs_ = Variable(data['tensor'].cuda())
            labels = Variable(data['label'].float())

            # forward pass
            outputs = m(model.forward(inputs_.detach()).cpu())

            # calculate loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

    epoch_loss = valid_running_loss / counter
    return epoch_loss

def train_model(opt, trainloader, valloader,
                testretestloader, model,
                fold, resultsfolder, runname):
    """Train a model to predict the abbundance of a marker of interest.

    2 datasets are used: MRI-localized biopsies (labeled) and test-retest
    (unlabeled)
    """

    # Set fixed random number seed
    torch.manual_seed(42)

    # define loss functions
    if opt.class_weight:
        # so it allows for assigning class weight to elements
        criterion = BCELoss(reduction='none')
        print('******* using loss function with class weight *******')
    else:
        criterion = BCELoss()

    # define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=opt.lr,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=opt.weight_decay,
                                  amsgrad=False)

    # define final result softmax
    m = nn.Softmax(dim=1)

    # initiate values
    old_lr = opt.lr

    # initialize SaveBestModel class
    save_best_model = SaveBestModel()

    # define path for saving models
    save_model_dir = os.path.join(opt.checkpoints_dir, opt.name)

    # lists to keep track of losses and accuracies
    loss_dct = {'train_loss': [],
                'labeled_loss': [],
                'unlabeled_loss': [],
                'val_loss': []}

    # start training
    counter = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

        train_running_loss = 0.0

        # loop over training dataset
        for i, data in enumerate(trainloader):

            iter_start_time = time.time()

            counter += opt.batchSize
            inputs_ = Variable(data['tensor'].cuda())
            labels = Variable(data['label'].float())

            # clearing the gradients of the model parameters
            optimizer.zero_grad()

            # forward pass
            outputs = m(model.forward(inputs_.detach()).cpu())

            # calculate the loss (with weights)
            if opt.class_weight:

                # create weights for labels
                labels_bin = [a[1].numpy().item() for a in labels]

                # we want class 0 to have less influence than class 1
                # ratio can be adjusted, lets start with 1:2
                weights = [[0.5*(a+1)] for a in labels_bin]
                weights = torch.tensor(weights)

                # mean is used because reduction is 'none'
                intermediate_loss = criterion(outputs, labels)
                loss = torch.mean(weights*intermediate_loss)

            else:
                loss = criterion(outputs, labels)

            train_running_loss += loss.item()

            # backpropagation
            loss.backward()

            # update the optimizer parameters
            optimizer.step()

        # training loss and accuracy for the complete epoch
        epoch_train_loss = train_running_loss / counter

        # end of training with labeled data for this epoch
        # use the model on test-retest samples
        if testretestloader:

            print('..test-retest addition..')
            # th for prediction confidence
            thresh = 0.75

            #  ensure there is no weighting for BCE loss here
            criterion.reduction = 'mean'

            # not sure if we need to do this or not
            optimizer.zero_grad()

            uloss = get_unlabeled_loss(testretestloader, model,
                                       thresh, criterion)

            sum_loss = epoch_train_loss + uloss * float(opt.lambda_u)
            print('--> labeled loss:', epoch_train_loss,
                  '- test-retest loss: ', uloss,
                  '- total:', sum_loss)

            loss_dct['labeled_loss'].append(epoch_train_loss)
            loss_dct['unlabeled_loss'].append(uloss)

            epoch_train_loss = sum_loss

            # propagate loss back
            loss = torch.tensor(epoch_train_loss, requires_grad=True)

            # backpropagation
            loss.backward()

            # update the optimizer parameters
            optimizer.step()

        # validate on validation set
        epoch_val_loss = validate_model(model, valloader, criterion)

        loss_dct['train_loss'].append(epoch_train_loss)
        loss_dct['val_loss'].append(epoch_val_loss)

        t = (time.time() - iter_start_time) / opt.batchSize
        print(t, 'secs - fold:', fold,
              ' - epoch:', epoch,
              '- train loss:', epoch_train_loss,
              '- val loss:', epoch_val_loss)

        # save checkpoint, return state_dict for loading
        save_best_model(epoch_val_loss, epoch, model,
                        optimizer, criterion, save_model_dir, fold)
        print('-'*100)

        # update learning rate
        if epoch > opt.niter:

            lrd = opt.lr / opt.niter_decay
            lr = old_lr - lrd
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            print('update learning rate: %f -> %f' % (old_lr, lr))
            old_lr = lr

    # save the trained model weights for a final time
    save_model(fold, model, optimizer, criterion, save_model_dir)

    # save the loss and accuracy plots
    save_plots(loss_dct, resultsfolder, fold, runname)

    print('TRAINING COMPLETE.')
    return save_best_model.get_modelpath()

def main(opt, k_folds=5):

    # define path to results folder
    resultsfolder = opt.result_path
    if not os.path.exists(resultsfolder):
        os.makedirs(resultsfolder)

    # load all of labeled data
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print('len(dataset)', len(dataset))

    # load test-retest (unlabeled) as the second val dataset if lambda_u>0
    if opt.lambda_u > 0:

        opt2 = opt
        opt2.dataset_mode = 'unlabeled'
        data_loader2 = CreateDataLoader(opt2)
        testretestloader = data_loader2.load_data()

    else:

        testretestloader = None

    # prepare output dataframe
    out_df = data_loader.dataset.get_dataset_labels_df()
    y = out_df[opt.target + ' LI'].values
    y = [int(v) for v in y]

    print(out_df.head())
    out_df['pred'] = None
    out_df['fold'] = None

    ## define final result softmax
    m = nn.Softmax(dim=1)

    # create train, validation, test sets during the K-fold cv
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=False)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset.dataset, y)):

        # select 30% of train ids for validation (stratifying by class)
        train_y = [y[i] for i in train_ids]
        train_ids, val_ids = train_test_split(train_ids, test_size=0.3,
                                              random_state=77,
                                              stratify=train_y)

        # double check ratios are OK
        train_y = [y[i] for i in train_ids]
        val_y = [y[i] for i in val_ids]
        print('train set size:', len(train_ids),
              'class ratio:', float(sum(train_y))/len(train_ids))
        print('val set size:', len(val_ids),
              'class ratio:', float(sum(val_y))/len(val_ids), '\n')

        # Sample elements using ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(dataset.dataset,
                                                  batch_size=opt.batchSize,
                                                  sampler=train_subsampler)

        valloader = torch.utils.data.DataLoader(dataset.dataset,
                                                batch_size=opt.batchSize,
                                                sampler=val_subsampler)

        testloader = torch.utils.data.DataLoader(dataset.dataset,
                                                 batch_size=1,
                                                 sampler=test_subsampler)

        # create the net, and parallelize on GPUs
        net = networks.define_network(opt.which_model,
                                      len(opt.input_seq),
                                      opt.norm,
                                      not opt.no_dropout,
                                      opt.gpu_ids).cuda().float()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            print('............\n Found', torch.cuda.device_count(), 'GPUs!')
            model = nn.DataParallel(net)
            print(model)
            print(model.device_ids)
            print('........... Using parallel GPUs (watch nvidia-smi)')

        model.to(device)

        # train and validate, keep track of best model
        best_model_path = train_model(opt, trainloader, valloader,
                                      testretestloader, model, fold,
                                      resultsfolder, opt.name)

        # load best model
        print('\n----loading best checkpoint: ', best_model_path)

        ckpnt = torch.load(best_model_path)
        model.load_state_dict(ckpnt['model_state_dict'])

        # test model (best model not the final model)
        print('testing the model for this fold')

        test_preds = []
        with torch.no_grad():
            for i, data in enumerate(testloader):

                # Generate outputs
                input_ = Variable(data['tensor'].cuda())

                prob = m(model.forward(input_.detach()).cpu())
                pred = (prob > 0.5).float()

                # convert to 0 and 1 for saving
                if torch.equal(pred, torch.tensor([[1., 0.]])):
                    test_preds.append(0)
                elif torch.equal(pred, torch.tensor([[0., 1.]])):
                    test_preds.append(1)
                else:
                    test_preds.append(-1)

        out_df.loc[test_ids, 'pred'] = test_preds
        out_df.loc[test_ids, 'fold'] = fold

    # print statistics of predictions
    allpreds = out_df['pred'].values
    print('done. range of preds:', min(allpreds), '-', max(allpreds))

    # save predictions to csv
    predfilepath = os.path.join(resultsfolder, opt.name+'-cv-predictions.csv')
    out_df.to_csv(predfilepath, index=False)

    # return path to prediction csv file
    return predfilepath

def get_unlabeled_loss(dataset, model, thresh, criterion):
    """Calculate unlabeled loss."""

    unlabeled_losses = []
    m = nn.Softmax(dim=1)

    for i, data in enumerate(dataset):

        pred_tensors = {'date1': torch.tensor([]),
                        'date2': torch.tensor([])}

        # get prediction for date1 and date 2
        for tensor_, name in zip([data['tensor1'], data['tensor2']],
                                 ['date1', 'date2']):

            input_ = Variable(tensor_.cuda())

            outputs = m(model.forward(input_.detach()).cpu())
            outputs = outputs.detach().numpy()

            # convert outputs to predictions
            if name == 'date1':
                preds = (outputs >= thresh).astype(np.float)
                pred_tensors[name] = torch.tensor(preds)
            else:
                pred_tensors[name] = torch.tensor(outputs.astype(np.float))

        loss = criterion(pred_tensors['date2'], pred_tensors['date1'])

        unlabeled_losses.append(loss.item())


    if len(unlabeled_losses) > 0:
        mean_ = float(np.mean(unlabeled_losses))
        print('average unlabeled loss: %.4f' % mean_)
        return mean_
    else:
        print('no unlabeled loss')
        return 0.0

def measure_cv_performance(csvpath):
    """Measures performance per fold and overall.
    Todo: needs modules.
    """

    csvname = os.path.basename(csvpath)
    csvname = csvname.split('-pred')[0]
    csvfolder = os.path.dirname(csvpath)

    if 'criterion' not in csvname:
        print('Error! criterion not in ', csvname)
        return

    # load csv in dataframe
    df = pd.read_csv(csvpath)

    # make sure there are predictions in the csv file
    if 'pred' not in df.columns.values:
        print('no prediction found in dataframe')
        return

    print('\n---csv name:\n%s\n' % csvname)
    print('\n---measuring performance of non-augmented rows')

    # skip augmented rows, they should not count toward performance measures
    df1 = df[df['flipaxis'] == -1]
    df2 = df1[df1['rot'] == -1]

    df = df2.copy()
    df.reindex(list(np.arange(df2.shape[0])))

    # idenify target column
    for col in ['Ki67 LI', 'SOX2 LI']:
        if col in df.columns.values:
            target = col

    # add tp, fp, fn, tn to dataframe
    truth = df.loc[:, target]
    preds = df.loc[:, 'pred']

    tps = [1 if t == 1 and p == 1 else 0 for t, p in zip(truth, preds)]
    fns = [1 if t == 1 and p == 0 else 0 for t, p in zip(truth, preds)]
    tns = [1 if t == 0 and p == 0 else 0 for t, p in zip(truth, preds)]
    fps = [1 if t == 0 and p == 1 else 0 for t, p in zip(truth, preds)]

    df['tp'] = tps
    df['fp'] = fps
    df['tn'] = tns
    df['fn'] = fns

    stats_df = pd.DataFrame(index=np.arange(6*4),
                            columns=['metrics', 'group', 'value'])

    # calculate overall stats
    print('\n---overall performance:')
    sum_tps = sum(tps)
    sum_tns = sum(tns)
    sum_fps = sum(fps)
    sum_fns = sum(fns)
    N = df.shape[0]

    if sum_tps + sum_fns > 0:
        sensitivity = float(sum_tps) / (sum_tps + sum_fns)
        print('overall sensitivity:', sensitivity)
    else:
        sensitivity = 'nan'

    if sum_tns + sum_fps > 0:
        specificity = float(sum_tns) / (sum_tns + sum_fps)
        print('overall specificity:', specificity)
    else:
        specificity = 'nan'

    f1 = float(sum_tps) / (sum_tps + 0.5 * (sum_fps + sum_fns))
    print('overall f1:', f1)

    accuracy = float(sum_tps + sum_tns) / N
    print('overall accuracy:', accuracy)

    stats_df.loc[0, :] = ['sensitivity', 'overall', sensitivity]
    stats_df.loc[1, :] = ['specificity', 'overall', specificity]
    stats_df.loc[2, :] = ['accuracy', 'overall', accuracy]
    stats_df.loc[3, :] = ['f1', 'overall', f1]

    # calculate per fold stats
    index = 4
    for i in range(5):
        print('-fold%d' % i)
        rows = df[df['fold'] == i]
        tps = rows['tp'].values
        fps = rows['fp'].values
        tns = rows['tn'].values
        fns = rows['fn'].values

        sum_tps = sum(tps)
        sum_tns = sum(tns)
        sum_fps = sum(fps)
        sum_fns = sum(fns)
        N = rows.shape[0]
        if sum_tns + sum_fps > 0:
            sensitivity = float(sum_tps) / (sum_tps + sum_fns)
            print('sensitivity:', sensitivity)
        else:
            sensitivity = 'nan'

        if sum_tns + sum_fps > 0:
            specificity = float(sum_tns) / (sum_tns + sum_fps)
            print('specificity:', specificity)
        else:
            specificity = 'nan'

        f1 = float(sum_tps) / (sum_tps + 0.5 * (sum_fps + sum_fns))
        print('f1:', f1)

        accuracy = float(sum_tps + sum_tns) / N
        print('accuracy:', accuracy)

        stats_df.loc[index, :] = ['sensitivity', 'fold' + str(i), sensitivity]
        index += 1

        stats_df.loc[index, :] = ['specificity', 'fold' + str(i), specificity]
        index += 1

        stats_df.loc[index, :] = ['accuracy', 'fold' + str(i), accuracy]
        index += 1

        stats_df.loc[index, :] = ['f1', 'fold' + str(i), f1]
        index += 1
        print('\n')

    outputpath = os.path.join(csvfolder, csvname + '_performance.csv')
    stats_df.to_csv(outputpath, index=False)
    print(stats_df)
    print('\n***Done!***\n')


train_start_time = time.time()
opt = TrainOptions().parse()
prediction_csvpath = main(opt)
measure_cv_performance(prediction_csvpath)
print(time.time() - train_start_time, 'seconds.')
print('.'*100)
