# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import csv

from tl.utils.utils import str2bool
from utils.network import backbone_net
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar
from utils.utils import fix_random_seed, cal_acc_comb, data_loader, cal_auc_comb, cal_score_online, makedir_if_not_exist
from utils.alg_utils import EA, EA_online
from scipy.linalg import fractional_matrix_power
from utils.loss import Entropy
from sklearn.metrics import roc_auc_score, accuracy_score

import gc
import sys
import time

# This is the implementation of model BN-adapt from paper: 
# Schneider S, Rusak E, Eck L, et al. Improving robustness against common corruptions by covariate shift adaptation
# [J].Advances in neural information processing systems, 2020, 33: 11539-11551.
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : bn-adapt.py
# from github https://github.com/sylyoung/DeepTransferEEG/tree/main

def BN_adapt(loader, model, args, balanced=True):

    if balanced == False and args.data_name == 'BNCI2014001-4':
        print('ERROR, imbalanced multi-class not implemented')
        sys.exit(0)

    y_true = []
    y_pred = []

    # initialize test reference matrix for Incremental EA
    if args.align:
        R = 0

    iter_test = iter(loader)

    # loop through test data stream one by one
    for i in range(len(loader)):
        #################### Phase 1: target label prediction ####################
        model.eval()
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        inputs = inputs.reshape(1, 1, inputs.shape[-2], inputs.shape[-1]).cpu()

        # accumulate test data
        if i == 0:
            data_cum = inputs.float().cpu()
        else:
            data_cum = torch.cat((data_cum, inputs.float().cpu()), 0)

        # Incremental EA
        if args.align:
            start_time = time.time()

            if i == 0:
                sample_test = data_cum.reshape(args.chn, args.time_sample_num)
            else:
                sample_test = data_cum[i].reshape(args.chn, args.time_sample_num)
            # update reference matrix
            R = EA_online(sample_test, R, i)

            sqrtRefEA = fractional_matrix_power(R, -0.5)
            # transform current test sample
            sample_test = np.dot(sqrtRefEA, sample_test)

            EA_time = time.time()
            if args.calc_time:
                print('sample ', str(i), ', pre-inference IEA finished time in ms:', np.round((EA_time - start_time) * 1000, 3))
            sample_test = sample_test.reshape(1, 1, args.chn, args.time_sample_num)
        else:
            sample_test = data_cum[i].numpy()
            sample_test = sample_test.reshape(1, 1, sample_test.shape[1], sample_test.shape[2])

        if args.data_env != 'local':
            sample_test = torch.from_numpy(sample_test).to(torch.float32).cuda()
        else:
            sample_test = torch.from_numpy(sample_test).to(torch.float32)

        _, outputs = model(sample_test)

        softmax_out = nn.Softmax(dim=1)(outputs)

        outputs = outputs.float().cpu()
        labels = labels.float().cpu()
        _, predict = torch.max(outputs, 1)

        y_pred.append(softmax_out.detach().cpu().numpy())
        y_true.append(labels.item())

        #################### Phase 2: target model update ####################
        model.train()
        # sliding batch
        if (i + 1) >= args.test_batch and (i + 1) % args.stride == 0:
            if args.align:
                batch_test = np.copy(data_cum[i - args.test_batch + 1:i + 1])
                # transform test batch
                batch_test = np.dot(sqrtRefEA, batch_test)
                batch_test = np.transpose(batch_test, (1, 2, 0, 3))
            else:
                batch_test = data_cum[i - args.test_batch + 1:i + 1].numpy()
                batch_test = batch_test.reshape(args.test_batch, 1, batch_test.shape[2], batch_test.shape[3])

            if args.data_env != 'local':
                batch_test = torch.from_numpy(batch_test).to(torch.float32).cuda()
            else:
                batch_test = torch.from_numpy(batch_test).to(torch.float32)

            start_time = time.time()
            for step in range(args.steps):

                model[0].block1[2].train()
                model[0].block1[4].train()
                model[0].block2[3].train()

                # forward pass for model BN update
                _, outputs = model(batch_test)

                model[0].block1[2].eval()
                model[0].block1[4].eval()
                model[0].block2[3].eval()

            TTA_time = time.time()
            if args.calc_time:
                print('sample ', str(i), ', post-inference model update finished in ms:', np.round((TTA_time - start_time) * 1000, 3))

        model.eval()

    if balanced:
        _, predict = torch.max(torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.class_num), 1)
        pred = torch.squeeze(predict).float()
        score = accuracy_score(y_true, pred)
        if args.data_name == 'BNCI2014001-4':
            y_pred = np.array(y_pred).reshape(-1, )  # multiclass
        else:
            y_pred = np.array(y_pred).reshape(-1, args.class_num)  # binary
    else:
        predict = torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.class_num)
        y_pred = np.array(predict).reshape(-1, args.class_num)  # binary
        score = roc_auc_score(y_true, y_pred)

    return score * 100, (y_pred, predict, y_true)


def train_target(args):
    if not args.align:
        extra_string = '_noEA'
    else:
        extra_string = ''
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)
    
    args.sample_rate = 6  # to set EEGNet kernal as 3, should be modified 
    
    netF, netC = backbone_net(args, return_type='xy')
    if args.data_env != 'local':
        netF, netC = netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)

    if args.max_epoch == 0:
        if args.align:
            if args.data_env != 'local':
                base_network.load_state_dict(torch.load('./runs/' + str(args.data_name) + '/' + str(args.backbone) +
                    '_S' + str(args.idt) + '_seed' + str(args.SEED) + extra_string + '.ckpt'))
            else:
                base_network.load_state_dict(torch.load('./runs/' + str(args.data_name) + '/' + str(args.backbone) +
                    '_S' + str(args.idt) + '_seed' + str(args.SEED) + extra_string + '.ckpt', map_location=torch.device('cpu')))
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
        optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)

        max_iter = args.max_epoch * len(dset_loaders["source"])
        interval_iter = max_iter // args.max_epoch
        args.max_iter = max_iter
        iter_num = 0
        base_network.train()

        while iter_num < max_iter:
            try:
                inputs_source, labels_source = next(iter_source)
            except:
                iter_source = iter(dset_loaders["source"])
                inputs_source, labels_source = next(iter_source)

            if inputs_source.size(0) == 1:
                continue

            iter_num += 1

            features_source, outputs_source = base_network(inputs_source)

            classifier_loss = criterion(outputs_source, labels_source)

            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            classifier_loss.backward()
            optimizer_f.step()
            optimizer_c.step()

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                base_network.eval()

                if args.balanced:
                    acc_t_te, _ = cal_acc_comb(dset_loaders["Target"], base_network, args=args)
                    log_str = 'Task: {}, Iter:{}/{}; Offline-EA Acc = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                else:
                    acc_t_te, _ = cal_auc_comb(dset_loaders["Target-Imbalanced"], base_network, args=args)
                    log_str = 'Task: {}, Iter:{}/{}; Offline-EA AUC = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                args.log.record(log_str)
                print(log_str)

                base_network.train()

        print('saving model...')
        makedir_if_not_exist(os.path.join('./runs/', str(args.data_name)))
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(
                       args.idt) + '_seed' + str(args.SEED) + extra_string + '.ckpt')


    base_network.eval()

    score = cal_score_online(dset_loaders["Target-Online"], base_network, args=args)
    if args.balanced:
        log_str = 'Task: {}, Online IEA Acc = {:.2f}%'.format(args.task_str, score)
    else:
        log_str = 'Task: {}, Online IEA AUC = {:.2f}%'.format(args.task_str, score)
    args.log.record(log_str)
    print(log_str)

    print('executing TTA...')

    if args.balanced:
        acc_t_te, (y_pred, predict, y_true) = BN_adapt(dset_loaders["Target-Online"], base_network, args=args, balanced=True)
        log_str = 'Task: {}, TTA Acc = {:.2f}%'.format(args.task_str, acc_t_te)
    else:
        acc_t_te, (y_pred, predict, y_true) = BN_adapt(dset_loaders["Target-Online-Imbalanced"], base_network, args=args, balanced=False)
        log_str = 'Task: {}, TTA AUC = {:.2f}%'.format(args.task_str, acc_t_te)
    args.log.record(log_str)
    print(log_str)

    if args.balanced:
        print('Test Acc = {:.2f}%'.format(acc_t_te))

    else:
        print('Test AUC = {:.2f}%'.format(acc_t_te))

    torch.save(base_network.state_dict(), './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(
        args.SEED) + extra_string + '_adapted' + '.ckpt')

    # save the predictions for ensemble
    file_path = os.path.join(str(args.result_dir), str(args.data_name) + '_' + str(args.method) + '_seed_' + str(args.SEED) + "_pred.csv")
    # Check if the file exists
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
    else:
        df_existing = pd.DataFrame()
    # Convert torch.tensor to np.array and reshape
    predict = predict.numpy().reshape(-1, 1)  # Convert torch.tensor to np.array and reshape
    y_true = np.array(y_true).reshape(-1, 1)  # Convert list to np.array and reshape
    # Combine y_pred, predict, and y_true into a single array
    combined_data = np.hstack((y_pred, predict, y_true))
    # Generate column names
    class_columns = [f'Subject_{args.idt}_Class_{i}' for i in range(args.class_num)]
    additional_columns = [f'Subject_{args.idt}_predict', f'Subject_{args.idt}_true']
    header = class_columns + additional_columns
    # Create a new DataFrame with the combined data and the header
    df_new = pd.DataFrame(combined_data, columns=header)
    # Combine existing data with new columns
    df_combined = pd.concat([df_existing, df_new], axis=1)
    # Save the combined DataFrame to CSV
    df_combined.to_csv(file_path, index=False)

    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()

    return acc_t_te


if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='BNCI2014001', help='the data set name, now support BNCI2014001, BNCI2014002, BNCI2015001 from moabb')
    parser.add_argument('--data_save', type=str2bool, default=True, help='whether save the data to file')
    parser.add_argument('--data_path', type=str, default='./data/', help='the path to save the data from mobba dataset')
    parser.add_argument('--data_path_MI', type=str, default='/home/jyt/workspace/transfer_models/datasets_MI/hand_elbow/derivatives', help='the path to save the data from other datasets')
    parser.add_argument('--log_path', type=str, default='./logs/', help='the path to save the logs')
    parser.add_argument('--gpu_idx', type=int, default=0, help='index of GPU')
    parser.add_argument('--use_pretrained_model', type=str2bool, default=False, help='whether to use the pretrained model parameters')
    parser.add_argument('--finetune', type=str2bool, default=False, help='whether to finetune the model with part of the target data')
    parser.add_argument('--ft_volume', type=int, default=7*40, help='the amount of data for finetuning in target domain')
    parser.add_argument('--momentum', type=str2bool, default=False, help='whether to use the momentum updating for model parameters')
    parser.add_argument('--momentum_param', type=float, default=0.5, help='the value for momentum updating')

    args = parser.parse_args()

    data_name = args.dataset_name
    data_save = args.data_save
    data_path = args.data_path
    data_path_MI = args.data_path_MI
    log_path = args.log_path
    gpu_idx = args.gpu_idx
    use_pretrained_model = args.use_pretrained_model
    finetune = args.finetune
    ft_volume = args.ft_volume
    momentum = args.momentum
    momentum_param = args.momentum_param

    print('dataset_name: {}, type: {}'.format(data_name, type(data_name)))
    print('data_save: {}, type: {}'.format(data_save, type(data_save)))
    print('data_path_MI: {}, type: {}'.format(data_path_MI, type(data_path_MI)))
    print('data_path: {}, type: {}'.format(data_path, type(data_path)))
    print('log_path: {}, type: {}'.format(log_path, type(log_path)))
    print('gpu_idx: {}, type: {}'.format(gpu_idx, type(gpu_idx)))

    data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'BNCI2014001-4', 'MI-hand_elbow','MI-elbow_rest', 'MI-hand_rest']

    dct = pd.DataFrame(columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13'])

    if data_name in data_name_list:
        # N: number of subjects, chn: number of channels
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248
        if data_name == 'MI-hand_elbow': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
        if data_name == 'MI-elbow_rest': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
        if data_name == 'MI-hand_rest': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200

        # whether to use pretrained model
        # if source models have not been trained, set use_pretrained_model to False to train them
        # alternatively, run dnn.py to train source models, in seperating the steps
        if use_pretrained_model:
            # no training
            max_epoch = 0
        else:
            # training epochs
            max_epoch = 100

        # learning rate
        lr = 0.001

        # test batch size
        test_batch = 8

        # update step
        steps = 1

        # update stride
        stride = 1

        # whether to use EA
        align = True

        # temperature rescaling, for test entropy calculation
        t = 2

        # whether to test balanced or imbalanced (2:1) target subject
        balanced = True

        # whether to record running time
        calc_time = False

        # whether to use finetuning methods for some of the MI tasks and set how much data for finetuning
        if finetune:
            print('finetune: {}, ft_volume: {}'.format(finetune, ft_volume))

        # whether to use momentum updating method
        if momentum:
            print('momentum: {}, momentum_param: {}'.format(momentum, momentum_param))

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, align=align, lr=lr, t=t, max_epoch=max_epoch,
                                  trial_num=trial_num, time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, stride=stride, steps=steps, calc_time=calc_time,
                                  paradigm=paradigm, test_batch=test_batch, data_name=data_name, balanced=balanced,
                                  data_path_MI = data_path_MI,finetune=finetune,ft_volume=ft_volume,momentum=momentum,momentum_param=momentum_param)

        args.method = 'BN-adapt'
        args.backbone = 'EEGNet'

        # train batch size
        args.batch_size = 32

        # GPU device id
        try:
            device_id = gpu_idx
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
        except:
            args.data_env = 'local'
        total_acc = []

        # update multiple models, independently, from the source models
        for s in [2, 3, 4, 5]:
            args.SEED = s

            fix_random_seed(args.SEED)
            torch.backends.cudnn.deterministic = True

            args.data = data_name
            print(args.data)
            print(args.method)
            print(args.SEED)
            print(args)

            args.local_dir = data_path + str(data_name) + '/'
            args.result_dir = log_path
            my_log = LogRecord(args)
            my_log.log_init()
            my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

            sub_acc_all = np.zeros(N)
            for idt in range(N):
                args.idt = idt
                source_str = 'Except_S' + str(idt)
                target_str = 'S' + str(idt)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                print(info_str)
                my_log.record(info_str)
                args.log = my_log

                sub_acc_all[idt] = train_target(args)
            print('Sub acc: ', np.round(sub_acc_all, 3))
            print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
            total_acc.append(sub_acc_all)

            acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
            acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
            args.log.record("\n==========================================")
            args.log.record(acc_sub_str)
            args.log.record(acc_mean_str)

        args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)

        print(str(total_acc))

        args.log.record(str(total_acc))

        subject_mean = np.round(np.average(total_acc, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_acc)), 5)
        total_std = np.round(np.std(np.average(total_acc, axis=1)), 5)

        print(subject_mean)
        print(total_mean)
        print(total_std)

        args.log.record(str(subject_mean))
        args.log.record(str(total_mean))
        args.log.record(str(total_std))

        result_dct = {'dataset': data_name, 'avg': total_mean, 'std': total_std}
        for i in range(len(subject_mean)):
            result_dct['s' + str(i)] = subject_mean[i]

        dct = dct.append(result_dct, ignore_index=True)

    # save results to csv
    dct.to_csv(log_path + str(args.method) + ".csv")