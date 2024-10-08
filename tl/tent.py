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
from utils.utils import fix_random_seed, cal_acc_comb, data_loader, cal_auc_comb, cal_score_online
from utils.alg_utils import EA, EA_online
from scipy.linalg import fractional_matrix_power
from models.tent import configure_model, collect_params, Tent
from sklearn.metrics import roc_auc_score, accuracy_score

import gc
import sys
import time

# This is the implementation of Tent from paper
# Wang D, Shelhamer E, Liu S, et al. Tent: Fully test-time adaptation by entropy minimization[J]. arXiv preprint arXiv:2006.10726, 2020.
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : tent.py
# from github https://github.com/sylyoung/DeepTransferEEG/tree/main

def Tent_func(loader, model, args, balanced=True):
    # Tent

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

        if (i + 1) >= args.test_batch:
            if args.stride != 1:
                print('must have stride 1')
                sys.exit(1)
            else:
                if (i + 1) == args.test_batch:
                    # Tent mode initialize
                    model = configure_model(model)
                    params, param_names = collect_params(model)
                    optimizer = torch.optim.Adam(params, lr=args.lr)
                    tented_model = Tent(model, optimizer)

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

                outputs = tented_model(batch_test)[-1].reshape(1, -1)
        else:
            _, outputs = model(sample_test)

        softmax_out = nn.Softmax(dim=1)(outputs)

        outputs = outputs.float().cpu()
        labels = labels.float().cpu()
        _, predict = torch.max(outputs, 1)

        y_pred.append(softmax_out.detach().cpu().numpy())
        y_true.append(labels.item())

    if balanced:
        _, predict = torch.max(torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.class_num), 1)
        pred = torch.squeeze(predict).float()
        score = accuracy_score(y_true, pred)
        if args.data_name == 'BNCI2014001-4':
            y_pred = np.array(y_pred).reshape(-1, )  # multiclass
        else:
            y_pred = np.array(y_pred).reshape(-1, args.class_num)[:, 1]  # binary
    else:
        predict = torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.class_num)
        y_pred = np.array(predict).reshape(-1, args.class_num)[:, 1]  # binary
        score = roc_auc_score(y_true, y_pred)
    return score * 100, y_pred


def train_target(args):
    if not args.align:
        extra_string = '_noEA'
    else:
        extra_string = ''
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

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
        acc_t_te, y_pred = Tent_func(dset_loaders["Target-Online"], base_network, args=args, balanced=True)
        log_str = 'Task: {}, TTA Acc = {:.2f}%'.format(args.task_str, acc_t_te)
    else:
        acc_t_te, y_pred = Tent_func(dset_loaders["Target-Online-Imbalanced"], base_network, args=args, balanced=False)
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
    with open('./logs/' + str(args.data_name) + '_' + str(args.method) + '_seed_' + str(args.SEED) +"_pred.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(y_pred)

    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()

    return acc_t_te


if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='BNCI2014001', help='the data set name, now support BNCI2014001, BNCI2014002, BNCI2015001 from moabb')
    parser.add_argument('--data_save', type=str2bool, default=True, help='whether save the data to file')
    parser.add_argument('--data_path', type=str, default='./data/', help='the path to save the data')
    parser.add_argument('--log_path', type=str, default='./logs/', help='the path to save the logs')
    parser.add_argument('--gpu_idx', type=int, default=0, help='index of GPU')
    args = parser.parse_args()

    data_name = args.dataset_name
    data_save = args.data_save
    data_path = args.data_path
    log_path = args.log_path
    gpu_idx = args.gpu_idx

    print('dataset_name: {}, type: {}'.format(data_name, type(data_name)))
    print('data_save: {}, type: {}'.format(data_save, type(data_save)))
    print('data_path: {}, type: {}'.format(data_path, type(data_path)))
    print('log_path: {}, type: {}'.format(log_path, type(log_path)))
    print('gpu_idx: {}, type: {}'.format(gpu_idx, type(gpu_idx)))

    data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'BNCI2014001-4']

    dct = pd.DataFrame(columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13'])

    if data_name in data_name_list:
        # N: number of subjects, chn: number of channels
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248

        # whether to use pretrained model
        # if source models have not been trained, set use_pretrained_model to False to train them
        # alternatively, run dnn.py to train source models, in seperating the steps
        use_pretrained_model = True
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

        # whether to test balanced or imbalanced (2:1) target subject
        balanced = True

        # whether to record running time
        calc_time = False

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, align=align, lr=lr, max_epoch=max_epoch,
                                  trial_num=trial_num, time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, stride=stride, steps=steps, calc_time=calc_time,
                                  paradigm=paradigm, test_batch=test_batch, data_name=data_name, balanced=balanced)

        args.method = 'Tent'
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
        for s in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
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