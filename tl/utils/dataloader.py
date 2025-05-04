# -*- coding: utf-8 -*-
# @Time    : 2023/7/11
# @Author  : Siyang Li
# @File    : dataloader.py
import numpy as np
from sklearn import preprocessing
import os
import scipy.io as sio
from utils.data_utils import traintest_split_cross_subject, traintest_split_cross_subject_meta, traintest_split_domain_classifier, traintest_split_multisource, traintest_split_domain_classifier_pretest, traintest_split_multisource
import pandas as pd

def data_process(args):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''
    dataset = args.data

    if dataset in ['BNCI2014001-4', 'BNCI2014001-4-all', 'BNCI2014001-4-train', 'BNCI2014001-4-test']:
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
        meta = pd.read_csv('./data/' + 'BNCI2014001' + '/meta.csv')
        print(X.shape, y.shape)
    elif dataset in ['BNCI2014_004-train', 'BNCI2014_004-test']:
        X = np.load('./data/' + 'BNCI2014_004' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014_004' + '/labels.npy')
        meta = pd.read_csv('./data/' + 'BNCI2014_004' + '/meta.csv')
        print(X.shape, y.shape)
    elif dataset not in ['MI-hand_elbow', 'MI-elbow_rest', 'MI-hand_rest', 'WBCIC-SHU-3C']:  # other datasets that use the moabb
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
        meta = pd.read_csv('./data/' + dataset + '/meta.csv')
        print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(100) + (160 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))

        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22
        
        # using all the sessions
        # X = X
        # y = y
        """
        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
        """
        
    elif dataset in ['BNCI2014001-4-all', 'BNCI2014001-4-train', 'BNCI2014001-4-test']:
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # using all the sessions
        # X = X
        # y = y
    elif dataset in ['BNCI2014_004', 'BNCI2014_004-train', 'BNCI2014_004-test']:
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 3

        # using all the sessions
        # X = X
        # y = y
    elif dataset == 'Schirrmeister2017':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 250
        ch_num = 44

        # using all the sessions
        # X = X
        # y = y
    elif dataset == 'MI-hand_elbow':
        # hand_elbow dataset is MI of movements of hand and elbow on the same side of the limb
        # dataset paper: 
        # Ma X, Qiu S, He H. Multi-channel EEG recording during motor imagery of different joints from the same limb[J]. Scientific data, 2020, 7(1): 191.
        # three classes: rest, hand and elbow, in this setting, we only use the hand and elbow classes.
        paradigm = 'MI'
        num_subjects = 25
        sample_rate = 200
        ch_num = 62
        X = None
        y = None

        folder_path = args.data_path_MI
        for num in range(num_subjects):
            sub_file = f'{(num+1):03}'
            sub_mat = sio.loadmat(os.path.join(folder_path, 'sub-' + sub_file, 'eeg', 'sub-' + sub_file + '_task-motorimagery_eeg.mat'))
            # each subject's MI data (hand and elbow) is 15*40 trials, each trial contains 4s data with 62 channels and 250 sampling rate
            sub_task_data = sub_mat['task_data'].reshape(-1, 62, 800)
            sub_task_label = sub_mat['task_label'].reshape(-1, 1)

            # concatenate all subjects' data
            if X is None:
                X = sub_task_data
                y = sub_task_label
            else:
                X = np.concatenate((X, sub_task_data), axis=0)
                y = np.concatenate((y, sub_task_label), axis=0)
    
    elif dataset == 'MI-hand_rest':
        # hand_elbow dataset is MI of movements of hand and elbow on the same side of the limb
        # dataset paper: 
        # Ma X, Qiu S, He H. Multi-channel EEG recording during motor imagery of different joints from the same limb[J]. Scientific data, 2020, 7(1): 191.
        # three classes: rest, hand and elbow, in this setting, we only use the hand and rest classes.
        paradigm = 'MI'
        num_subjects = 25
        sample_rate = 200
        ch_num = 62
        X = None
        y = None

        folder_path = args.data_path_MI
        for num in range(num_subjects):
            sub_file = f'{(num+1):03}'
            sub_mat = sio.loadmat(os.path.join(folder_path, 'sub-' + sub_file, 'eeg', 'sub-' + sub_file + '_task-motorimagery_eeg.mat'))
            # each subject's MI data (hand and elbow) is 15*40 trials, each trial contains 4s data with 62 channels and 250 sampling rate
            sub_task_data = sub_mat['task_data'].reshape(-1, 62, 800)
            sub_task_label = sub_mat['task_label'].reshape(-1, 1)
            sub_rest_data = sub_mat['rest_data'].reshape(-1, 62, 800)

            # in this setting, we want to use the data of hand and rest classes, we use data of rest to take the place of data of elbow
            # Find indices where the label is 2 (elbow)
            label_indices = np.where(sub_task_label == 2)[0]
            # Ensure the number of samples to be replaced matches the number of available replacement samples
            assert len(label_indices) <= sub_rest_data.shape[0], "The number of samples with label 2 is greater than the available replacement samples"
            # Replace samples
            for idx, label_idx in enumerate(label_indices):
                sub_task_data[label_idx] = sub_rest_data[idx]

            
            """
            # for debug
            test_id = 514
            _sub_test_data = sub_task_data[int(test_id)]
            _sub_test_labe = sub_task_label[int(test_id)]
            _rest_test_idx = np.where(label_indices==int(test_id))
            _rest_test_data = sub_rest_data[_rest_test_idx]
            """
            
            # concatenate all subjects' data
            if X is None:
                X = sub_task_data
                y = sub_task_label
            else:
                X = np.concatenate((X, sub_task_data), axis=0)
                y = np.concatenate((y, sub_task_label), axis=0)
    
    elif dataset == 'MI-elbow_rest':
        # hand_elbow dataset is MI of movements of hand and elbow on the same side of the limb
        # dataset paper: 
        # Ma X, Qiu S, He H. Multi-channel EEG recording during motor imagery of different joints from the same limb[J]. Scientific data, 2020, 7(1): 191.
        # three classes: rest, hand and elbow, in this setting, we only use the hand and rest classes.
        paradigm = 'MI'
        num_subjects = 25
        sample_rate = 200
        ch_num = 62
        X = None
        y = None

        folder_path = args.data_path_MI
        for num in range(num_subjects):
            sub_file = f'{(num+1):03}'
            sub_mat = sio.loadmat(os.path.join(folder_path, 'sub-' + sub_file, 'eeg', 'sub-' + sub_file + '_task-motorimagery_eeg.mat'))
            # each subject's MI data (hand and elbow) is 15*40 trials, each trial contains 4s data with 62 channels and 250 sampling rate
            sub_task_data = sub_mat['task_data'].reshape(-1, 62, 800)
            sub_task_label = sub_mat['task_label'].reshape(-1, 1)
            sub_rest_data = sub_mat['rest_data'].reshape(-1, 62, 800)

            # in this setting, we want to use the data of elbow and rest classes, we use data of rest to take the place of data of hand
            # Find indices where the label is 1 (hand)
            label_indices = np.where(sub_task_label == 1)[0]
            # Ensure the number of samples to be replaced matches the number of available replacement samples
            assert len(label_indices) <= sub_rest_data.shape[0], "The number of samples with label 2 is greater than the available replacement samples"
            # Replace samples
            for idx, label_idx in enumerate(label_indices):
                sub_task_data[label_idx] = sub_rest_data[idx]

            """
            # for debug
            test_id = 114
            _sub_test_data = sub_task_data[int(test_id)]
            _sub_test_label = sub_task_label[int(test_id)]
            _rest_test_idx = np.where(label_indices==int(test_id))
            _rest_test_data = sub_rest_data[_rest_test_idx]
            """

            # concatenate all subjects' data
            if X is None:
                X = sub_task_data
                y = sub_task_label
            else:
                X = np.concatenate((X, sub_task_data), axis=0)
                y = np.concatenate((y, sub_task_label), axis=0)
    
    elif dataset == 'WBCIC-SHU-3C':
        # the 3 class multi-day dataset contains MI of left-right hand and foot
        # dataset paper: 
        # Yang B, Rong F, Xie Y, et al. A multi-day and high-quality EEG dataset for motor imagery brain-computer interface[J]. Scientific Data, 2025, 12(1): 488.
        # three classes: left hand, right hand and foot
        paradigm = 'MI'
        num_subjects = 11
        sample_rate = 250
        ch_num = 58
        sessions = 3
        X = None
        y = None
        meta = None  # we do not use the meta information in this dataset 
        _subject_trialnum = []

        folder_path = args.data_path_MI
        for num in range(num_subjects):
            sub_task_data = None
            sub_task_label = None

            sub_file = f'{(num+1):03}'
            for session_idx in range(sessions):
                _sub_path = os.path.join(folder_path, 'Sub-' + sub_file, 'dataset2_processeddata_' + 'Sub-' + sub_file + '_sess-0' + str(session_idx+1) + '_task-MI_eeg.mat')
                sub_mat = sio.loadmat(_sub_path) # example: dataset2_processeddata_Sub-008_sess-01_task-MI_eeg.mat
                # each subject's MI data contains 3*300 trials (3 sessions on different days, and each session contains 300 trials with 100 trials for each MI class), 
                # each trial contains 4s data with 58 channels and 250 sampling rate
                _task_data = np.transpose(sub_mat['data'], (2,0,1))  # the original data [channels × time × trials], transpose to [trials × channels × time]
                _task_label = sub_mat['labels'].reshape(-1, 1)
                # save the data for each subject
                if sub_task_data is None:
                    sub_task_data = _task_data
                    sub_task_label = _task_label
                else:
                    sub_task_data = np.concatenate((sub_task_data, _task_data), axis=0)
                    sub_task_label = np.concatenate((sub_task_label, _task_label), axis=0)

            _subject_trialnum.append(sub_task_data.shape[0])

            # concatenate all subjects' data
            if X is None:
                X = sub_task_data
                y = sub_task_label
            else:
                X = np.concatenate((X, sub_task_data), axis=0)  
                y = np.concatenate((y, sub_task_label), axis=0)
        
        print(_subject_trialnum)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    
    return X, y, num_subjects, paradigm, sample_rate, ch_num, meta


def data_process_secondsession(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if dataset == 'BNCI2014001-4':
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i) + 288) # use second sessions
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            #indices.append(np.arange(100) + (160 * i))
            indices.append(np.arange(60) + (160 * i) + 100) # use second sessions
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            # use second sessions
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))

        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def read_mi_combine_tar(args):
    if 'ontinual' in args.method:  # TODO
        # Continual TTA
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_process_secondsession(args.data)
    else:
        X, y, num_subjects, paradigm, sample_rate, ch_num, meta = data_process(args)
    
    if args.data in ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'BNCI2014001-4']:
        src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(args.data, X, y, num_subjects, args.idt)
    elif args.data in ['BNCI2014001-4-all', 'BNCI2014_004-train', 'BNCI2014001-4-test', 'Schirrmeister2017','BNCI2014001-4-train','BNCI2014_004-test', 'WBCIC-SHU-3C']:
        src_data, src_label, tar_data, tar_label = traintest_split_cross_subject_meta(args.data, X, y, num_subjects, args.idt, meta)

    return src_data, src_label, tar_data, tar_label


def read_mi_combine_domain(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num, meta = data_process(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_domain_classifier(args.data, X, y, num_subjects, args.idt)

    return src_data, src_label, tar_data, tar_label


def read_mi_combine_domain_split(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num, meta = data_process(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_domain_classifier_pretest(args.data, X, y, num_subjects, args.ratio)

    return src_data, src_label, tar_data, tar_label


def read_mi_multi_source(args):
    X, y, num_subjects, paradigm, sample_rate, ch_num, meta = data_process(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_multisource(args.data, X, y, num_subjects, args.idt)

    return src_data, src_label, tar_data, tar_label


def data_normalize(fea_de, norm_type):
    if norm_type == 'zscore':
        zscore = preprocessing.StandardScaler()
        fea_de = zscore.fit_transform(fea_de)

    return fea_de
