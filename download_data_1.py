import os
import sys

import numpy as np
import argparse
from easydict import EasyDict as edict
# from tl.utils.utils import str2bool

import moabb
from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2015001, Lee2019_MI, BNCI2014004, Schirrmeister2017
from moabb.paradigms import MotorImagery, P300

from moabb.datasets import utils
from pooch import HTTPDownloader


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def dataset_to_file(dataset_name, data_save, data_path='./data/', proxy=None, timeout=60, retries=5):
    moabb.set_log_level("ERROR")
    
    # 1. 设置代理（如果提供）
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
    
    # 2. 初始化数据集和范式
    dataset, paradigm = None, None
    if dataset_name == 'BNCI2014001':
        dataset = BNCI2014001()
        paradigm = MotorImagery(n_classes=4)
        # (5184, 22, 1001) (5184,) 250Hz 9subjects * 4classes * (72+72)trials for 2sessions
    elif dataset_name == 'BNCI2014002':
        dataset = BNCI2014002()
        paradigm = MotorImagery(n_classes=2)
        # (2240, 15, 2561) (2240,) 512Hz 14subjects * 2classes * (50+30)trials * 2sessions(not namely separately)
    elif dataset_name == 'BNCI2015001':
        dataset = BNCI2015001()
        paradigm = MotorImagery(n_classes=2)
        # (5600, 13, 2561) (5600,) 512Hz 12subjects * 2 classes * (200 + 200 + (200 for Subj 8/9/10/11)) trials * (2/3)sessions
    elif dataset_name == "Lee2019_MI":
        dataset = Lee2019_MI()
        paradigm = MotorImagery(n_classes=2)
    elif dataset_name == "BNCI2014_004":
        # (6520, 3, 1126)  (6520, ) 250Hz 9subjects * 2 classes * (appro. 400trials offline and 320 trials online per subject)
        dataset = BNCI2014004()
        paradigm = MotorImagery(n_classes=2)
    elif dataset_name == "Schirrmeister2017":
        # choosing 44 scenors from the 128 channels, donwsampled from 500Hz to 250Hz
        # 44 scenors referring to https://github.com/robintibor/high-gamma-dataset/blob/master/example.py
        # (13484, 44, 1000) (13484, ) 500Hz, 14 subjects * 4classes * ()
        C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
        dataset = Schirrmeister2017()
        paradigm = MotorImagery(n_classes=4, channels=C_sensors, resample=250)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    if data_save:
        print('preparing ' + str(dataset_name) + ' data...')
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:])
        ar_unique, cnts = np.unique(labels, return_counts=True)
        print("labels:", ar_unique)
        print("Counts:", cnts)
        print(X.shape, labels.shape)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if not os.path.exists(data_path + dataset_name + '/'):
            os.makedirs(data_path + dataset_name + '/')
        np.save(data_path + dataset_name + '/X', X)
        np.save(data_path + dataset_name + '/labels', labels)
        meta.to_csv(data_path + dataset_name + '/meta.csv')
        print('done!')
    else:
        if isinstance(paradigm, MotorImagery):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]], return_epochs=True)
            return X.info
        elif isinstance(paradigm, P300):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]], return_epochs=True)
            return X.info


if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='BNCI2014001', help='the data set name, now support BNCI2014001, BNCI2014002, BNCI2015001 from moabb')
    parser.add_argument('--data_save', type=str2bool, default=True, help='whether save the data to file')
    parser.add_argument('--data_path', type=str, default='./data/', help='the path to save the data')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    data_save = args.data_save
    data_path = args.data_path

    print('dataset_name: {}, type: {}'.format(dataset_name, type(dataset_name)))
    print('data_save: {}, type: {}'.format(data_save, type(data_save)))
    print('data_path: {}, type: {}'.format(data_path, type(data_path)))

    # load the dataset
    if dataset_name in ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'Lee2019_MI', 'Liu2024', 'BNCI2014_004', 'Schirrmeister2017']:
        info = dataset_to_file(dataset_name, data_save=data_save, data_path=data_path)

    '''
    BNCI2014001
    <Info | 8 non-empty values
     bads: []
     ch_names: 'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
     chs: 22 EEG
     custom_ref_applied: False
     dig: 25 items (3 Cardinal, 22 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 22
     projs: []
     sfreq: 250.0 Hz
    >

    BNCI2014002
    <Info | 7 non-empty values
     bads: []
     ch_names: 'EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8', 'EEG9', 'EEG10', 'EEG11', 'EEG12', 'EEG13', 'EEG14', 'EEG15'
     chs: 15 EEG
     custom_ref_applied: False
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 15
     projs: []
     sfreq: 512.0 Hz
    >

    BNCI2015001
    <Info | 8 non-empty values
     bads: []
     ch_names: 'FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4'
     chs: 13 EEG
     custom_ref_applied: False
     dig: 16 items (3 Cardinal, 13 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 13
     projs: []
     sfreq: 512.0 Hz
    >
    '''