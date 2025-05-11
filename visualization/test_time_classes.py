import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tl.utils.utils import str2bool
from sklearn.metrics import confusion_matrix

def Test_time_visualizationClass(data_path, class_num, trial_num, current_dir, data_name, imgdata_save=False):
    # load the data
    data = pd.read_csv(data_path, header=0)
    
    # calculate the num of subjects
    n_subjects = len(data.columns) // (class_num+2)  
    all_recalls = {cls: [] for cls in range(class_num)}  # for restoring recall values
    
    for subject_id in range(n_subjects):
        # obtain the prediction and true labels
        pred_col = subject_id * (class_num+2) + (class_num+2)-2
        true_col = subject_id * (class_num+2) + (class_num+2)-1
        subject_data = data.iloc[:, [pred_col, true_col]].dropna()
        subject_data.columns = ['pred', 'true']
        
        # calculate the recall value in each segment
        num_segments = len(subject_data) // trial_num
        if subject_id == 9 and data_name == "WBCIC-SHU-3C":
            num_segments = num_segments+1
        for seg in range(num_segments):
            if subject_id == 9 and data_name == "WBCIC-SHU-3C" and seg==0:
                seg_data = subject_data.iloc[0 : 299]
            elif subject_id == 9 and data_name == "WBCIC-SHU-3C" and seg==1:
                seg_data = subject_data.iloc[299 : 599]
            elif subject_id == 9 and data_name == "WBCIC-SHU-3C" and seg==2:
                seg_data = subject_data.iloc[599 : ]
            else:
                seg_data = subject_data.iloc[seg*trial_num : (seg+1)*trial_num]
            
            pred = seg_data['pred'].astype(int)
            true = seg_data['true'].astype(int)

            # calculate the confusing matrix
            cm = confusion_matrix(true, pred, labels=range(class_num))
            
            # calcuate recall for each class
            for cls in range(class_num):
                tp = cm[cls, cls]  # true positive
                fn = cm[cls, :].sum() - tp  # false negative
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                all_recalls[cls].append(recall)
    
    # obtain the results for each subject
    stats = {
        cls: {
            'mean': [],
            'std': [],
            'segments': len(all_recalls[cls])//n_subjects  # segments for each subject
        } for cls in range(class_num)
    }
    
    # calculate the mean and std for each class
    for cls in range(class_num):
        recalls = np.array(all_recalls[cls]).reshape(n_subjects, -1)
        stats[cls]['mean'] = recalls.mean(axis=0)
        stats[cls]['std'] = recalls.std(axis=0)

    if imgdata_save:
        # Prepare to save the plot and statistics to the specified directory
        # plot the results
        plt.figure(figsize=(10, 6))
        x = np.arange(stats[0]['segments'])
        
        for cls in range(class_num):
            plt.plot(x, stats[cls]['mean'], 
                    label=f'Class {cls}', 
                    marker='o')
            plt.fill_between(x,
                            stats[cls]['mean'] - stats[cls]['std'],
                            stats[cls]['mean'] + stats[cls]['std'],
                            alpha=0.2)
        
        plt.xlabel('Sample Segment Start Index')
        plt.ylabel('Recall')
        plt.title(f'Recall Change Over Segments (Window Size={trial_num})')
        plt.legend()
        plt.grid(True)

        output_filename = os.path.join(current_dir, "MI_recall_segments.png")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')  # save as .png
        print(f"{output_filename} saved")

        # Export statistics to CSV
        stats_df = pd.DataFrame()
        for cls in range(class_num):
            stats_df[f'Class_{cls}_mean'] = stats[cls]['mean']
            stats_df[f'Class_{cls}_std'] = stats[cls]['std']
        
        csv_path = os.path.join(current_dir, "MI_recall_stats.csv")
        stats_df.to_csv(csv_path, index=False)
        print(f"Statistics saved to {csv_path}")

    return stats

def Test_time_visualizationClass_seeds(class_num, trial_num, current_dir, data_name, args):
    # find the .csv files of different seeds
    _pattern = re.compile(r'_seed_\d+_pred\.csv$')
    csv_files = []
    for _file_name in os.listdir(args.log_path):
        if _file_name.endswith('.csv') and _pattern.search(_file_name):
            full_path = os.path.join(args.log_path, _file_name)
            csv_files.append(full_path)

    stats_ensamble = {
        cls: {
            'mean': [],
            'mean_ensamble': [],
            'std_ensamble': [],
        } for cls in range(class_num)
    }
    # Process each CSV file found
    for data_path in csv_files:
        stats = Test_time_visualizationClass(data_path, class_num, trial_num, current_dir, data_name, imgdata_save=args.data_save)
        for _cls in range(class_num):
            stats_ensamble[_cls]['mean'].append(stats[_cls]['mean'])  # Append mean recall for each class from current seed
    
    for _cls in range(class_num):
        # Calculate mean recall across all seeds for each class
        stats_ensamble[_cls]['mean'] = np.vstack(stats_ensamble[_cls]['mean'])
        stats_ensamble[_cls]['mean_ensamble'] = np.mean(stats_ensamble[_cls]['mean'], axis=0)  # Calculate mean recall across all seeds for each class
        stats_ensamble[_cls]['std_ensamble'] = np.std(stats_ensamble[_cls]['mean'], axis=0)
    
    # Prepare to save the plot and statistics to the specified directory
    # plot the results
    plt.figure(figsize=(10, 6))
    x = np.arange(stats_ensamble[0]['mean_ensamble'].shape[0])
    
    for cls in range(class_num):
        plt.plot(x, stats_ensamble[cls]['mean_ensamble'], 
                label=f'Class {cls}', 
                marker='o')
        plt.fill_between(x,
                        stats_ensamble[cls]['mean_ensamble'] - stats_ensamble[cls]['std_ensamble'],
                        stats_ensamble[cls]['mean_ensamble'] + stats_ensamble[cls]['std_ensamble'],
                        alpha=0.2)
    
    plt.xlabel('Sample Segment Start Index')
    plt.ylabel('Recall')
    plt.title(f'Recall Change Over Segments (Window Size={trial_num})')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.4, 1.0)
    #plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    output_filename = os.path.join(current_dir, "MI_recall_segments_seeds.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')  # save as .png
    print(f"{output_filename} saved")

    # Export statistics to CSV
    stats_df = pd.DataFrame()
    for cls in range(class_num):
        stats_df[f'Class_{cls}_mean'] = stats_ensamble[cls]['mean_ensamble']
    
    csv_path = os.path.join(current_dir, "MI_recall_stats_seeds.csv")
    stats_df.to_csv(csv_path, index=False)
    print(f"Statistics saved to {csv_path}")



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
    parser.add_argument('--align', type=str2bool, default=True, help='use EA alignment and IEA alignment')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in offline training')
    parser.add_argument('--batch_size_online', type=int, default=8, help='batch size in online adaptation')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate in offline and online training')
    parser.add_argument('--lr_online', type=float, default=0.001, help='learning rate in online adaptation')
    parser.add_argument('--epoch', type=int, default=100, help='epoches in offline and online training')
    parser.add_argument('--backbone', type=str, default='EEGNet-4,2', help='backbone of the model')
    parser.add_argument('--param_runs', type=str, default='./runs/', help='folder for saving the run paramters')

    # for visulization
    parser.add_argument('--visualfile_csv', type=str, default="MI-elbow_rest_T-TIME_seed_1_pred.csv", help='the name of .csv file for visualization')
    parser.add_argument('--visualfile_trial', type=int, default=40, help='the num of trials in each segment for visualization')
    parser.add_argument('--visual_acc', type=str2bool, default=False, help='whether to show the acc with segments in visualization')
    parser.add_argument('--visual_ensamble', type=str2bool, default=False, help='whether to ensamble all the results from differernt seeds')
    # parser.add_argument('--tta_method', type=str, default=None, help='the method for visualization')

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
    align = args.align
    batch_size = args.batch_size
    batch_size_online = args.batch_size_online
    lr = args.lr
    epoch = args.epoch
    backbone = args.backbone
    param_runs = args.param_runs
    lr_online = args.lr_online
    
    visualfile_csv = args.visualfile_csv
    visualfile_trial = args.visualfile_trial
    visual_acc = args.visual_acc
    # tta_method = args.tta_method
    visual_ensamble = args.visual_ensamble

    if backbone == 'EEGNet':
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 576, 496
        if data_name == 'BNCI2014001-4-train': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 496
        if data_name == 'MI-hand_elbow': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
        if data_name == 'MI-elbow_rest': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
        if data_name == 'MI-hand_rest': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
        if data_name == 'BNCI2014001-4-all': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 576, 496
        if data_name == 'BNCI2014001-4-test': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 496
        if data_name == 'BNCI2014_004-train': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 3, 2, 1126, 250, 400, 560
        if data_name == 'BNCI2014_004-test': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 3, 2, 1126, 250, 400, 560
        if data_name == 'WBCIC-SHU-3C': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 11, 58, 3, 1000, 250, 900, 496
    if backbone == 'EEGNet-4,2':
        if data_name == 'BNCI2014001-4-train': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248
        if data_name == 'MI-hand_elbow': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
        if data_name == 'MI-elbow_rest': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
        if data_name == 'MI-hand_rest': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
        if data_name == 'BNCI2014001-4-all': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 576, 248
        if data_name == 'BNCI2014001-4-test': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248
        if data_name == 'BNCI2014_004-train': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 3, 2, 1126, 250, 400, 280
        if data_name == 'BNCI2014_004-test': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 3, 2, 1126, 250, 400, 280
        if data_name == 'WBCIC-SHU-3C': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 11, 58, 3, 1000, 250, 900, 248
    
    if not visual_ensamble:
        stats = Test_time_visualizationClass(os.path.join(log_path, visualfile_csv), class_num=class_num, trial_num=visualfile_trial, current_dir=log_path, data_name=data_name)
    else:
        Test_time_visualizationClass_seeds(class_num=class_num, trial_num=visualfile_trial, current_dir=log_path, data_name=data_name, args=args)


