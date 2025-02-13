import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tl.utils.utils import str2bool
from sklearn.metrics import confusion_matrix

def Test_time_visualizationClass(data_path, class_num, trial_num, current_dir):
    # load the data
    data = pd.read_csv(data_path, header=0)
    
    # calculate the num of subjects
    n_subjects = len(data.columns) // 4  
    all_recalls = {cls: [] for cls in range(class_num)}  # for restoring recall values
    
    for subject_id in range(n_subjects):
        # obtain the prediction and true labels
        pred_col = subject_id * 4 + 2
        true_col = subject_id * 4 + 3
        subject_data = data.iloc[:, [pred_col, true_col]].dropna()
        subject_data.columns = ['pred', 'true']
        
        # calculate the recall value in each segment
        num_segments = len(subject_data) // trial_num
        for seg in range(num_segments):
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

    return stats

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
    
    # for visulization
    parser.add_argument('--visualfile_csv', type=str, default="MI-elbow_rest_T-TIME_seed_1_pred.csv", help='the name of .csv file for visualization')
    parser.add_argument('--visualfile_trial', type=int, default=40, help='the num of trials in each segment for visualization')
    parser.add_argument('--visual_acc', type=str2bool, default=False, help='whether to show the acc with segments in visualization')
    
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
    
    visualfile_csv = args.visualfile_csv
    visualfile_trial = args.visualfile_trial
    visual_acc = args.visual_acc

    if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
    if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
    if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
    if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248
    if data_name == 'MI-hand_elbow': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
    if data_name == 'MI-elbow_rest': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
    if data_name == 'MI-hand_rest': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200

    stats = Test_time_visualizationClass(os.path.join(log_path, visualfile_csv), class_num=class_num, trial_num=visualfile_trial, current_dir=log_path)


