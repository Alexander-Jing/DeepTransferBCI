import os
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import accuracy_score, f1_score

def compute_subject_metrics(result_dir, data_name, method, seeds, class_num, subject_ids):
    subject_acc = {sid: [] for sid in subject_ids}
    subject_f1 = {sid: [] for sid in subject_ids}

    for seed in seeds:
        file_path = os.path.join(result_dir, f'{data_name}_{method}_seed_{seed}_pred.csv')
        df = pd.read_csv(file_path)
        for sid in subject_ids:
            class_cols = [f'Subject_{sid}_Class_{i}' for i in range(class_num)]
            true_col = f'Subject_{sid}_true'
            # Check if columns exist
            if all(col in df.columns for col in class_cols + [true_col]):
                # Drop rows with NaN in subject columns
                subject_df = df[class_cols + [true_col]].dropna()
                if not subject_df.empty:
                    logits = subject_df[class_cols].values
                    y_pred = np.argmax(logits, axis=1)
                    y_true = subject_df[true_col].values.flatten()
                    acc = accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred, average='macro')
                    subject_acc[sid].append(acc)
                    subject_f1[sid].append(f1)

    mean_acc = [np.mean(subject_acc[sid]) for sid in subject_ids]
    mean_f1 = [np.mean(subject_f1[sid]) for sid in subject_ids]
    overall_acc = np.mean(mean_acc)
    overall_f1 = np.mean(mean_f1)

    return mean_acc, mean_f1, overall_acc, overall_f1


def compute_subject_metrics_with_predict(result_dir, data_name, method, seeds, class_num, subject_ids):
    subject_acc = {sid: [] for sid in subject_ids}
    subject_f1 = {sid: [] for sid in subject_ids}

    for seed in seeds:
        file_path = os.path.join(result_dir, f'{data_name}_{method}_seed_{seed}_pred.csv')
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path)
        for sid in subject_ids:
            pred_col = f'Subject_{sid}_predict'
            true_col = f'Subject_{sid}_true'
            if all(col in df.columns for col in [pred_col, true_col]):
                subject_df = df[[pred_col, true_col]].dropna()
                if not subject_df.empty:
                    y_pred = subject_df[pred_col].values.flatten()
                    y_true = subject_df[true_col].values.flatten()
                    acc = accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred, average='macro')
                    subject_acc[sid].append(acc)
                    subject_f1[sid].append(f1)

    mean_acc = [np.mean(subject_acc[sid]) if subject_acc[sid] else np.nan for sid in subject_ids]
    mean_f1 = [np.mean(subject_f1[sid]) if subject_f1[sid] else np.nan for sid in subject_ids]
    overall_acc = np.nanmean(mean_acc)
    overall_f1 = np.nanmean(mean_f1)

    return mean_acc, mean_f1, overall_acc, overall_f1


def compute_seed_subject_metrics_forShot(
    predicts_dirs, true_labels_dirs, data_name, method, seeds, subject_ids, class_num):
    subject_acc = {sid: [] for sid in subject_ids}
    subject_f1 = {sid: [] for sid in subject_ids}

    for seed in seeds:
        predicts_path = os.path.join(
            str(predicts_dirs),
            f"{data_name}_{method}_seed_{seed}_pred.csv"
        )
        true_labels_path = os.path.join(
            str(true_labels_dirs),
            f"{data_name}_source_seed_{seed}_pred.csv"
        )

        # Read predicts
        predicts = []
        if os.path.exists(predicts_path):
            with open(predicts_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    predicts_sub = []
                    # row 是一个长度为1的列表，内容是整个字符串
                    for val in row:
                        softmax_str = val.strip().replace('[', '').replace(']', '')
                        softmax_values = [float(_val) for _val in softmax_str.split()]
                        predicts_sub.append(softmax_values)
                    predicts.append(predicts_sub)
        else:
            continue

        # Read true labels
        df = pd.read_csv(true_labels_path)
        for sid in subject_ids:
            true_col = f'Subject_{sid}_true'
            if true_col in df.columns:
                y_true = df[true_col].dropna().values.flatten()
                n = len(y_true)
                y_pred = np.argmax(np.array(predicts[sid]), axis=1)
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='macro')
                subject_acc[sid].append(acc)
                subject_f1[sid].append(f1)

    mean_acc = [np.mean(subject_acc[sid]) if subject_acc[sid] else np.nan for sid in subject_ids]
    mean_f1 = [np.mean(subject_f1[sid]) if subject_f1[sid] else np.nan for sid in subject_ids]
    overall_acc = np.nanmean(mean_acc)
    overall_f1 = np.nanmean(mean_f1)

    return mean_acc, mean_f1, overall_acc, overall_f1



if __name__ == "__main__":
    
    # data_name = "WBCIC-SHU-3C"
    true_labels_dirs = "./logs/Baselines-WBCIC-SHU-3C-e300-b64/source-WBCIC-SHU-3C-EEGNet-4,2-e300-b64/"
    data_name = "WBCIC-SHU-3C"
    
    class_num = 3
    subject_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    predicts_dirs = "./logs/Baselines-WBCIC-SHU-3C-e300-b64/shot-im-WBCIC-SHU-3C-EEGNet-4,2-e300-b64/"
    method = "SHOT-IM"

    mean_acc, mean_f1, overall_acc, overall_f1 = compute_seed_subject_metrics_forShot(
    predicts_dirs, true_labels_dirs, data_name, method, [1,2,3,4,5], subject_ids, class_num
    )
    
    print(f"Overall Mean Accuracy: {overall_acc:.4f}")
    print(f"Overall Mean F1 Score: {overall_f1:.4f}")