import os
import numpy as np
import glob
from sklearn.metrics import accuracy_score, f1_score

def evaluate_results(result_dir):
    results = {}
    file_pattern = os.path.join(result_dir, "results", "test_logits_labels_sub*_seed*.npz")
    files = glob.glob(file_pattern)

    for file in files:
        # Extract sub and seed from filename
        basename = os.path.basename(file)
        sub = basename.split("sub")[1].split("_")[0]
        seed = basename.split("seed")[1].split(".")[0]

        data = np.load(file)
        logits = data['logits']
        labels = data['labels']
        # Compute predicted labels
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        

        if sub not in results:
            results[sub] = {}
        results[sub][seed] = {'accuracy': acc, 'f1': f1}

    # Compute averages per subject
    subject_avgs = {}
    for sub, seeds in results.items():
        accs = [v['accuracy'] for v in seeds.values()]
        f1s = [v['f1'] for v in seeds.values()]
        subject_avgs[sub] = {
            'avg_accuracy': np.mean(accs),
            'avg_f1': np.mean(f1s)
        }

    # Compute overall averages
    all_accs = [v['avg_accuracy'] for v in subject_avgs.values()]
    all_f1s = [v['avg_f1'] for v in subject_avgs.values()]
    overall_avg_accuracy = np.mean(all_accs)
    overall_avg_f1 = np.mean(all_f1s)

    print("Per subject averages:")
    for sub, vals in subject_avgs.items():
        print(f"Subject {sub}: Accuracy={vals['avg_accuracy']:.4f}, F1={vals['avg_f1']:.4f}")
    print(f"\nOverall average: Accuracy={overall_avg_accuracy:.4f}, F1={overall_avg_f1:.4f}")


if __name__ == "__main__":
    result_directory = "./logs/Baselines-001-all-e300-b64/mdd-BNCI2014001-all-EEGNet-4,2-e300-b64-1/"
    evaluate_results(result_directory)