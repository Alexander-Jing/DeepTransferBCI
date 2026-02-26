# python

import numpy as np
from sklearn.metrics import accuracy_score

# 加载数据
data = np.load('./logs/Baselines-001-all-e300-b64-debugs/cdan-BNCI2014001-test-EEGNet-4,2-e300-b64/results/test_logits_labels_sub0_seed1.npz')
logits = data['logits']
labels = data['labels']
acc_t = data['acc_t']

# 预测类别
preds = np.argmax(logits, axis=1)

# 计算准确率
acc = accuracy_score(labels, preds)
print(f"Test Accuracy: {acc * 100:.2f}%")