import numpy as np
from sklearn.model_selection import train_test_split

from generate_data import generate_inputs, generate_labels


X = generate_inputs()
yA, yB_high, yB_medium, yB_low = generate_labels(X)


# 先生成样本编号
indices = np.arange(len(X))


# 第一次划分：
# 70% train, 30% temporary
train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=42 #保证结果能够复现
)


# 第二次划分：
# 把剩下的30%平均分成 validation 和 test
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42
)


# 根据相同的 index 切 X
X_train = X[train_idx]
X_val = X[val_idx]
X_test = X[test_idx]


# Task A
yA_train = yA[train_idx]
yA_val = yA[val_idx]
yA_test = yA[test_idx]


# Task B high
yB_high_train = yB_high[train_idx]
yB_high_val = yB_high[val_idx]
yB_high_test = yB_high[test_idx]


# Task B medium
yB_medium_train = yB_medium[train_idx]
yB_medium_val = yB_medium[val_idx]
yB_medium_test = yB_medium[test_idx]


# Task B low
yB_low_train = yB_low[train_idx]
yB_low_val = yB_low[val_idx]
yB_low_test = yB_low[test_idx]


print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)