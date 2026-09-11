# 制造问题的输入
import numpy as np


def generate_inputs(n_samples=1000, seed=42):
    rng = np.random.default_rng(seed) #创建seed42的随机数生成器

    #从正态分布里抽随机数
    X = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_samples, 4)
    )

    return X

# 定义y_A和y_B
def generate_labels(X):
    # 取出四个特征
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    x4 = X[:, 3]

    # Task A
    yA = (x1 + x2 > 0).astype(int)

    # Task B
    yB_high = (x1 + 0.5 * x2 > 0).astype(int)
    yB_medium = (x1 + x3 > 0).astype(int)
    yB_low = (x3 + x4 > 0).astype(int)

    return yA, yB_high, yB_medium, yB_low

if __name__ == "__main__":
    X = generate_inputs()

    print("Shape:", X.shape)
    print("\nFirst 5 samples:")
    print(X[:5])

    print("\nMean of each feature:")
    print(X.mean(axis=0)) #axis=0:沿着行的方向往下算。理论上应该接近0

    print("\nStandard deviation of each feature:")
    print(X.std(axis=0)) #理论上应该接近1

    # 生成标签
    yA, yB_high, yB_medium, yB_low = generate_labels(X)

    print("\nyA shape:", yA.shape)
    print("yB_high shape:", yB_high.shape)
    print("yB_medium shape:", yB_medium.shape)
    print("yB_low shape:", yB_low.shape)

    print("\nFirst 10 labels:")
    print("A:       ", yA[:10])
    print("B high:  ", yB_high[:10])
    print("B medium:", yB_medium[:10])
    print("B low:   ", yB_low[:10])

    print("\nPositive ratio:")
    print("A:", yA.mean())
    print("B high:", yB_high.mean())
    print("B medium:", yB_medium.mean())
    print("B low:", yB_low.mean())