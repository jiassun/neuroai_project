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


if __name__ == "__main__":
    X = generate_inputs()

    print("Shape:", X.shape)
    print("\nFirst 5 samples:")
    print(X[:5])

    print("\nMean of each feature:")
    print(X.mean(axis=0)) #axis=0:沿着行的方向往下算。理论上应该接近0

    print("\nStandard deviation of each feature:")
    print(X.std(axis=0)) #理论上应该接近1