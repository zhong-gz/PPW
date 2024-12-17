import numpy as np

def transform_arrays(a, b, alpha):
    # 将数组 a 归一化到 [0, 1]
    mean_a = np.mean(a)

    b_transformed = b - alpha*(a-mean_a)
    print(alpha*(a-mean_a))
    
    return b_transformed

# 示例数组
a = np.array([10, 20, 30, 40, 50])
b = np.array([1, 2, 3, 4, 5])

# 变换数组，调整 alpha 控制变化幅度
alpha = 0.01  # 你可以根据需要调整 alpha 的值
b_transformed = transform_arrays(a, b, alpha)

print("原始数组 a:", a)
print("原始数组 b:", b)
print("变换后的数组 b:", b_transformed)
