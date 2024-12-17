import numpy as np

def transform_array(arr):
    # 计算最大值和最小值
    # max_val = np.max(arr)
    # min_val = np.min(arr)
    mean_val = np.mean(arr)
    
    # 设定变换的幅度
    scale_factor = 0.1  # 控制变换的强度，0 < scale_factor < 1
    shift = 0.1  # 设定一个小的偏移量
    
    # 进行变换
    transformed_arr = (1 - scale_factor) * arr + scale_factor * mean_val / 2 
    
    return transformed_arr

# 示例数组
array = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
transformed_array = transform_array(array)

print("原数组:", array)
print("变换后的数组:", transformed_array)
