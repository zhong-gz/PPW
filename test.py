import numpy as np

# 创建一个向量
vector = np.array([1, 4, 9, 16, 25])

# 检查是否所有元素都是非负的
if np.all(vector >= 0):
    # 将向量的每个元素开根号
    sqrt_vector = np.sqrt(vector)
    print("原始向量:", vector)
    print("开根号后的向量:", sqrt_vector)
else:
    print("向量中包含负数，无法进行开根号操作。")
