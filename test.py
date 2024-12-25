import numpy as np

# 创建一个 2x3 的矩阵
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6]])

# 在矩阵的每个元素上重复 2 次
repeated_matrix = np.repeat(matrix, repeats=2)

print("原始矩阵:")
print(matrix)
print("\n重复后的矩阵（展平为一维）:")
print(repeated_matrix)

# 在轴 0（行）上重复
repeated_matrix_axis0 = np.repeat(matrix, repeats=2, axis=0)
print("\n在轴 0 上重复的矩阵:")
print(repeated_matrix_axis0)

# 在轴 1（列）上重复
repeated_matrix_axis1 = np.repeat(matrix, repeats=2, axis=1)
print("\n在轴 1 上重复的矩阵:")
print(repeated_matrix_axis1)
