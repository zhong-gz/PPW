import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, k=1, b=0):
    return 1 / (1 + np.exp(-k * (x - b)))

# 生成 x 值
x = np.linspace(-10, 10, 100)

# 不同参数的 Sigmoid 函数
y1 = sigmoid(x)          # k=1, b=0
y2 = sigmoid(x, k=2)     # k=2, b=0 (更陡)
y3 = sigmoid(x, k=0.5)   # k=0.5, b=0 (更平)
y4 = sigmoid(x, k=10)   # k=0.5, b=0 (更平)

# 绘图
plt.plot(x, y1, label='k=1, b=0')
plt.plot(x, y2, label='k=2, b=0')
plt.plot(x, y3, label='k=0.5, b=0')
plt.plot(x, y4, label='k=10, b=0')
plt.title('Adjusted Sigmoid Functions')
plt.xlabel('hat_y')
plt.ylabel('sigmoid(hat_y)')
plt.legend()
plt.grid()
plt.show()
