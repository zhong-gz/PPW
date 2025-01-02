import matplotlib.pyplot as plt
import numpy as np

# 定义x的范围
x = np.linspace(0, 5, 400)

# 计算y的值
y = 0.5 ** x

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制函数图像
plt.plot(x, y, label=r'$y = 0.5^x$')

# 添加标题和标签
plt.title('Graph of $y = 0.5^x$')
plt.xlabel('$x$')
plt.ylabel('$y$')

# 添加网格
plt.grid(True)

# 添加图例
plt.legend()

# 显示图像
plt.show()
