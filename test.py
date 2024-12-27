import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 生成二维数据X和标签y
np.random.seed(0)
num_samples = 100
num_features = 2

X = np.random.rand(num_samples, num_features)
true_weights = np.array([1.5, -2.0])
true_bias = 3.0
noise = np.random.randn(num_samples) * 0.1
y = X.dot(true_weights) + true_bias + noise

# 创建线性回归模型
model = LinearRegression()
model.fit(X, y)

# 模型的权重和偏置
print("Learned weights:", model.coef_)
print("Learned intercept:", model.intercept_)

# 绘制原始数据和模型的预测结果
plt.figure(figsize=(10, 5))

# 绘制第一个特征与标签的关系
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], y, color='blue', label='Original data')
plt.scatter(X[:, 0], model.predict(X), color='red', label='Fitted data')
plt.xlabel('Feature 1')
plt.ylabel('Label')
plt.title('Feature 1 vs Label')
plt.legend()

# 绘制第二个特征与标签的关系
plt.subplot(1, 2, 2)
plt.scatter(X[:, 1], y, color='blue', label='Original data')
plt.scatter(X[:, 1], model.predict(X), color='red', label='Fitted data')
plt.xlabel('Feature 2')
plt.ylabel('Label')
plt.title('Feature 2 vs Label')
plt.legend()

plt.tight_layout()
plt.show()
