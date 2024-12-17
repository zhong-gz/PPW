import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# 生成线性回归数据集
np.random.seed(0)  # 为了可重复性
X = 2 * np.random.rand(100, 1)  # 生成100个随机数作为自变量
y = 4 + 3 * X + np.random.randn(100, 1)  # 生成因变量，添加一些噪声

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1))

# 创建线性回归模型
model = Ridge(alpha = 1) #, fit_intercept=False
model.fit(X, y)  # 拟合模型

# 进行预测
X_new = np.array([[-2], [2]])  # 用于绘制预测线
y_predict = model.predict(X_new)

# 8. 查看模型的系数和截距
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')

# 绘制数据点和回归线
plt.scatter(X, y, color='blue', label='data')  # 绘制数据点
plt.plot(X_new, y_predict, color='red', label='regression function')  # 绘制回归线
plt.xlabel('X')
plt.ylabel('y')
plt.title('ridge regression')
plt.legend()
plt.show()
