import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import StandardScaler,MinMaxScaler

def data_distribution_map(mu,X,y,model):
    # 将数组 a 归一化到 [0, 1]
    hat_y = model.predict(X)
    mean_hat_y = np.mean(hat_y)

    y_transformed = y - mu*(hat_y-mean_hat_y)
    
    return y_transformed

# 生成线性回归数据集
np.random.seed(0)  # 为了可重复性
X = 2 * np.random.rand(100, 1)  # 生成100个随机数作为自变量
y = 3 * X + np.random.randn(100, 1)  # 生成因变量，添加一些噪声

num_iters = 100
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1))

mu = 1

for t in range(num_iters):
    if t==0:
        y_new = y
    else:
        y_new = data_distribution_map(mu,X,y,model)

    # 创建线性回归模型
    model = Ridge(alpha = 20) #, fit_intercept=False
    model.fit(X, y_new)  # 拟合模型

    X_plot = np.array([[-2], [2]])  # 用于绘制预测线
    y_plot = model.predict(X_plot)

    plt.clf()
    # 绘制数据点和回归线
    plt.scatter(X, y_new, color='blue', label='data')  # 绘制数据点
    plt.plot(X_plot, y_plot, color='red', label='regression function')  # 绘制回归线
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('ridge regression')
    # plt.legend()
    plt.pause(0.2) 
plt.show() 
