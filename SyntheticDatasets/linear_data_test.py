import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error

def data_distribution_map(mu,X,y,model):
    # 将数组 a 归一化到 [0, 1]
    hat_y = model.predict(X)
    mean_hat_y = np.mean(hat_y)
    scaler = MinMaxScaler()
    # y_transformed = scaler.fit_transform(y - mu*(hat_y-mean_hat_y)) + np.random.normal(0, 0.1, size=y.shape)
    # y_transformed = y - mu*(hat_y-mean_hat_y) + np.random.normal(0, 0.1, size=y.shape)
    y_transformed = y - mu*(hat_y-0.5*mean_hat_y)
    # y_transformed = 1.67 * X + mu*hat_y
    x_transformed = X #- mu*(hat_y-0.5*mean_hat_y)

    y_transformed = scaler.fit_transform(y_transformed)
    x_transformed = scaler.fit_transform(x_transformed)
    return x_transformed,y_transformed

# 生成线性回归数据集
np.random.seed(0)  # 为了可重复性

num_iters = 50

# X = np.random.rand(200, 1) + 1.67  # 生成100个随机数作为自变量
# # X_b = np.c_[np.ones((X.shape[0], 1)), X]
# # scaler = MinMaxScaler()
# y = 1.67 + 1.67 * X

mu = 5

# X = np.random.randn(200, 1) + 1.67
# y =  1.67*X

# model = Ridge(alpha = 0, fit_intercept=False) 
# model.fit(X, y)


for t in range(num_iters):
    X = np.random.randn(200, 1) + 1.67   # 生成100个随机数作为自变量
    # X_b = np.c_[np.ones((X.shape[0], 1)), X]
    scaler = MinMaxScaler()
    y = 1.67 * X #+ 0.1*np.random.randn(200, 1)  # 生成因变量，添加一些噪声
    y = scaler.fit_transform(y)
    # percentage_to_shuffle = 0.2
    # num_elements_to_shuffle = int(len(y) * percentage_to_shuffle)
    # indices = np.random.choice(len(y), num_elements_to_shuffle, replace=False)
    # elements_to_shuffle = y[indices]
    # y[indices] = elements_to_shuffle

    if t==0:
        x_new = X
        y_new = y
        mse_start = 0
    else:
        x_new,y_new = data_distribution_map(mu,X,y,model)
        hat_y = model.predict(X)
        mse_start = mean_squared_error(hat_y,y_new)
        # print('mse start:',mse_start)

    # 创建线性回归模型
    model = Ridge(alpha = 0)#, fit_intercept=False) 
    model.fit(x_new, y_new)  # 拟合模型

    # hat_y = model.predict(X)
    # mse_end = mean_squared_error(hat_y,y_new)
    # print('mse end:',mse_end)

    # print(np.abs(mse_end-mse_start))

    X_plot = np.array([[-100], [100]])  # 用于绘制预测线
    # X_b_plot = np.c_[np.ones((X_plot.shape[0], 1)), X_plot]
    y_plot = model.predict(X_plot)

    plt.clf()
    # 绘制数据点和回归线
    plt.scatter(x_new, y_new, color='blue', label='data')  # 绘制数据点
    plt.plot(X_plot, y_plot, color='red', label='regression function')  # 绘制回归线
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('ridge regression')
    plt.xlim(-4 ,4)  
    plt.ylim(-8, 8)
    # plt.legend()
    plt.pause(0.2) 
plt.show() 
