import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

class LinearRegressionGD:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.theta = None

    def fit(self, X, y, model=None):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 在X的第一列添加1
        m = X_b.shape[0]  # 样本数量

        if model is None and self.theta is None:
            LR = Ridge(alpha=0, fit_intercept=False)
            LR.fit(X_b, y)
            self.theta = LR.coef_.T.copy()
        # elif model is not None:
        #     self.theta = model

        gradients = (2 / m) * X_b.T.dot(X_b.dot(self.theta) - y)  # 计算梯度
        self.theta -= self.learning_rate * gradients  # 更新参数

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 在X的第一列添加1
        return X_b.dot(self.theta)

seed_value = 42
num_iters = 100
d_list = [0] #
num_experiments = 10
map = 2
np.random.seed(seed_value)
folder_path = 'Communities and Crime/result/'

initial=pd.read_csv('Communities and Crime/communities-crime-clean.csv')
initial = initial.drop('communityname', axis=1)
initial = initial.drop('fold', axis=1)
initial = initial.drop('state', axis=1)
y = initial['ViolentCrimesPerPop'].values.reshape(-1, 1)
initial = initial.drop('ViolentCrimesPerPop', axis=1)
X = initial.values

# 创建线性回归模型实例
model = LinearRegressionGD(learning_rate=0.01)

# 拟合模型并显示每次更新后的参数
n_iter = 1000
for i in range(n_iter):
    model.fit(X, y)
    y = y + np.random.normal(0, 0.1, size=y.shape)
    # print(f"Iteration {i+1}: theta = {model.theta.ravel()}")

# 预测
predictions = model.predict(X)

print(mean_squared_error(y, predictions))
