from sklearn.linear_model import LinearRegression
import numpy as np

# 生成示例数据
np.random.seed(0)
X = 2 * np.random.rand(100, 3)  # 输入矩阵 (100, 3)
true_W = np.array([[4, 5], [3, 2], [1, 3], [2, 4]])  # 参数矩阵 (4, 2) 包括截距项
Y = X.dot(true_W[1:, :]) + true_W[0, :] + np.random.randn(100, 2)  # 输出矩阵 (100, 2)

# 使用 scikit-learn 进行线性回归
lin_reg = LinearRegression(fit_intercept=False)
lin_reg.fit(X, Y)

print("权重矩阵:", lin_reg.coef_)

# 测试预测
X_new = np.array([[1, 2, 3], [4, 5, 6]])
Y_predict = lin_reg.predict(X_new)
print("预测值:", Y_predict)
