import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None  # 参数，包括截距项

    def fit(self, X, y):
        # 添加截距项
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 在X的第一列添加1
        m = X_b.shape[0]  # 样本数量

        # 初始化参数
        self.theta = np.random.randn(X_b.shape[1], 1)  # 随机初始化参数，确保是列向量

        # 梯度下降
        for iteration in range(self.n_iterations):
            gradients = (2/m) * X_b.T.dot(X_b.dot(self.theta) - y)  # 计算梯度
            self.theta -= self.learning_rate * gradients  # 更新参数

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加截距项
        return X_b.dot(self.theta)  # 预测值

# 使用示例
if __name__ == "__main__":
    # 生成一些示例数据
    X = 2 * np.random.rand(100, 1)  # 100个样本，1个特征
    y = 4 + 3 * X + np.random.randn(100, 1)  # 线性关系，加上随机噪声

    # 创建并训练模型
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    # 进行预测
    X_new = np.array([[0], [2]])  # 新的数据点
    predictions = model.predict(X_new)

    print("预测结果:", predictions)
