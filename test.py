import numpy as np

def ridge_regression(X, y, lambda_, learning_rate=0.01, num_iterations=1000):
    m, n = X.shape
    # 初始化权重
    w = np.zeros(n)
    
    for i in range(num_iterations):
        # 计算预测值
        y_pred = X.dot(w)
        # 计算损失函数的梯度
        gradient = - (1/m)*(X.T.dot(y - y_pred)) + lambda_ * w
        # 更新权重
        w -= learning_rate * gradient
    
    return w

# 示例数据
if __name__ == "__main__":
    # 创建特征矩阵 X 和目标值 y
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([1, 2, 2, 3])
    
    # 正则化参数
    lambda_ = 1.0
    
    # 训练岭回归模型
    weights = ridge_regression(X, y, lambda_)
    
    print("学习到的权重:", weights)
