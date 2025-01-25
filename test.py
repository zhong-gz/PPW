import numpy as np

def logistic_loss(y_true, y_pred):

    # 确保输入是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 防止出现log(0)的情况
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # 计算每个样本的损失
    loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    
    # 返回平均损失
    return np.mean(loss)

# 示例使用
y_true = [0, 1, 1, 0, 1]
y_pred = [0.1, 0.9, 0.8, 0.3, 0.95]

loss = logistic_loss(y_true, y_pred)
print(f"Logistic Loss: {loss}")
