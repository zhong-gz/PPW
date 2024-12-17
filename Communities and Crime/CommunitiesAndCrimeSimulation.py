import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# problems parameters
threshold = 0.1
seed_value = 42
num_iters = 100
d_list = [5,7.5,10]
num_experiments = 10
map = 2
np.random.seed(seed_value)
random.seed(seed_value)

initial=pd.read_csv('Communities and Crime/communities-crime-clean.csv')
initial = initial.drop('communityname', axis=1)
initial = initial.drop('fold', axis=1)
initial = initial.drop('state', axis=1)
y = initial['ViolentCrimesPerPop'].values - 0.48088582168965543
initial = initial.drop('ViolentCrimesPerPop', axis=1)
X = initial.values

# n = X.shape[0]
# d = X.shape[1]
# print('Sample number : ',n)
# print('Sample dimension : ',d)

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1))
# y = y-0.48088582168965543

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 创建岭回归模型
ridge_model = Ridge(alpha=1, fit_intercept=False)  # alpha 是正则化参数，可以根据需要调整

# 5. 训练模型
ridge_model.fit(X_train, y_train)

# 6. 进行预测
y_pred = ridge_model.predict(X_test)

# 7. 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 8. 查看模型的系数和截距
# print(f'Coefficients: {ridge_model.coef_}')
print(f'Intercept: {ridge_model.intercept_}')
