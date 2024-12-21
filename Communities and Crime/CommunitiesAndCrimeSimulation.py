import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
sys.path.insert(0, sys.path[0]+"Algorithm")
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from Algorithm.plot import plot_fig
from Algorithm.alg_method_1 import method_1
from Algorithm.alg_RRM import RRM

# problems parameters
seed_value = 42
num_iters = 100
d_list = [0.8]
num_experiments = 10
map = 2
np.random.seed(seed_value)
random.seed(seed_value)
folder_path = 'Communities and Crime/result/'

initial=pd.read_csv('Communities and Crime/communities-crime-clean.csv')
initial = initial.drop('communityname', axis=1)
initial = initial.drop('fold', axis=1)
initial = initial.drop('state', axis=1)
y = initial['ViolentCrimesPerPop'].values.reshape(-1, 1) * 10
initial = initial.drop('ViolentCrimesPerPop', axis=1)
X = initial.values

# method 1
model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    method_1(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
print(file_name_npy)
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
            mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
print(f"Data saved to {file_name_npy}")

# RRM
model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    RRM(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
            mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
print(f"Data saved to {file_name_npy}")

plot_fig(num_iters,d_list,folder_path)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 4. 创建岭回归模型
# ridge_model = Ridge(alpha=1)  # alpha 是正则化参数，可以根据需要调整

# # 5. 训练模型
# ridge_model.fit(X_train, y_train)

# # 6. 进行预测
# y_pred = ridge_model.predict(X_test)

# # 7. 评估模型
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')

# # 8. 查看模型的系数和截距
# # print(f'Coefficients: {ridge_model.coef_}')
# print(f'Intercept: {ridge_model.intercept_}')
