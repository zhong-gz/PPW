import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
sys.path.insert(0, sys.path[0]+"Algorithm")
import numpy as np
import random
import pandas as pd
from Algorithm.alg_method_1 import method_1
from Algorithm.plot_par_sen import plot_fig_par
import warnings
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import os

# problems parameters
seed_value = 42
num_iters = 100
d_list = [2,4,8,16] 
num_experiments = 10
map = 3
np.random.seed(seed_value)
random.seed(seed_value)

# selecting the noise level, result2 for std = 0.2, and change the std in 'function.py'-'def data_distribution_map3'
folder_path = f'StudentMarks/result_small/'
os.makedirs(folder_path, exist_ok=True)

initial=pd.read_csv('StudentMarks/Student_Marks.csv')
y = initial[["Marks"]].values
scaler = MinMaxScaler()
y_ini = scaler.fit_transform(y)*1.1
X = initial[["number_courses","time_study"]].values
X_ini = scaler.fit_transform(X)*1.1 #avoid gradient disappear in gradient methods


ns = [0.05,0.1,0.2,0.4]

for n_p in ns:
    print('n_p = ',n_p)
    n = int(n_p*100)#X_ini.shape[0])
    random_indices = np.random.choice(X_ini.shape[0], size=n, replace=False)
    X = X_ini[random_indices, :]
    y = y_ini[random_indices, :]
    # method 1
    model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    method_1(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value,alpha_1 = 2.1)
    file_name_npy = f"{folder_path}{n_p}.npz"
    np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
                mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
                mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
    print(f"Data saved to {file_name_npy}")

rmse_min = 0.2
rmse_max = 1.5
gap_min = 1e-7
gap_max = 1

plot_fig_par(num_iters,d_list,folder_path,ns,latex_text = "n",rmse_min=rmse_min, rmse_max=rmse_max, gap_min=gap_min, gap_max=gap_max)