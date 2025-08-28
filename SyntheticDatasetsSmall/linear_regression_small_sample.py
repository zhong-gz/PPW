import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
sys.path.insert(0, sys.path[0]+"/../")
sys.path.insert(0, sys.path[0]+"Algorithm")
import numpy as np
import random
from Algorithm.alg_method_1 import method_1
from Algorithm.plot_par_sen import plot_fig_par
from functions import linear_data_generation
import os

methods = ['RRM Linear Regression','RGD Linear Regression','Two-Stage Approach','PerfGD','DFO','PPW']

# problems parameters
seed_value = 42
num_iters = 100
d_list = [2,4,8,16] 
num_experiments = 10
map = 1
# folder_path = 'SyntheticDatasets/result/'
# folder_path = 'SyntheticDatasets/result_uniform/'
# folder_path = 'SyntheticDatasetsSmall/result{n}/'

np.random.seed(seed_value)
random.seed(seed_value)
n_features = 20
folder_path = f'SyntheticDatasetsSmall/result/'
os.makedirs(folder_path, exist_ok=True)
strat_features = None
ns = [0.05,0.1,0.2,1]

# for n_p in ns:
#     print('n_p = ',n_p)
#     n = int(n_p*100)
#     X,y = linear_data_generation(n = n, n_features=n_features)
#     d = X.shape[1]
#     print('Sample number : ',n)
#     print('Sample dimension : ',d)
#     print('-'*50)
#     # method 1
#     model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
#     method_1(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value,alpha_1 = 2.1)
#     file_name_npy = f"{folder_path}{n_p}.npz"
#     np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
#                 mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
#                 mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
#     print(f"Data saved to {file_name_npy}")

rmse_min = 0.2
rmse_max = 5
gap_min = 1e-10
gap_max = 1

plot_fig_par(num_iters,d_list,folder_path,ns,latex_text = "n",rmse_min=rmse_min, rmse_max=rmse_max, gap_min=gap_min, gap_max=gap_max)