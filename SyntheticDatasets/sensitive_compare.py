import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
import numpy as np
import random
from functions import linear_data_generation_sensitive,est_varepsilon,data_distribution_map1,remove_outliers_iqr
import ot  
from sklearn.linear_model import Ridge
import copy

def chi2_distance(observed, expected):
    """
    计算卡方距离
    :param observed: 观察值数组
    :param expected: 期望值数组
    :return: 卡方距离
    """
    # 处理expected中为0的情况，避免除零错误
    expected = np.where(expected == 0, np.finfo(float).eps, expected)
    return np.sum((observed - expected) ** 2 / expected)

# problems parameters
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
num_iters = 10
d_list = [2,4] #,8,16] #,1000,10000 10,100,1000,60,80,100
dimension_list = np.arange(10, 1011, 100)
num_experiments = 100
map = 1
folder_path = 'SyntheticDatasets/sensitive_result/'
n = 100

num_d  = len(d_list)
num_dimension_list = len(dimension_list)


episilon_sensitive    = np.zeros((num_d, num_dimension_list)) 
episilon_w_1          = np.zeros((num_d, num_dimension_list))
episilon_chi_square   = np.zeros((num_d, num_dimension_list))

for k, d in enumerate(d_list):
    print('     Running epsilon =  {}'.format(d))

    for n_d,n_features in enumerate(dimension_list):
        print(f'          Running with {n_features} features', end='\r')
        true_coef = np.random.rand(n_features)
        X,y = linear_data_generation_sensitive(n = n,n_features = n_features,true_coefficients = true_coef)
        n = X.shape[0]
        model_int = Ridge(alpha = 1)
        model_int.fit(X, y)
        norm_w_w = []
        varepsilon = []
        ridge_model = copy.deepcopy(model_int)
        X_old = np.copy(X)
        y_old = np.copy(y)
        model_list = [model_int]
        for i in range(num_experiments):
            seed_value = seed_value+1
            np.random.seed(seed_value)
            random.seed(seed_value)
            # print(f'       Current iteration =  {i}', end='\r')

            X,y = linear_data_generation_sensitive(n = n,n_features = n_features,true_coefficients = true_coef)
            X_strat,y_strat = data_distribution_map1(X, y,mu = d, model = ridge_model)

            ridge_model_new = Ridge(alpha = 1)#, fit_intercept=False)
            ridge_model_new.fit(X_strat, y_strat)

            # keep track of statistics
            model_list.append(ridge_model_new)
            varepsilon_star,norm_w_w = est_varepsilon(X_old,y_old,X_strat,y_strat,model_list,norm_w_w)
            varepsilon.append(varepsilon_star)
            episilon_sensitive[k,n_d] = max(varepsilon)

            data_p = np.concatenate((X_old, y_old.reshape(-1, 1)), axis=1)
            data_q = np.concatenate((X_strat, y_strat.reshape(-1, 1)), axis=1)
            # 计算Wasserstein距离（使用Python Optimal Transport库）
            # 首先计算样本的经验分布
            a, b = np.ones((n,)) / n, np.ones((n,)) / n
            M = ot.dist(data_p, data_q)  # 计算距离矩阵
            w1_distance_pot = ot.emd2(a, b, M)/norm_w_w[-1]
            episilon_w_1[k,n_d] = max(w1_distance_pot,episilon_w_1[k,n_d])

            # 计算卡方距离
            # 逐维计算直方图
            bins = 100
            hist_p_list = []
            hist_q_list = []
            for j in range(n_features):
                hist_p_i, _ = np.histogram(data_p[:, j], bins=bins)
                hist_q_i, _ = np.histogram(data_q[:, j], bins=bins)
                hist_p_list.append(hist_p_i)
                hist_q_list.append(hist_q_i)

            hist_p = np.concatenate(hist_p_list)
            hist_q = np.concatenate(hist_q_list)

            # 将直方图转换为概率分布
            prob_p = hist_p / np.sum(hist_p)
            prob_q = hist_q / np.sum(hist_q)
            chi2_dist = chi2_distance(prob_p, prob_q)/norm_w_w[-1]
            episilon_chi_square[k,n_d] = max(chi2_dist,episilon_chi_square[k,n_d])

            X_old = np.copy(X_strat)
            y_old = np.copy(y_strat)
            ridge_model = copy.deepcopy(ridge_model_new)
    print('')
    print('-'*50)


# episilon_sensitive_avg = np.nanmean(episilon_sensitive, axis=2)
# episilon_w_1_avg = np.nanmean(episilon_w_1, axis=2)
# episilon_chi_square_avg = np.nanmean(episilon_chi_square, axis=2)
# print('episilon_sensitive :',np.std(episilon_sensitive, axis=1))
# print('episilon_w_1       :',np.std(episilon_chi_square, axis=1))
# print('episilon_chi_square:',np.std(episilon_chi_square, axis=1))

print('episilon_sensitive :',episilon_sensitive)
print('episilon_w_1       :',episilon_w_1)
print('episilon_chi_square:',episilon_chi_square)