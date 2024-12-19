import numpy as np
import random
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy.stats as st
from sklearn.datasets import make_classification
from sklearn.datasets import make_moons,make_circles
from sklearn.preprocessing import StandardScaler

# D(w) = X - mu * w
def data_distribution_map1(X,y, mu = 0, model = None, strat_features = None):
    if model is not None:
        y_strat = np.copy(y)
        if strat_features is None:
            strat_features = np.arange(X.shape[1])
        m = len(strat_features)

        # the number of generating random points
        n = 100
        X_strat = np.zeros_like(X)
        _,pre_label = model.predict(X)
        random_features_point = generate_random_points_in_circle(mu, n,m)
        random_point = np.zeros([n,X.shape[1]])
        random_point[:, strat_features] = random_features_point
        for i in range(X.shape[0]):
            # if y[i] == 1 :
            #     X_strat[i] = np.copy(X[i])
            # else:
            x_temp = random_point + X[i]
            score,_ = model.predict(x_temp)
            min_index = np.argmin(score)
            X_strat[i] = np.copy(x_temp[min_index])
    else:
        X_strat = np.copy(X)
        y_strat = np.copy(y)
    return X_strat,y_strat

# D(w) in crime rate prediction
def data_distribution_map2(X,y, mu = 0, model = None):
    scaler = StandardScaler()
    y_strat = np.zeros_like(y)
    hat_y = model.predict(X)
    scaled_hat_y = scaler.fit_transform(hat_y)
    mean_scaled_hat_y = np.mean(scaled_hat_y)
    y_strat = y - mu*(scaled_hat_y-mean_scaled_hat_y)
    return X,y_strat

# D(w) = X - mu * w
def data_distribution_map3(X,y, mu = 0, model = None, strat_features = None,t = 0):
    if model is not None:
        y_strat = np.copy(y)
        if strat_features is None:
            strat_features = np.arange(X.shape[1])
        d = -0.01 * mu + 15  #-2.5 * np.log10(mu) + 17.5
        # d = 20
        m = len(strat_features)
        # the number of generating random points
        n = 100
        X_strat = np.zeros_like(X)
        score,pre_label = model.predict(X)
        random_features_point = generate_random_points_in_circle(mu, n,m)
        random_point = np.zeros([n,X.shape[1]])
        random_point[:, strat_features] = random_features_point
        for i in range(X.shape[0]):
            # if pre_label[i] == -1 :
            # random_point = np.zeros([n,X.shape[1]])
            # random_point[:, strat_features] = random_features_point
            # for j in range(m):
            #     random_point[:,strat_features[j]] = random_features_point[:,j]
            x_temp = random_point + X[i]
            scorei,_ = model.predict(x_temp)
            min_index = np.argmin(scorei) # np.argmax(score)
            X_strat[i] = np.copy(x_temp[min_index])

            if (y[i] == 1) and (t > 1):
                # score_larger_0 = score[score > 0]
                max_score = max(score) #score
                max_min_score = 0
                min_score = min(score)
                min_max_score = 0
                if pre_label[i] == 1:
                    p = (score[i]-max_min_score)/(max_score-max_min_score)
                else:
                    p = (score[i]-min_max_score)/(min_score-min_max_score)
                p = np.exp(d * p) / (1 + np.exp(d * p))
                if random.random() < p:
                    y_strat[i] = np.copy(y[i])
                else:
                    y_strat[i] = -1*np.copy(y[i])
    else:
        X_strat = np.copy(X)
        y_strat = np.copy(y)
    return X_strat,y_strat

def linear_data_generation(n = 100):
    # # np.random.seed(0)
    # cov = [[1, 0], [0, 1]]

    # # 生成类别1的数据
    # class1_center = [0, 0]
    # class1_samples = np.random.multivariate_normal(class1_center, cov, int(n/2))

    # # 生成类别2的数据
    # class2_center = [-5, 5]
    # class2_samples = np.random.multivariate_normal(class2_center, cov, int(n/2))

    # X = np.concatenate((class1_samples, class2_samples), axis=0)
    # y = np.concatenate((-1*np.ones(int(n/2)), np.ones(int(n/2))))

    X, y = make_classification(n_samples=n, n_features=50, n_informative=40, n_redundant=0, random_state=42)
    y[y != 1] = -1
    return X,y

def generate_half_moon(n_samples, radius, width):
    angles = np.linspace(0, np.pi, n_samples)
    noise = np.random.uniform(-width, width, n_samples)
    
    x = (radius + noise) * np.cos(angles)
    y = (radius + noise) * np.sin(angles)
    
    return x, y

def non_linear_data_generation(n = 100):
    # X,y = make_moons(n_samples=n, noise=0.2)
    
    X,y = make_circles(n_samples=n, noise=0.2,factor = 0.1)

    y[y != 1] = -1
    # # 生成类别1的半圆数据点（上半圆）
    # radius = 10
    # width = 2
    # x1, y1 = generate_half_moon(int(n/2), radius, width)

    # # 生成类别2的半圆数据点（下半圆）
    # x2, y2 = generate_half_moon(int(n/2), radius, width)
    # x2 = x2 + 10  
    # y2 = -y2 + 2

    # class1_samples = np.vstack((x1, y1)).T
    # class2_samples = np.vstack((x2, y2)).T

    # X = np.concatenate((class1_samples, class2_samples), axis=0)
    # y = np.concatenate((-1*np.ones(int(n/2)), np.ones(int(n/2))))
    return X,y

def generate_random_points_in_circle(mu, num_points,d):
    data = []
    for _ in range(num_points):
        # 生成d-1维球面的坐标
        angles = np.random.uniform(0, 2*np.pi, d)  # 生成d-1维均匀分布的角度
        radius = np.random.uniform(0, mu)  # 生成[0, mu^(1/d))范围内的随机半径

        # 构建d维球面坐标
        coords = [mu * np.prod(np.sin(angles[:i])) * np.cos(angles[i]) for i in range(d-1)]
        coords.append(mu * np.prod(np.sin(angles[:d])))

        data.append(coords)
    
    return np.array(data)

def accuracy(y_true, y_pred):
    correct = 0
    total = len(y_true)
    
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    
    accuracy = correct / total
    return accuracy
    

def est_varepsilon(X,y,X_new,y_new,w_arr,norm_w_w):
    n = X.shape[0]
    n_new = X_new.shape[0]

    mean_value = [0,0]
    w_t = w_arr[-1]
    # w_{t} x_{t-1}
    score,_ = w_t.predict(X)
    y_score = y*score
    mask = y_score < 1
    new_arr = np.copy(y_score[mask])
    if new_arr.size == 0:
        mean_value[0] = 0
    else:
        mean_value[0] = np.sum(new_arr)/(n)

    # w_{t} x_{t}
    score,_ = w_t.predict(X_new)
    y_score = y_new*score
    mask = y_score < 1
    new_arr = np.copy(y_score[mask])
    if new_arr.size == 0:
        mean_value[1] = 0
    else:
        mean_value[1] = np.sum(new_arr)/(n_new)
    w_t_norm = w_t.norm()

    if len(w_arr) == 1:
        norm_w_w.append(w_t.norm())
    # else:
        # norm_w_w.append(model_sub_norm(w_arr[-1],w_arr[-2]))

    if norm_w_w[-1] == 0:
        norm_w_w[-1] = 1e-4
    if w_t_norm == 0:
        w_t_norm = 1e-4

    est_epsilon = (np.abs(mean_value[0]-mean_value[1]))/(norm_w_w[-1]*w_t_norm)
    if est_epsilon == 0:
        est_epsilon = 1e-4

    return est_epsilon,norm_w_w

def calculate_max_norm(data):
    max_norm = 0
    for d in data:
        norm = np.linalg.norm(d)
        if norm > max_norm:
            max_norm = norm
    return max_norm*max_norm

def remove_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = [x for x in data if x >= lower_bound and x <= upper_bound]
    return filtered_data

def preprocess_data_shift(X_strat, X, strat_features, n):
    if strat_features is None:
        strat_features = np.arange(X.shape[1])
    X_temp = np.copy(X_strat)
    X_combined = np.concatenate((X_temp, X), axis=0)
    X_subset = X_combined[:, strat_features]
    X_subset_scaled = preprocessing.scale(X_subset)
    X_combined[:, strat_features] = X_subset_scaled
    X_temp = X_combined[:n, :]
    return X_temp

def plot_step(i, offset, start_list, end_list, method_name, colors,markers,num_iters,c):
    if i == 1:
        plt.plot([i, i+offset], [start_list[c,i], end_list[c,i]],color=colors, marker=markers, linestyle='-', label=method_name)
    else:
        plt.plot([i, i+offset], [start_list[c,i], end_list[c,i]], color=colors, marker=markers, linestyle='-')
    if i < num_iters-1:
        plt.plot([i+offset, i+1], [end_list[c,i], start_list[c,i+1]], 'g:')

def plot_mse(mse_list_start,mse_list_start_std,colors,markers,linestyles,method_name,std=1):
    plt.plot(range(len(mse_list_start)),mse_list_start,color=colors, marker=markers, linestyle=linestyles,label=method_name, linewidth=1, markersize=6)
    if std == 1:
        plt.fill_between(range(len(mse_list_start)), mse_list_start - mse_list_start_std, mse_list_start + mse_list_start_std, color= colors, alpha=0.2, linewidth=0)

def plot_model_gap(model_gaps_avg,model_gaps_std,colors,markers,linestyles,method_name,std=1):
    plt.plot(range(len(model_gaps_avg)),model_gaps_avg,color=colors, marker=markers, linestyle=linestyles,label=method_name, linewidth=1, markersize=6)
    if std == 1:
        plt.fill_between(range(len(model_gaps_avg)), model_gaps_avg - model_gaps_std, model_gaps_avg + model_gaps_std, alpha=0.2,color=colors, linewidth=0)

class CustomLogisticRegression(LogisticRegression):
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, 
                 intercept_scaling=1, class_weight=None, random_state=None, 
                 solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, 
                 warm_start=False, n_jobs=None, l1_ratio=None):
        super(CustomLogisticRegression, self).__init__(
            penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, 
            intercept_scaling=intercept_scaling, class_weight=class_weight, 
            random_state=random_state, solver=solver, max_iter=max_iter, 
            multi_class=multi_class, verbose=verbose, warm_start=warm_start, 
            n_jobs=n_jobs, l1_ratio=l1_ratio)
        
    def fit(self, X, y):
        super(CustomLogisticRegression, self).fit(X, y)
        self.w = self.coef_.ravel()

    def predict(self, X):
        w = self.coef_.ravel()
        b = self.intercept_
        score = np.dot(w,X.T) + b
        # score = self.sigmoid(z)
        # proba = self.predict_proba(X)[:,1]
        predictions = super(CustomLogisticRegression, self).predict(X)
        return score, predictions
    
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))