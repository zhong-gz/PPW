import numpy as np
import random
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# D(w) = X - mu * w
def data_distribution_map1(X,y, mu = 0, model = None, strat_features = None):
    if model is not None:
        # normal_noise_x = np.random.normal(0, 0.1, size=model.coef_.shape)
        # normal_noise_y = np.random.normal(0, 0.1, size=y.shape)
        # X_strat = X + mu * model.coef_ + normal_noise_x
        # y_strat = y + mu * X @ (model.coef_.T**2) + normal_noise_y

        # uniform_noise_x = np.random.uniform(low=-0.5, high=0.5, size=X.shape)
        # uniform_noise_y = np.random.uniform(low=-0.5, high=0.5, size=y.shape)
        # X_strat = X + mu * model.coef_ + uniform_noise_x
        # y_strat = y + mu * X @ (model.coef_.T**2) + uniform_noise_y

        laplace_noise_x = np.random.laplace(loc=0, scale=0.1, size=X.shape)
        laplace_noise_y = np.random.laplace(loc=0, scale=0.1, size=y.shape)
        X_strat = X + mu * model.coef_ + laplace_noise_x
        y_strat = y + mu * X @ (model.coef_.T**2) + laplace_noise_y

        if np.any(y_strat > 2):
            # y_strat = np.clip(y_strat, -10, 10)
            scaler = MinMaxScaler()
            y_strat = scaler.fit_transform(y_strat.reshape(-1, 1))*2
    else:
        X_strat = np.copy(X)
        y_strat = np.copy(y)
    return X_strat,y_strat

# D(w) in crime rate prediction
def data_distribution_map2(X,y, mu = 0, model = None,n=None):

    scaler = MinMaxScaler()
    y_strat = np.zeros_like(y)
    hat_y = model.predict(X)
    mean_hat_y = np.mean(hat_y)
    y_strat = y - mu*(hat_y-mean_hat_y) + np.random.normal(0, 0.1, size=y.shape)
    if np.any(y_strat > 2):
        # y_strat = np.clip(y_strat, -10, 10)
        scaler = MinMaxScaler()
        y_strat = scaler.fit_transform(y_strat.reshape(-1, 1))*2
    if n is not None:
        random_indices = np.random.choice(X.shape[0], size=n, replace=False)
        X = X[random_indices, :]
        y_strat = y_strat[random_indices, :]
    return X,y_strat

def data_distribution_map3(X,y, mu = 0, model = None, strat_features = None):
    if model is not None:
        hat_y = model.predict(X)
        max_hat_y = np.max(hat_y)
        X_strat = X.copy()
        std = 0.1 #0.1,0.2,0.3,0.4,0.5 change the noise level here
        X_strat[:, 1] = X[:, 1] + mu * (max_hat_y-hat_y).reshape(-1)**2  + np.random.normal(0, std, size=hat_y.shape).reshape(-1)
        y_strat = y + mu * (max_hat_y-hat_y)**0.5 + np.random.normal(0, std, size=y.shape)
        # scaler = MinMaxScaler()
        # y_strat = scaler.fit_transform(y_strat)*2
        # X_strat = scaler.fit_transform(X_strat)*2
    else:
        X_strat = X.copy()
        y_strat = y.copy()
    return X_strat,y_strat

def data_distribution_map4(X,y, mu = 0, model = None, strat_features = None):
    if model is not None:
        hat_y = model.predict(X)
        mean_hat_y = np.mean(hat_y)
        rev = hat_y.flatten()*X[:, 3]
        avg_rev = np.mean(rev)
        X_strat = X.copy()
        X_strat[:, 0] = X[:, 0] + mu * (hat_y-mean_hat_y).reshape(-1)  + np.random.normal(0, 0.1, size=rev.shape).reshape(-1)
        X_strat[:, 3] = X[:, 3] + mu * (hat_y-mean_hat_y).reshape(-1)  + np.random.normal(0, 0.1, size=rev.shape).reshape(-1)
        y_strat = y - mu * (hat_y-mean_hat_y).reshape(-1,1) + np.random.normal(0, 0.1, size=rev.shape).reshape(-1,1)
        if np.any(y_strat > 2):
            scaler = MinMaxScaler()
            y_strat = scaler.fit_transform(y_strat.reshape(-1, 1))*2
    else:
        X_strat = X.copy()
        y_strat = y.copy()
    return X_strat,y_strat

def data_distribution_map5(X,y, mu = 0, model = None, strat_features = None):
    if model is not None:
        hat_y = model.predict(X)
        mean_hat_y = np.mean(hat_y)
        rev = hat_y.flatten()*X[:, 3]
        X_strat = X.copy()
        X_strat[:, 9] = X[:, 9] - mu * (hat_y-mean_hat_y).reshape(-1)  + np.random.normal(0, 0.1, size=rev.shape).reshape(-1)
        X_strat[:, 11] = X[:, 11] + mu * (hat_y-mean_hat_y).reshape(-1)  + np.random.normal(0, 0.1, size=rev.shape).reshape(-1)
        y_strat = y - mu * (hat_y-mean_hat_y).reshape(-1,1) + np.random.normal(0, 0.1, size=rev.shape).reshape(-1,1)
        # if np.any(y_strat > 0.9):
        #     scaler = MinMaxScaler()
        #     y_strat = scaler.fit_transform(y_strat.reshape(-1, 1))*0.9
    else:
        X_strat = X.copy()
        y_strat = y.copy()
    return X_strat,y_strat

def data_distribution_map7(X,y, mu = 0, model = None, strat_features = None):
    if model is not None:
        hat_y = model.predict(X)
        mean_hat_y = np.mean(hat_y)
        X_strat = X.copy()
        X_strat[:, 0] = X[:, 0] + mu * ((hat_y-mean_hat_y).reshape(-1,1).flatten())  + np.random.normal(0, 0.1, size=hat_y.shape).reshape(-1)
        X_strat[:, 1] = X[:, 1] + mu * ((hat_y-mean_hat_y).reshape(-1,1).flatten())  + np.random.normal(0, 0.1, size=hat_y.shape).reshape(-1)
        y_strat = y - mu * ((hat_y-mean_hat_y).reshape(-1,1)) + np.random.normal(0, 0.1, size=hat_y.shape)
        if np.any(y_strat > 1.1):
            scaler = MinMaxScaler()
            y_strat = scaler.fit_transform(y_strat.reshape(-1, 1))*1.1
    else:
        X_strat = X.copy()
        y_strat = y.copy()
    return X_strat,y_strat

def linear_data_generation(n = 100,n_features = 20):
    X = np.random.rand(n, n_features)
    if n_features == 20:
        true_coefficients = [0.8, 0.5, 0.5, 0.3, 0.6, 0.3, 0.6, 0.2, 0.2, 0.4, 0.4, 0.3, 0.1, 0.0, 0.0, 0.3, 0.5, 0.7, 0.5, 0.9]
    elif n_features == 100:
        true_coefficients = [0.38867729, 0.27134903, 0.82873751, 0.35675333, 0.28093451,
       0.54269608, 0.14092422, 0.80219698, 0.07455064, 0.98688694,
       0.77224477, 0.19871568, 0.00552212, 0.81546143, 0.70685734,
       0.72900717, 0.77127035, 0.07404465, 0.35846573, 0.11586906,
       0.86310343, 0.62329813, 0.33089802, 0.06355835, 0.31098232,
       0.32518332, 0.72960618, 0.63755747, 0.88721274, 0.47221493,
       0.11959425, 0.71324479, 0.76078505, 0.5612772 , 0.77096718,
       0.4937956 , 0.52273283, 0.42754102, 0.02541913, 0.10789143,
       0.03142919, 0.63641041, 0.31435598, 0.50857069, 0.90756647,
       0.24929223, 0.41038292, 0.75555114, 0.22879817, 0.07697991,
       0.28975145, 0.16122129, 0.92969765, 0.80812038, 0.63340376,
       0.87146059, 0.80367208, 0.18657006, 0.892559  , 0.53934224,
       0.80744016, 0.8960913 , 0.31800347, 0.11005192, 0.22793516,
       0.42710779, 0.81801477, 0.86073058, 0.00695213, 0.5107473 ,
       0.417411  , 0.22210781, 0.11986537, 0.33761517, 0.9429097 ,
       0.32320293, 0.51879062, 0.70301896, 0.3636296 , 0.97178208,
       0.96244729, 0.2517823 , 0.49724851, 0.30087831, 0.28484049,
       0.03688695, 0.60956433, 0.50267902, 0.05147875, 0.27864646,
       0.90826589, 0.23956189, 0.14489487, 0.48945276, 0.98565045,
       0.24205527, 0.67213555, 0.76161962, 0.23763754, 0.72821635]
    else:
        true_coefficients = np.random.rand(n_features)
    y = X @ true_coefficients + np.random.randn(n) * 0.1
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.reshape(-1, 1))
    # scaler = MinMaxScaler()
    # y = scaler.fit_transform(y.reshape(-1, 1))
    return X,y

def linear_data_generation2(n = 100,n_features = 20,true_coefficients=[0.8, 0.5]):
    X = np.random.rand(n, n_features)
    y = X @ true_coefficients + np.random.randn(n) * 0.1
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.reshape(-1, 1))
    # scaler = MinMaxScaler()
    # y = scaler.fit_transform(y.reshape(-1, 1))
    return X,y

def linear_data_generation_sensitive(n = 100,n_features = 20,true_coefficients = None):
    X = np.random.rand(n, n_features)
    y = X @ true_coefficients + np.random.randn(n) * 0.1
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.reshape(-1, 1))
    return X,y

def est_varepsilon(X,y,X_new,y_new,w_arr,norm_w_w):
    ridge_model = w_arr[-1]
    
    # gradient of x
    n = X.shape[0]
    y_pred = ridge_model.predict(X)
    gradient = - (X.T.dot(y - y_pred)) # + 2*ridge_model.alpha * ridge_model.coef_.T
    mean_value = gradient # /n

    # gradient of x_new
    n_new = X_new.shape[0]
    y_pred_new = ridge_model.predict(X_new)
    gradient_new = - (X_new.T.dot(y_new - y_pred_new)) # + 2*ridge_model.alpha * ridge_model.coef_.T
    mean_value_new = gradient_new #/n_new

    if len(w_arr) == 2:
        norm_w_w.append(np.linalg.norm(w_arr[-1].coef_))
    else:
        norm_w_w.append(np.linalg.norm(w_arr[-1].coef_-w_arr[-2].coef_))

    if norm_w_w[-1] == 0:
        norm_w_w[-1] = 1e-4

    est_epsilon = np.linalg.norm(mean_value-mean_value_new)/(norm_w_w[-1])
    if est_epsilon < 1e-4:
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

def plot_step(i, offset, start_list, end_list, method_name, colors,markers,num_iters,c,linewidth = 1):
    if i == 1:
        plt.plot([i, i+offset], [start_list[c,i], end_list[c,i]],color=colors, marker=markers, linestyle='-', label=method_name)
    else:
        plt.plot([i, i+offset], [start_list[c,i], end_list[c,i]], color=colors, marker=markers, linestyle='-')
    if i < num_iters-1:
        plt.plot([i+offset, i+1], [end_list[c,i], start_list[c,i+1]], 'g:')

def plot_mse(mse_list_start,mse_list_start_std,colors,markers,linestyles,method_name,std=1,linewidth = 1):
    if method_name == 'PPW-AVG':
        plt.plot(range(len(mse_list_start)),mse_list_start,color=colors, marker=markers, linestyle=linestyles,label=method_name, linewidth=linewidth, markersize=9,alpha=1)
    else:
        plt.plot(range(len(mse_list_start)),mse_list_start,color=colors, marker=markers, linestyle=linestyles,label=method_name, linewidth=linewidth, markersize=6,alpha=1)
    if std == 1:
        plt.fill_between(range(len(mse_list_start)), mse_list_start - mse_list_start_std, mse_list_start + mse_list_start_std, color= colors, alpha=0.2, linewidth=0)

def plot_model_gap(model_gaps_avg,model_gaps_std,colors,markers,linestyles,method_name,std=1,linewidth = 1):
    if method_name == 'PPW-AVG':
        plt.plot(range(len(model_gaps_avg)),model_gaps_avg,color=colors, marker=markers, linestyle=linestyles,label=method_name, linewidth=linewidth, markersize=9,alpha=1)
    else:
        plt.plot(range(len(model_gaps_avg)),model_gaps_avg,color=colors, marker=markers, linestyle=linestyles,label=method_name, linewidth=linewidth, markersize=6,alpha=1)
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
    