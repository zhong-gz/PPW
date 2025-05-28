import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import sem
from collections import deque
import time
import csv
import sys

# Problem instance constants.
sigma = 1.
mu = 1.666666666666667
error_var = 2.033333333333333
a0 = 1.666666666666667
a1 = 1.666666666666667
reg = 3.3333333333333335


loss_reg = reg
fit_reg = 0

c = mu ** 2 + sigma ** 2
opt_theta = c * a0 * (1 - a1) / (c * (1 - a1) ** 2 + loss_reg)
fixed_theta = c * a0 / (c * (1 - a1) + loss_reg)

def clip_coord(z):
    R = 0.8
    if z > R:
        return R
    elif z < -R:
        return -R
    else:
        return z

def clip(theta):
    return np.array([clip_coord(z) for z in theta])

def beta(theta):
    return a0 + a1 * theta

def loss(x, y, theta, reg = loss_reg):
    """
    Squared loss on (x, y) with params theta.
    x should have a bias term appended in the 0-th coordinate.
    """
    return 0.5 * ((theta * x - y) ** 2 + reg * theta ** 2)

def performative_loss(theta, reg = loss_reg):
    return (((theta - beta(theta)) ** 2) * (mu ** 2 + sigma ** 2) + (error_var ** 2) + reg * (theta ** 2)) / 2

opt_loss = performative_loss(opt_theta)
fixed_loss = performative_loss(fixed_theta)

def grad1(X, Y, theta, reg=loss_reg):
    # 计算所有样本的预测误差
    error = np.dot(X, theta) - Y.reshape(-1)
    
    # 计算梯度的第一部分（数据项）
    data_grad = np.mean(X * error[:, np.newaxis], axis=0)
    
    # 计算梯度的第二部分（正则化项）
    reg_grad = reg * theta
    
    # 返回总梯度
    return data_grad + reg_grad

def approx_beta(X, Y, reg = fit_reg):
    mean_xy = np.mean(X * Y)
    mean_xx = np.mean(X ** 2)
    return mean_xy / (mean_xx + reg)

def approx_grad_beta(betas, thetas):
    dbetas = np.array([b - betas[-1] for b in betas])
    dthetas = np.array([t - thetas[-1] for t in thetas], dtype = float)
    
    return np.linalg.pinv(dthetas) @ dbetas
def grad2(X, Y, betas, thetas):
    """
    X, Y should be the data resulting from thetas[-1]
    """
    theta = thetas[-1]
    grad_beta = approx_grad_beta(betas, thetas)
    
    # 向量化计算预测误差
    error = np.dot(X, theta) - Y.reshape(-1)
    
    # 向量化计算梯度的第二部分
    data_grad = np.mean(X * error[:, np.newaxis], axis=0)
    
    # 计算最终梯度
    return -grad_beta * data_grad, grad_beta

