import numpy as np
from scipy.stats import sem
from collections import deque
from Algorithm.PerGD_function import shift_dist,approx_f,grad1,clip,grad2,est_performative_loss
from sklearn.linear_model import LogisticRegression, Ridge

class PerGD:
    def __init__(self,H = 50, lr = 0.01):
        self.s1 = 0.5
        self.H  = H
        self.lr = lr # learning rate
        # self.history = deque()
        # self.history.append(self.theta.copy())
        self.grad_fs = deque()
        # self.g2s     = deque()
        self.thetas = deque(maxlen = H + 1)
        self.means  = deque(maxlen = H + 1)

    def train(self,X,y_ture):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        d = X_b.shape[1]
        self.theta = 2 * np.random.rand(d) - 1
        if len(self.thetas) == 0:
            self.thetas.append(self.theta.copy())
        Y = np.copy(y_ture)
        if len(self.thetas) < 2:
            # model = Ridge(alpha = 0, fit_intercept=False)
            # model.fit(X_b, Y)
            # self.theta = model.theta.T #+ np.random.normal(0, 0.05, size=model.theta.T.shape)
            self.theta = np.random.randn(X_b.shape[1], 1) # 随机初始化参数，确保是列向量
        else:
            if len(self.thetas) < 2:
                self.means.append(approx_f(X_b, Y))
                grad = grad1(X_b, Y, self.theta)

                self.theta = clip(self.theta - self.lr * grad).copy()
                # self.theta = (self.theta - self.lr * grad).copy()
                # self.history.append(self.theta.copy())
                self.thetas.append(self.theta.copy())
            else:
                self.means.append(approx_f(X_b, Y))
                g2, grad_f = grad2(X_b, Y, self.means, self.thetas, self.s1)
                grad = grad1(X_b, Y, self.theta) + g2
                self.grad_fs.append(grad_f)
                # self.g2s.append(g2)

                self.theta = clip(self.theta - self.lr * grad).copy()
                # self.theta = (self.theta - self.lr * grad).copy()
                # self.history.append(self.theta.copy())
                self.thetas.append(self.theta.copy())
        self.coef_ = self.theta[1:].T
            
    def predict(self,X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        predictions = np.dot(X_b, self.theta)
        return predictions