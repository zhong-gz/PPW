import numpy as np
from numpy.linalg import lstsq

class two_stage_algo:
    def __init__(self,X_base,y_base,tol = 1e-6):
        self.X_base = np.copy(X_base)
        self.y_base = np.copy(y_base)
        self.y_base[self.y_base != 1] = 0
        self.theta = None
        self.tol = tol
        self.mu_x = None
        self.mu_y = None
        self.X_shift = None # np.zeros_like(X_base)
        self.y_shift = None
        self.theta_list = np.zeros_like(X_base)

    def calculate_performative_effect(self):
        n = len(self.y_base)
        repeat_theta = self.theta_list

        theta_theta_inv = np.linalg.inv(repeat_theta.T @ repeat_theta)
        theta_theta_inv = theta_theta_inv + 0.0001*np.eye(theta_theta_inv.shape[0])

        self.mu_x = theta_theta_inv @ repeat_theta.T @ self.X_shift
        self.mu_y = theta_theta_inv @ repeat_theta.T @ self.y_shift 

    def train(self,X,y_ture):
        y = np.copy(y_ture)
        n = len(y)
        d = self.X_base.shape[1]

        if self.X_shift is None:
            self.X_shift = X
            self.y_shift = y
        else:
            self.X_shift = np.concatenate((self.X_shift, X), axis=0)
            self.y_shift = np.concatenate((self.y_shift, y), axis=0)

        self.calculate_performative_effect()

        theta_init = np.random.randn(d)
        learning_rate = 0.01
        num_iterations = 10000
        theta_final, cost_history = self.gradient_descent(self.X_base, y, self.mu_x,self.mu_y, theta_init, learning_rate, num_iterations)

        self.theta = np.copy(theta_final)
        self.theta_list = np.concatenate((self.theta_list, np.tile(theta_final,(n, 1))), axis=0)
        return theta_final

    def predict(self,X):
        score = np.dot(X, self.theta)
        predictions = score
        return score,predictions

    def compute_cost(self,X, y, theta):
        m = len(y)
        theta_mu_y = np.repeat(theta @ self.mu_y, m, axis=0)
        theta_mu_x = np.repeat(theta @ self.mu_x, m, axis=0)
        cost = np.linalg(y+theta_mu_y - (X + theta_mu_x).T @ theta)
        return cost

    def gradient_descent(self,X, y,mu , theta, learning_rate, num_iterations):
        m = len(y)
        cost_history = []
        mu = np.insert(mu, 0, 0)

        for i in range(num_iterations):
            gradient = np.zeros_like(theta)
            for i in range(m):
                gradient = gradient + y[i]@self.mu_y + theta @self.mu_y@self.mu_y - X[i,:].T @ theta @ self.mu_y + self.mu_x.T@theta.T@theta@self.mu_y\
                                    - y[i] @ X[i,:].T - theta@self.mu_y@X[i,:].T + X[i,:].T @ theta @X[i,:].T - self.mu_x.T@theta.T@theta@X[i,:].T\
                                    + y[i] @self.mu_x.T@theta.T + theta@self.mu_y@self.mu_x.T@theta.T - X[i,:].T @ theta @self.mu_x.T@theta.T + self.mu_x.T@theta.T@theta@self.mu_x.T@theta.T
            gradient = 2 * gradient
            theta -= learning_rate * gradient
            cost = self.compute_cost(X, y, theta)
            cost_history.append(cost)
            if (i > 2) and (np.abs(cost_history[-1] - cost_history[-2]) < 1e-3/m):
                return theta, cost_history

        return theta, cost_history
