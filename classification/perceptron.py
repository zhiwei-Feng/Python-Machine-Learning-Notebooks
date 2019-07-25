import numpy as np


class Perceptron(object):
    """Perceptron classfier"""

    def __init__(self, eta=0.01, n_iter=10):
        '''
        eta : float
            学习率 (0-1)
        n_iter : int
            训练次数
        '''
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        '''拟合训练集
        X : array-like, shape=[n_samples, n_features]
            训练集矩阵
        y : array-like, shape=[n_samples]
            目标值向量
        '''
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def net_input(self, X):
        """计算净输入"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """返回类标"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class AdalineGD(object):
    """Adaptive Linear Neuron classifier"""

    def __init__(self, eta=0.01, n_iter=50):
        '''
        eta : float
            学习率 (0-1)
        n_iter : int
            训练次数
        '''
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        '''拟合训练集
        X : array-like, shape=[n_samples, n_features]
            训练集矩阵
        y : array-like, shape=[n_samples]
            目标值向量
        '''
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        """计算净输入"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """计算线性激励函数值"""
        return self.net_input(X)

    def predict(self, X):
        """返回类标"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
