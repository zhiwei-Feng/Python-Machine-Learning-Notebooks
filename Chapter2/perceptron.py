import numpy as np
from numpy.random import seed


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


class AdalineSGD(object):
    '''Adaptive Linear Neuron classifier'''

    def __init__(self, eta=0.0, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        '''拟合训练数据'''
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):
        """不重置权重的拟合训练数据"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        '''shuffle 训练数据'''
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        '''初始权重为0'''
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """使用Adaline学习规则来更新权重"""
        output = self.net_input(xi)
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        """计算净输入"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """计算线性激励函数值"""
        return self.net_input(X)

    def predict(self, X):
        """返回类标"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
