import numpy as np
from matplotlib.colors import ListedColormap

class Perceptron():
    """基于预测输出的类别标签进行权重更新"""

    def __init__(self, eta=0.01, n_iter=10):
        """
        :param eta: 学习速率
        :param n_iter: 训练轮次
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        训练n_iter轮，每轮基于每个样本进行权重更新
        :param X:
        :param y:
        :return:
        """
        self.w = np.zeros(1 + np.shape(X)[1])
        self.errors = []

        for _ in range(self.n_iter):
            errors_ = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w[1:] += update * xi
                self.w[0] += update  #  常数项（阈值），相当于x=1
                errors_ += int(update != 0.0)  # 记录误分类点的个数
            self.errors.append(errors_)  # 记录每轮迭代后误分类点个数的变化

    def net_input(self, X):

        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):

        return np.where(self.net_input(X) >= 0.0, 1, -1)

def plot_decision_regions(X, y, classifier, resolution=0.02):

    pass



