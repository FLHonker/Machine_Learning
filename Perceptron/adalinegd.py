import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdalineGD():

    """基于线性激励函数的连续型输出进行权重更新"""

    def __init__(self, eta=0.01, n_iter=10):
        """
        :param eta: 学习速率
        :param n_iter: 训练轮次
        """
        self.eta = eta
        self.n_iter = n_iter

    def load_data(self):
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
        y = df.iloc[0:100, 4].values
        y = np.where(y == 'Iris-setosa', -1, 1)
        X = df.iloc[0:100, [0,2]].values
        return X, y

    def fit(self, X, y):
        """
        训练n_iter轮，每轮基于整个数据集进行权重更新
        :param X:
        :param y:
        :return:
        """
        self.w = np.zeros(1 + np.shape(X)[1])
        self.cost = []

        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w[1:] += self.eta * X.T.dot(errors)
            self.w[0] += self.eta *errors.sum()  #  常数项（阈值），相当于x=1
            cost = (errors**2).sum()/2  # 记录每次迭代的误差
            self.cost.append(cost)  # 记录每轮迭代后误分类点个数的变化
        return self

    def net_input(self, X):

        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):

        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def plot_result(self, X, y):

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
        ada1 = self.fit(X, y)
        ax[0].plot(range(1, len(ada1.cost)+1), np.log10(ada1.cost), marker='o')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('log(Sum-squared-error)')
        ax[0].set_title('Adaline - learning rate 0.01')
        self.eta = 0.0001
        ada2 = self.fit(X, y)
        ax[1].plot(range(1, len(ada2.cost) + 1), ada2.cost, marker='o')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Sum-squared-error')
        ax[1].set_title('Adaline - learning rate 0.0001')
        plt.show()

if __name__ == '__main__':
    ada = AdalineGD()
    X, y = ada.load_data()
    ada.plot_result(X, y)
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] -X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] -X[:,1].mean()) / X[:,1].std()
    ada.plot_result(X_std, y) # 数据标准化后可加速收敛并得到较好的结果