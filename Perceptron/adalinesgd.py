import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import seed
from matplotlib.colors import ListedColormap

class AdalineGD():

    """随机梯度下降"""

    def __init__(self, eta=0.01, n_iter=15, shuffle=True, random_state=1):
        """
        :param eta: 学习速率
        :param n_iter: 训练轮次
        """
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:  # 有什么用
            seed(random_state)

    def load_data(self):
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
        y = df.iloc[0:100, 4].values
        y = np.where(y == 'Iris-setosa', -1, 1)
        X = df.iloc[0:100, [0,2]].values
        return X, y

    def fit(self, X, y):
        """
        训练n_iter轮，每轮在所有数据集上进行权重更新
        :param X:
        :param y:
        :return:
        """
        self.w = np.zeros(1 + np.shape(X)[1])
        self.cost = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)  # 每次迭代打乱数据集顺序
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost.append(avg_cost)  # 记录每轮迭代后误分类点个数的变化
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))  # 产生长度为len(y)的随机array, eg len(y)=3 -> [0,2,1]
        return X[r], y[r]

    def _update_weights(self, x, target):
        out_put = self.net_input(x)
        error = target - out_put
        self.w[1:] += self.eta * x.dot(error)
        self.w[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def partial_fit(self, X, y):
        """fit training data without reinitializing the weights"""
        for xi, target in zip(X ,y):
            self._update_weights(xi, target)



    def net_input(self, X):

        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):

        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def plot_result(self, X, y):

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
        ada1 = self.fit(X, y)
        w = ada1.w
        xmin = X[:,0].min() - 1
        xmax = X[:,0].max() + 1
        ymin = (-w[0] - w[1]*xmin)/w[2]
        ymax = (-w[0] - w[1]*xmax)/w[2]
        ax[0].plot([xmin, xmax], [ymin, ymax])
        ax[0].scatter(X[:50, 0], X[:50, 1], marker='+', label='setosa')
        ax[0].scatter(X[50:100, 0], X[50:100, 1], marker='o', label='versicolor')
        ax[0].set_xlabel('petal length')
        ax[0].set_ylabel('sepal length')
        ax[0].set_title('Adaline - random grad ascent')
        ax[0].legend(loc='upper left')

        ax[1].plot(range(1, len(ada1.cost)+1), ada1.cost, marker='o')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('average-error')
        ax[1].set_title('Adaline - learning rate 0.01')
        plt.show()

if __name__ == '__main__':
    ada = AdalineGD()
    X, y = ada.load_data()
    # ada.plot_result(X, y)
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] -X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] -X[:,1].mean()) / X[:,1].std()
    ada.plot_result(X_std, y) # 数据标准化后可加速收敛并得到较好的结果


from sklearn.linear_model import Perceptron