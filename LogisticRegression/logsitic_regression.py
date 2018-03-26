import numpy as np
from numpy import mat, exp
import matplotlib.pyplot as plt
import random

class LR():

    def __init__(self):
        self.data_mat, self.label_mat = self.load_dataset()

    def load_dataset(self):
        data_mat = list()
        label_mat = list()
        fr = open('/home/liuchao/Documents/Machine_Learning/machinelearninginaction/Ch05/testSet.txt')
        for line in fr:
            line_array = line.strip().split()
            data_mat.append([1.0, float(line_array[0]), float(line_array[1])])
            label_mat.append(int(line_array[2]))
        return np.array(data_mat), np.array(label_mat)

    def sigmoid(self, X):
        return 1.0/(1+exp(-X))

    def grad_ascent(self):
        data_matrix = mat(self.data_mat)
        label_mat = mat(self.label_mat).transpose() # 转置，只有一列的矩阵，表示样本类别实际值
        m,n = np.shape(data_matrix)
        alpha, maxCycles = 0.001, 500
        weights = np.ones((n,1)) # 只有一列的矩阵，表示要求的权值
        for k in range(maxCycles):
            h = self.sigmoid(data_matrix*weights) # 只有一列的矩阵，表示每个样本类别的预测值
            error = label_mat - h
            weights = weights + alpha * data_matrix.transpose()*error
        return weights # matrix

    def rand_grad_ascent(self):
        data_matrix = mat(self.data_mat)
        label_mat = mat(self.label_mat).transpose() # 转置，只有一列的矩阵，表示样本类别实际值
        m,n = np.shape(data_matrix)
        maxCycles = 500
        weights = np.ones((n,1)) # 只有一列的矩阵，表示要求的权值
        for j in range(maxCycles):
            for i in range(m):
                alpha = 4/(1+i+j) + 0.01 # 随着次数的增加，alpha逐渐变小
                rand_index = int(random.uniform(0,np.shape(data_matrix)[0]))
                h = self.sigmoid(data_matrix[rand_index]*weights) # 只有一列的矩阵，表示每个样本类别的预测值
                error = label_mat[rand_index] - h
                weights = weights + alpha * data_matrix[rand_index].transpose()*error
                # data_matrix = np.delete(data_matrix, rand_index, 0)
        return weights # matrix

    def plot_best_fix(self, weights):
        data_mat, label_mat = self.load_dataset()
        data_array = np.array(data_mat)
        n = np.shape(data_array)[0] # 样本个数
        xcord1, ycord1, xcord0, ycord0 = [], [], [], []
        for i in range(n):
            if int(label_mat[i]) == 1:
                xcord1.append(data_mat[i,1])
                ycord1.append(data_mat[i,2])
            else:
                xcord0.append(data_mat[i, 1])
                ycord0.append(data_mat[i, 2])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
        ax.scatter(xcord0, ycord0, s=30, c='green')

        x = np.arange(-3.0, 3.0, 0.1) # array
        # weights = np.array(weights) weights is a matrix
        y = (-weights[0,0]-weights[1,0]*x)/weights[2,0] # array w0*x0+w1*x1+w2*x2 = 0 x0=1
        ax.plot(x,y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()

if __name__ == '__main__':
    lr = LR()
    weights = lr.rand_grad_ascent()
    print(weights)
    lr.plot_best_fix(weights)






