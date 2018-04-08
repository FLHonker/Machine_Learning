import numpy as np
from numpy import mat, exp
import matplotlib.pyplot as plt
import random
from logsitic_regression import LR

class HorsePrediction(LR):

    def __init__(self):
        self.data_mat, self.label_mat, self.test_mat, self.test_label = self.load_dataset()

    def load_dataset(self):
        data_mat, label_mat, test_mat, test_label = list(), list(), list(), list()
        f_training = open('/home/liuchao/Documents/Machine_Learning/machinelearninginaction/Ch05/horseColicTraining.txt')
        f_test = open('/home/liuchao/Documents/Machine_Learning/machinelearninginaction/Ch05/horseColicTest.txt')
        for line in f_training:
            line_array = line.strip().split('\t')
            data_mat.append([1.0] + [float(line_array[i]) for i in range(1,21)])
            label_mat.append(float(line_array[21]))
        print(data_mat)
        for line in f_test:
            line_array = line.strip().split('\t')
            test_mat.append([1.0] + [float(line_array[i]) for i in range(1,21)])
            test_label.append(float(line_array[21]))
        return np.array(data_mat), np.array(label_mat), np.array(test_mat), np.array(test_label)

    def sigmoid(self, X):
        return 1.0/(1+exp(-X))

    def classify(self, input, weights):
        """
        :param input: matrix(1,n)
        :param weights: matrix(n,1)
        :return: classify result(0/1)
        """
        prob = self.sigmoid(input*weights)
        return 1 if prob > 0.5 else 0

    def test(self):
        weights = self.rand_grad_ascent()
        test_num = self.test_mat.shape[0]
        error_count = 0
        for i in range(test_num):
            if self.classify(mat(self.test_mat[i]), weights) != self.test_label[i]:
                error_count += 1
        print("total error count is: {}, error rate is: {}".format(error_count, error_count/test_num))


    # def plot_best_fix(self, weights):
    #     data_mat, label_mat = self.load_dataset()
    #     data_array = np.array(data_mat)
    #     n = np.shape(data_array)[0] # 样本个数
    #     xcord1, ycord1, xcord0, ycord0 = [], [], [], []
    #     for i in range(n):
    #         if int(label_mat[i]) == 1:
    #             xcord1.append(data_mat[i,1])
    #             ycord1.append(data_mat[i,2])
    #         else:
    #             xcord0.append(data_mat[i, 1])
    #             ycord0.append(data_mat[i, 2])
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    #     ax.scatter(xcord0, ycord0, s=30, c='green')
    #
    #     x = np.arange(-3.0, 3.0, 0.1) # array
    #     # weights = np.array(weights) weights is a matrix
    #     y = (-weights[0,0]-weights[1,0]*x)/weights[2,0] # array w0*x0+w1*x1+w2*x2 = 0 x0=1
    #     ax.plot(x,y)
    #     plt.xlabel('X1')
    #     plt.ylabel('X2')
    #     plt.show()

if __name__ == '__main__':
    hp = HorsePrediction()
    weights = hp.rand_grad_ascent()
    print(weights)
    # lr.plot_best_fix(weights)
    hp.test()






