import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import time
from svm_kernel import SVMKernel
from os import path, listdir
import re


class SVMWriting(SVMKernel):

    def __init__(self, C, toler, maxIter, ktup, first=False):
        self.first = first
        self.ktup = ktup
        self.C = C
        self.toler = toler
        self.maxIter = maxIter

    def file2vector(self, filename):
        """将数字矩阵转换为一维向量"""
        f = open(filename)
        vector = list() # 创建一个空矩阵
        classLabel = re.findall('(\d+)_\d+\.txt', filename)[0]
        classLabel = -1 if int(classLabel) == 9 else 1
        for l in f:
            line = l.strip()
            for i in line:
                vector.append(int(i))
        vector = np.array(vector)
        return vector, classLabel

    def load_data(self, filepath):
        files = listdir(filepath)
        datamat = np.empty((0, 1024))
        labellist = list()
        for file in files:
            filename = path.join(filepath, file)
            vector, classLable = self.file2vector(filename)
            datamat = np.row_stack([datamat, vector]) # 一定不要忘了重新赋值
            labellist.append(classLable)
        labellist = np.array(labellist)
        return datamat, labellist

    def save_surport_data(self):
        """支持向量及数据存储至本地"""
        nonzero_index = np.nonzero(self.alpha.A > 0)[0]
        print("there are {} support vectors".format(len(nonzero_index)))
        nonzero_data = self.data_mat[nonzero_index]
        nonzero_label = self.label_mat[nonzero_index]
        nonzero_alpha = self.alpha[nonzero_index]
        np.savetxt("model/nonzero_data_writing.txt", nonzero_data)
        np.savetxt("model/nonzero_label_writing.txt", nonzero_label)
        np.savetxt("model/nonzero_alpha_writing.txt", nonzero_alpha)
        np.savetxt("model/b_writing.txt", self.b)
        return nonzero_data, nonzero_label, nonzero_alpha, self.b

    def test_rbf(self):

        self.dataset, self.label_set = self.load_data('data/trainingDigits')
        self.data_mat = np.mat(self.dataset)
        self.label_mat = np.mat(self.label_set).transpose()
        if self.first:
            print("train model")
            self.data_init()
            b, alpha = self.train_svm()
            nonzero_data, nonzero_label, nonzero_alpha, b = self.save_surport_data()
        else:
            print("load model from local file")
            nonzero_data = np.loadtxt("model/nonzero_data_writing.txt")
            nonzero_label = np.loadtxt("model/nonzero_label_writing.txt")
            nonzero_alpha = np.loadtxt("model/nonzero_alpha_writing.txt")
            b = np.loadtxt("model/b_writing.txt")

        m = self.data_mat.shape[0]
        error_count = 0

        for j in range(m):
            K = self.kernel_trans(nonzero_data, self.data_mat[j,:], self.ktup)
            Fj = np.multiply(nonzero_alpha, nonzero_label).T * K + b
            Lj = self.label_mat[j, 0]
            if np.sign(Fj) != np.sign(Lj):
                error_count += 1
        print("train_set: error_count: {}, error_rate: {}".format(error_count, error_count/m))

        error_count = 0
        test_set, test_label = self.load_data('data/testDigits')
        test_mat = np.mat(test_set)
        label_mat = np.mat(test_label).transpose()
        n = test_mat.shape[0]
        for j in range(n):
            K = self.kernel_trans(nonzero_data, test_mat[j, :], self.ktup)
            # 注意！！！F = (alpha_i * li * xi)*x, i=1-m, alpha_i, li, xi均来自训练集！！！只有x来自测试集！！！
            Fj = np.multiply(nonzero_alpha, nonzero_label).T * K + b
            Lj = label_mat[j, 0]
            if np.sign(Fj) != np.sign(Lj):
                error_count += 1
        print("test_set: error_count: {}, error_rate: {}".format(error_count, error_count/n))

    def showClassifer(self, alpha, b):
        # 绘制样本点
        data_plus = []  # 正样本
        data_minus = []  # 负样本

        for i in range(len(self.dataset)):
            if self.label_set[i] > 0:
                data_plus.append(self.dataset[i, :])
            else:
                data_minus.append(self.dataset[i, :])
        data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
        data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
        plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)  # 正样本散点图
        plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)  # 负样本散点图
        # 绘制直线
        w = self.alpha2w(alpha)
        x1 = np.max(self.dataset, 0)[0] # x的最小值
        x2 = np.min(self.dataset, 0)[0] # x的最大值
        a1, a2 = w[0,0], w[0,1]
        b = float(b)
        a1 = float(a1)
        a2 = float(a2)
        y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2 # 由w1*x + w2*y = 0,得y与x的关系式

        plt.plot([x1, x2], [y1, y2])
        # 找出支持向量点
        for i, alpha in enumerate(alpha):
            if alpha > 0:
                x, y = self.dataset[i]
                plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
        plt.show()

if __name__ == '__main__':
    time1 = time.time()
    svm = SVMWriting(200, 0.0001, 10, ('rbf', 10),first=False)
    # b, alpha = svm.train_svm()
    # print(b, alpha)
    svm.test_rbf()
    # w = svm.alpha2w(alpha)
    # print(w)
    # svm.showClassifer(alpha, b)
    time2 = time.time()
    print("total time is {}".format(time2-time1))




