import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import time

class SVMKernel():

    def __init__(self, C, toler, maxIter, ktup, first=False):
        self.first = first
        self.ktup = ktup
        self.C = C
        self.toler = toler
        self.maxIter = maxIter


    def data_init(self):
        m, n = np.shape(self.data_mat)
        self.alpha = np.mat(np.zeros((m, 1)))  # 参数初始化, 矩阵
        self.b = 0  # 参数初始化
        self.emap = dict()  # 只选择不满足KKT条件的进行更新（即已经更新过的）
        for i in range(m):
            self.emap[i] = 0
        self.K = np.mat(np.zeros((m, m)))
        for i in range(m):
            self.K[:, i] = self.kernel_trans(self.data_mat, self.data_mat[i, :], self.ktup)

    def kernel_trans(self, data_mat, inX, ktup):
        """计算数据集的核矩阵"""
        m = np.shape(data_mat)[0]
        K = np.mat(np.zeros((m,1)))
        if ktup[0] == 'lin': # 线性核函数
            K = data_mat * inX.T
        elif ktup[0] == 'rbf': # 高斯核函数
            for i in range(m):
                K[i] = (data_mat[i, :] - inX) * (data_mat[i, :] - inX).T
            K = np.exp( -K / (ktup[1]**2))
        else:
            raise NameError("unknown kernel function")
        return K


    def load_data(self, file):

        f = open(file, 'r')
        data_set = []
        label_set = []
        # 将数字转换为浮点型，否则会报错
        for l in f:
            line = l.strip().split('\t')
            data_set.append([float(line[0]), float(line[1])])
            label_set.append(float(line[2]))

        return np.array(data_set), np.array(label_set)

    def clip_alpha(self, alpha, L, H):
        if alpha < L:
            return L
        elif alpha > H:
            return H
        else:
            return alpha

    def calc_Ej(self, j):
        """
        计算第j个样本的误差;
        Fj = W * Fai(Xj) + b
        """
        Fj = np.multiply(self.alpha, self.label_mat).T * self.K[:, j] + self.b
        Ej = Fj - self.label_mat[j, 0]
        return Ej

    def select_j(self, i, Ei):
        """选取与Ei差距最大的j"""
        data_mat = self.data_mat
        m,n = np.shape(data_mat)
        self.emap[i] = 1
        index_list = list(range(m))
        index_list.remove(i)
        max_diff, max_index, Ej = 0, 0, 0 # 初始化最大差距为0
        non_zero_index = list(filter(lambda x:x[1]==1, self.emap.items()))
        print(non_zero_index)
        if len(non_zero_index) > 1: # 优先选择不满足KKT条件的，如果没有则随机选取
            for k in [i[0] for i in non_zero_index]:
                if k == i:
                    continue
                Ek = self.calc_Ej(k)
                diff = abs(Ei - Ek)
                if (diff > max_diff):
                    max_diff = diff
                    max_index = k
                    Ej = Ek
            print(max_index, Ej)
            return max_index, Ej
        else:
            j = random.sample(index_list, 1)[0]
            Ej = self.calc_Ej(j)
            return j, Ej


    def innerL(self, i):
        data_mat = self.data_mat
        label_mat = self.label_mat
        Ei = self.calc_Ej(i)

        if ((label_mat[i, 0] * Ei < -self.toler) and (self.alpha[i, 0] < self.C)) or \
                ((label_mat[i, 0] * Ei > self.toler) and (self.alpha[i, 0] > 0)):
            j, Ej = self.select_j(i, Ei)  # 选取一个不同的作为alphaj
            alpha_old = copy.deepcopy(self.alpha)  # 得到权向量的复制集
            ai_old, aj_old = alpha_old[i, 0], alpha_old[j, 0]
            # Fj = np.multiply(self.alpha, label_mat).T * data_mat * (data_mat[j].T) + self.b
            # Ej = Fj - label_mat[j, 0]
            # 根据i,j的标签，得到ai, aj的范围：
            if label_mat[i, 0] == label_mat[j, 0]:
                L = max(0, ai_old + aj_old - self.C)
                H = min(self.C, ai_old + aj_old)
            else:
                L = max(0, aj_old - ai_old)
                H = min(self.C, self.C + aj_old - ai_old)
            if L == H:
                print("L equals H, continue")
                return 0
            # 更新alpha, b
            eta = 2 * self.K[i, j] - self.K[i ,i] - self.K[j ,j]
            if eta >= 0:
                return 0  # why ,不是只要！=0就行吗？
            self.alpha[j, 0] = aj_old - label_mat[j, 0] * (Ei - Ej) / eta  # aj_new
            self.alpha[j, 0] = self.clip_alpha(self.alpha[j, 0], L, H)  # 满足约束条件的aj_new
            if (abs(self.alpha[j, 0] - aj_old) < 0.00001):  # 如果变化太小则跳过
                print("j not modified enough, continue")
                return 0
            self.alpha[i, 0] = ai_old + label_mat[i, 0] * label_mat[j, 0] * (aj_old - self.alpha[j, 0])
            self.emap[i] = 1

            # 更新b值
            bi = self.b - Ei - label_mat[i, 0] * (self.alpha[i, 0] - ai_old) * self.K[i, i] - \
                 label_mat[j, 0] * (self.alpha[j, 0] - aj_old) * self.K[i, j]
            bj = self.b - Ej - label_mat[i, 0] * (self.alpha[i, 0] - ai_old) * self.K[i, j] - \
                 label_mat[j, 0] * (self.alpha[j, 0] - aj_old) * self.K[j ,j]

            if 0 < self.alpha[i, 0] < self.C:
                self.b = bi
            elif 0 < self.alpha[j, 0] < self.C:
                self.b = bj
            else:
                self.b = (bi + bj) / 2
            print("modified!")
            return 1
        else:
            return 0

    def train_svm(self):
        """
        第一次遍历整个数据集
        第二次遍历0<alpha<C的样本，如果没有样本被更新，则重新遍历所有样本
        重复以上步骤
        """
        data_mat = self.data_mat
        m, n = np.shape(data_mat)
        iter = 0
        alpha_pair_changed = 0
        entire_set = True
        while (iter < self.maxIter) and ((alpha_pair_changed > 0) or (entire_set)):
            alpha_pair_changed = 0
            # 随机选取一个作为alphai
            if entire_set:
                print("modify entire_set")
                for i in range(0, m):
                    alpha_pair_changed += self.innerL(i)
                iter += 1
                print("iter", iter)
            else:
                print("modify non_bound")
                nonbound = np.nonzero((self.alpha.A > 0) * (self.alpha.A < self.C))[0]
                for i in nonbound:
                    alpha_pair_changed += self.innerL(i)
                iter += 1
                print("iter", iter)

            if entire_set:
                entire_set = False
            elif alpha_pair_changed == 0:
                entire_set = True

        return self.b, self.alpha

    def save_surport_data(self):
        """支持向量及数据存储至本地"""
        nonzero_index = np.nonzero(self.alpha.A > 0)[0]
        print("there are {} support vectors".format(len(nonzero_index)))
        nonzero_data = self.data_mat[nonzero_index]
        nonzero_label = self.label_mat[nonzero_index]
        nonzero_alpha = self.alpha[nonzero_index]
        np.savetxt("model/nonzero_data.txt", nonzero_data)
        np.savetxt("model/nonzero_label.txt", nonzero_label)
        np.savetxt("model/nonzero_alpha.txt", nonzero_alpha)
        np.savetxt("model/b.txt", self.b)
        return nonzero_data, nonzero_label, nonzero_alpha, self.b


    def alpha2w(self, alpha):
        """将alpha转换为w"""
        data_mat = np.mat(self.dataset)
        label_mat = np.mat(self.label_set).transpose()
        w = np.multiply(alpha, label_mat).T * data_mat
        return w

    def test_rbf(self):

        self.dataset, self.label_set = self.load_data('data/testSetRBF.txt')
        self.data_mat = np.mat(self.dataset)
        self.label_mat = np.mat(self.label_set).transpose()
        if self.first:
            print("train model")
            self.data_init()
            b, alpha = self.train_svm()
            nonzero_data, nonzero_label, nonzero_alpha, b = self.save_surport_data()
        else:
            print("load model from local file")
            nonzero_data = np.loadtxt("model/nonzero_data.txt")
            nonzero_label = np.loadtxt("model/nonzero_label.txt")
            nonzero_alpha = np.loadtxt("model/nonzero_alpha.txt")
            b = np.loadtxt("model/b.txt")

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
        test_set, test_label = self.load_data('data/testSetRBF2.txt')
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
    svm = SVMKernel(20, 0.0001, 100, ('rbf', 1.3),first=False)
    # b, alpha = svm.train_svm()
    # print(b, alpha)
    svm.test_rbf()
    # w = svm.alpha2w(alpha)
    # print(w)
    # svm.showClassifer(alpha, b)
    time2 = time.time()
    print("total time is {}".format(time2-time1))




