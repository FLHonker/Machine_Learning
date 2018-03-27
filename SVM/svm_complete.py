import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import time

class SVMComplete():

    def __init__(self):
        self.dataset, self.label_set = self.load_data()
        self.data_mat = np.mat(self.dataset)
        self.label_mat = np.mat(self.label_set).transpose()
        m, n = np.shape(self.data_mat)
        self.alpha = np.mat(np.zeros((m, 1)))  # 参数初始化, 矩阵
        self.b = 0  # 参数初始化
        self.emap = dict() # 只选择不满足KKT条件的进行更新（即已经更新过的）
        for i in range(m):
            self.emap[i] = 0
        

    def load_data(self):

        f = open('data/testSet.txt', 'r')
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
        """计算第j个样本的误差"""
        Fj = np.multiply(self.alpha, self.label_mat).T * self.data_mat * (self.data_mat[j].T) + self.b
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
            print('j:{}'.format(j))
            Ej = self.calc_Ej(j)
            return j, Ej


    def innerL(self, i, C, toler):
        data_mat = self.data_mat
        label_mat = self.label_mat
        Ei = self.calc_Ej(i)

        if ((label_mat[i, 0] * Ei < -toler) and (self.alpha[i, 0] < C)) or \
                ((label_mat[i, 0] * Ei > toler) and (self.alpha[i, 0] > 0)):
            j, Ej = self.select_j(i, Ei)  # 选取一个不同的作为alphaj
            alpha_old = copy.deepcopy(self.alpha)  # 得到权向量的复制集
            ai_old, aj_old = alpha_old[i, 0], alpha_old[j, 0]
            # Fj = np.multiply(self.alpha, label_mat).T * data_mat * (data_mat[j].T) + self.b
            # Ej = Fj - label_mat[j, 0]
            # 根据i,j的标签，得到ai, aj的范围：
            if label_mat[i, 0] == label_mat[j, 0]:
                L = max(0, ai_old + aj_old - C)
                H = min(C, ai_old + aj_old)
            else:
                L = max(0, aj_old - ai_old)
                H = min(C, C + aj_old - ai_old)
            if L == H:
                print("L equals H, continue")
                return 0
            # 更新alpha, b
            eta = 2 * data_mat[i, :] * data_mat[j, :].T - \
                  data_mat[i, :] * data_mat[i, :].T - data_mat[j, :] * data_mat[j, :].T
            if eta >= 0:
                return 0  # why ,不是只要！=0就行吗？
            self.alpha[j, 0] = aj_old - label_mat[j, 0] * (Ei - Ej) / eta  # aj_new
            self.alpha[j, 0] = self.clip_alpha(self.alpha[j, 0], L, H)  # 满足约束条件的aj_new
            self.emap[j] = 1
            if (abs(self.alpha[j, 0] - aj_old) < 0.00001):  # 如果变化太小则跳过
                print("j not modified enough, continue")
                return 0
            self.alpha[i, 0] = ai_old + label_mat[i, 0] * label_mat[j, 0] * (aj_old - self.alpha[j, 0])
            self.emap[i] = 1

            # 更新b值
            bi = self.b - Ei - label_mat[i, 0] * (self.alpha[i, 0] - ai_old) * data_mat[i, :] * data_mat[i, :].T - \
                 label_mat[j, 0] * (self.alpha[j, 0] - aj_old) * data_mat[i, :] * data_mat[j, :].T
            bj = self.b - Ej - label_mat[i, 0] * (self.alpha[i, 0] - ai_old) * data_mat[i, :] * data_mat[j, :].T - \
                 label_mat[j, 0] * (self.alpha[j, 0] - aj_old) * data_mat[j, :] * data_mat[j, :].T

            if 0 < self.alpha[i, 0] < C:
                self.b = bi
            elif 0 < self.alpha[j, 0] < C:
                self.b = bj
            else:
                self.b = (bi + bj) / 2
            print("modified!")
            return 1
        else:
            self.emap[i] = 1

            return 0

    def train_svm(self, C, toler, maxIter):
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
        while (iter < maxIter) and ((alpha_pair_changed > 0) or (entire_set)):
            alpha_pair_changed = 0
            # 随机选取一个作为alphai
            if entire_set:
                print("modify entire_set")
                for i in range(0, m):
                    alpha_pair_changed += self.innerL(i, C, toler)
                iter += 1
                print("iter", iter)
            else:
                print("modify non_bound")
                nonbound = np.nonzero((self.alpha.A > 0) * (self.alpha.A < C))[0]
                for i in nonbound:
                    alpha_pair_changed += self.innerL(i, C, toler)
                iter += 1
                print("iter", iter)

            if entire_set:
                entire_set = False
            elif alpha_pair_changed == 0:
                entire_set = True

        return self.b, self.alpha

    def alpha2w(self, alpha):
        # 将alpha转换为w
        data_mat = np.mat(self.dataset)
        label_mat = np.mat(self.label_set).transpose()
        w = np.multiply(alpha, label_mat).T * data_mat
        return w

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
    svm = SVMComplete()
    b, alpha = svm.train_svm(0.6, 0.001, 40)
    print(b, alpha)
    w = svm.alpha2w(alpha)
    print(w)
    svm.showClassifer(alpha, b)
    time2 = time.time()
    print("total time is {}".format(time2-time1))




