import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import time

class SVM_Simple():

    def __init__(self):
        self.dataset, self.label_set = self.load_data()
        self.data_mat = np.mat(self.dataset)
        self.label_mat = np.mat(self.label_set).transpose()
        

    def load_data(self):

        f = open('testSet.txt', 'r')
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

    def train_svm(self, C, toler, maxIter):
        data_mat = self.data_mat
        label_mat = self.label_mat
        m, n = np.shape(data_mat)
        self.alpha = np.mat(np.zeros((m,1))) # 参数初始化, 矩阵
        self.b = 0 # 参数初始化
        iter = 0
        while iter < maxIter:
            alpha_pair_changed = 0
            # 随机选取一个作为alphai
            for i in range(0, m):
                index_list = list(range(m))
                index_list.remove(i)
                # multiply 矩阵逐元素相乘，Fi为第i个样本的分类标签
                # Ei = Fi - Yi; Yi*Ei = YiFi - 1
                Fi = np.multiply(self.alpha, label_mat).T * data_mat * (data_mat[i].T) + self.b
                Ei = Fi - float(label_mat[i])

                # 如果alphai不满足KKT条件则更新
                if ((label_mat[i, 0] * Ei < -toler) and (self.alpha[i, 0] < C)) or \
                        ((label_mat[i, 0] * Ei > toler) and (self.alpha[i, 0] > 0)):
                    j = random.sample(index_list, 1)[0]
                    alpha_old = copy.deepcopy(self.alpha) # 得到权向量的复制集
                    ai_old, aj_old = alpha_old[i,0], alpha_old[j,0]
                    Fj = np.multiply(self.alpha, label_mat).T * data_mat * (data_mat[j].T) + self.b
                    Ej = Fj - label_mat[j, 0]
                    # 根据i,j的标签，得到ai, aj的范围：
                    if label_mat[i, 0] == label_mat[j, 0]:
                        L = max(0, ai_old+aj_old-C)
                        H = min(C, ai_old+aj_old)
                    else:
                        L = max(0, aj_old - ai_old)
                        H = min(C, C + aj_old - ai_old)
                    if L == H:
                        print("L equals H, continue")
                        continue
                    # 更新alpha, b
                    eta = 2 * data_mat[i,:] * data_mat[j,:].T - \
                          data_mat[i,:] * data_mat[i,:].T - data_mat[j,:] * data_mat[j,:].T
                    if eta == 0:
                        continue # why ,不是只要！=0就行吗？
                    self.alpha[j, 0] = aj_old - label_mat[j, 0]*(Ei - Ej)/eta # aj_new
                    self.alpha[j, 0] = self.clip_alpha(self.alpha[j, 0], L, H) # 满足约束条件的aj_new
                    if (abs(self.alpha[j, 0] - aj_old) < 0.00001): # 如果变化太小则跳过
                        print("j not modified enough, continue")
                        continue
                    self.alpha[i, 0] = ai_old + label_mat[i, 0]*label_mat[j, 0]*(aj_old-self.alpha[j, 0])

                    # 更新b值
                    bi = self.b - Ei - label_mat[i,0]*(self.alpha[i,0]-ai_old)*data_mat[i,:]*data_mat[i,:].T-\
                         label_mat[j,0]*(self.alpha[j, 0]-aj_old)*data_mat[i,:]*data_mat[j,:].T
                    bj = self.b - Ej - label_mat[i,0]*(self.alpha[i,0]-ai_old)*data_mat[i,:]*data_mat[j,:].T-\
                         label_mat[j,0]*(self.alpha[j, 0]-aj_old)*data_mat[j,:]*data_mat[j,:].T

                    if 0 < self.alpha[i, 0] < C:
                        self.b = bi
                    elif 0 < self.alpha[j, 0] < C:
                        self.b = bj
                    else:
                        self.b = (bi+bj)/2
                    alpha_pair_changed += 1
                    print("iter:{},alpha_pair_changed:{}".format(iter, alpha_pair_changed))
            if alpha_pair_changed == 0:
                print("no self.alpha changed, continue")
                iter += 1
            else:
                iter = 0
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
    svm = SVM_Simple()
    b, alpha = svm.train_svm(0.6, 0.001, 20)
    print(b, alpha)
    w = svm.alpha2w(alpha)
    print(w)
    svm.showClassifer(alpha, b)
    time2 = time.time()
    print("total time is {}".format(time2-time1))




