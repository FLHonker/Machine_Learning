import numpy as np
from numpy import log

class AdaBoost():
    # todo: 只适用于连续值分类
    def load_data(self, source):
        data_mat, label_mat, test_mat, test_label = list(), list(), list(), list()
        if source == 'train':
            file = 'horseColicTraining2'
        else:
            file = 'horseColicTest2'
        f_training = open(
            '/home/liuchao/Documents/Machine_Learning/machinelearninginaction/Ch07/{}.txt'.format(file))
        for line in f_training:
            line_array = line.strip().split('\t')
            data_mat.append([float(line_array[i]) for i in range(21)])
            label_mat.append(float(line_array[21]))
        return np.mat(data_mat), np.mat(label_mat).T

    def cal_err_rate(self, data_mat, dim, threshold, compare):
        """
        计算错误率
        :param data_mat: 数据集
        :param dis: 数据权重分布, 行矩阵
        :param dim: 比较的维度
        :param threshold: 分类阈值
        :param compare: 'lt'/'gt'
        :return: 预测标签集
        """
        size = np.shape(data_mat)[0]
        predict_label = np.mat(np.ones((size, 1))) # 初始化为1
        if compare == 'lt':
            # 数组过滤，为true的位置设为-1
            predict_label[data_mat[:, dim] <= threshold] = -1
        else:
            predict_label[data_mat[:, dim] > threshold] = -1

        return predict_label

    def buildstump(self, data_mat, label_mat, dis):
        """
        得到当前轮次最佳决策树
        :param data_mat: 数据集
        :param label_mat: 标签集，列矩阵
        :param dis: 数据权重分布, 行矩阵
        :return: 决策树，误差率，预测标签
        """
        m, n = data_mat.shape
        best_tree, best_err, best_label = dict(), 1, np.mat(np.zeros((m,1)))
        for i in range(n):
            values = sorted(list(set([data_mat[x,i] for x in range(m)])))
            ls = len(values)
            # 分类阈值，取中间值
            steps =[0.5 * values[0]] + [0.5*(values[x] + values[x+1]) for x in range(ls-1)] + [1.5*values[-1]]
            for s in steps:
                for c in ['lt', 'gt']:
                    predict_label = self.cal_err_rate(data_mat, i, s, c)
                    err_mat = np.mat(np.ones((m, 1)))
                    err_mat[predict_label[:, 0] == label_mat[:, 0]] = 0
                    err_rate = dis * err_mat  # matrix[[err]], 是一个矩阵，不能直接与size不同的矩阵相乘，应通过下标[a,b]得到数值
                    if err_rate < best_err:
                        best_tree = {
                            'dim': i,
                            'threshold': s,
                            'compare': c
                        }
                        best_err = err_rate
                        best_label = predict_label
        return best_err[0, 0], best_tree, best_label

    def train_adaboost(self, data_mat, label_mat, turns):
        """
        训练分类器
        :param data_mat: 数据集
        :param label_mat: 标签集
        :param turns: 训练轮数
        :return: 强学习器
        """
        m, n = data_mat.shape
        dis = np.mat(np.array([1/m]*m))
        agg_label_mat = np.mat(np.zeros((m,1))) # 累加后的模型预测标签
        strengthen_tree = list() # 强学习器
        for i in range(turns):
            err, tree, label = self.buildstump(data_mat, label_mat, dis)
            alpha = 0.5 * log((1-err)/max(err, 1e-16))
            tree['alpha'] = alpha
            strengthen_tree.append(tree)
            agg_label_mat += alpha * label #每轮次的预测标签乘以轮次权重
            predict_label = np.sign(agg_label_mat) # 累加后的预测标签
            error_label = np.mat(np.zeros((m, 1))) # 初始化错误标签为0
            error_label[predict_label[:,0] != label_mat[:,0]] = 1
            error_rate = np.sum(error_label, 0)/m # 弱学习器累计后预测错误的样本个数
            if error_rate == 0:
                print('error_rate is 0, break the loop')
                return strengthen_tree
            exp = np.exp(-alpha * np.multiply(label_mat, label)) # 列矩阵
            dis = np.multiply(dis, exp.T)/(dis*exp)
            print("train turn:{}, error rate:{}".format(i, error_rate))
        return strengthen_tree, agg_label_mat

    def classify(self, input_mat, label_mat, strengthen_tree):
        """
        预测分类结果
        :param input_mat: 数据集
        :param strenghten_tree: 强学习器
        :return: 预测结果
        """
        m = np.shape(input_mat)[0]
        agg_label_mat = np.mat(np.zeros((m, 1)))
        for i in range(len(strengthen_tree)):
            predict_label = self.cal_err_rate(
                input_mat, strengthen_tree[i]['dim'],
                strengthen_tree[i]['threshold'],
                strengthen_tree[i]['compare'])
            agg_label_mat += strengthen_tree[i]['alpha'] * predict_label
        predict_label = np.sign(agg_label_mat)  # 累加后的预测标签
        error_label = np.mat(np.zeros((m, 1)))  # 初始化错误标签为0
        error_label[predict_label[:, 0] != label_mat[:, 0]] = 1
        error_rate = np.sum(error_label, 0) / m  # 弱学习器累计后预测错误的样本个数
        return predict_label, error_rate, agg_label_mat

    def plotRoc(self, agg_label_mat, classlabels):
        """
        绘制ROC曲线，X(假正例率：FP/FP+TN)，Y(真正例率：TP/TP+FN)
        :param agg_label_mat: 累加后的标签预测值
        :param classlabels: 真实标签值
        :return: ROC曲线，AUC(ROC面积)
        """
        import matplotlib.pyplot as plt
        cur = (1.0, 1.0) # 初始化
        ysum = 0.0
        numPosClas = np.sum(np.array(classlabels) == 1) # 正例的数目
        ystep = 1/numPosClas # y步长
        xstep = 1/(len(classlabels) - numPosClas) # x步长
        # argsort 返回每个元素的排序
        sorted_label = np.argsort(agg_label_mat, axis=0) # 升序排列，从左往右设置阈值，大于阈值设为正例，小于为反例
        fig = plt.figure()
        fig.clf()
        ax = plt.subplot(111)
        for i in sorted_label.tolist():# 得到排序列表，将排序作为索引得到从小到大排列的标签
            if classlabels[i, 0] == 1: # 正例判为反例，Y - 1/m+
                deltay = ystep
                deltax = 0
            else:
                deltax = xstep # 反例判为反例，假正利率减小， X - 1/m-
                deltay = 0
                ysum += cur[1] # x变化时才会加y
            ax.plot([cur[0], cur[0]-deltax], [cur[1], cur[1]-deltay], c='b')
            cur = (cur[0]-deltax, cur[1]-deltay)
        ax.plot([0,1],[0,1], 'b--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for adaboost')
        ax.axis([0,1,0,1])
        plt.show()
        AUC = xstep * ysum
        print("the area of ROC curve is: {}".format(AUC))


if __name__ == '__main__':
    ada = AdaBoost()
    data_mat, label_mat = ada.load_data('train')
    strengthen_tree, agg_label_mat = ada.train_adaboost(data_mat, label_mat, 100)
    print(strengthen_tree)
    # input_mat, label_mat = ada.load_data('test')
    # print(ada.classify(input_mat, label_mat, strengthen_tree))
    ada.plotRoc(agg_label_mat, label_mat)