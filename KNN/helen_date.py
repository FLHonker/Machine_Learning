import numpy as np
from os import path
import matplotlib
import matplotlib.pyplot as plt

filepath = path.dirname(path.dirname(__file__))
file_path = path.join(filepath, 'machinelearninginaction/Ch02/datingTestSet2.txt')
# print(file_path)

class HelenDate():

    def __init__(self):
        self.LIKE_MAP = {
            1: "不喜欢的人",
            2: "魅力一般的人",
            3: "极具魅力的人"
        }

    def file2matrix(self, filename):
        f = open(filename)
        returnMat = np.empty((0,3)) # 创建一个空矩阵
        classLabelvector = list()
        for l in f:
            line = l.strip()
            listFromline = line.split('\t')
            returnMat = np.row_stack([returnMat,
                                      [float(i) for i in listFromline[0:3]]]) #添加行；column_stack添加列
            classLabelvector.append(int(listFromline[-1]))
        return returnMat, classLabelvector

    def showFigure(self, returnMat, labels):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(returnMat[:,1], returnMat[:,2],
                   15.0*(np.array(labels)), 15.0*(np.array(labels)))
        plt.show()

    def autoNorm(self, dataSet):
        minvals = dataSet.min(0) # 列向量最小值组成的矩阵
        maxvals = dataSet.max(0)
        ranges = maxvals - minvals
        # normdataset = np.zeros(np.shape(dataSet))
        m = dataSet.shape[0] # 行向量个数
        normdataset = dataSet - np.tile(minvals, (m,1))
        normdataset = normdataset/np.tile(ranges, (m,1))
        return normdataset, ranges, minvals

    def classify0(self, inX, dataSet, labels, k):
        dataSetSize = dataSet.shape[0] # 求矩阵size(元素个数)
        diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet # 将inX复制dataSetSize份后求差
        sqDiffMat = diffMat**2 # 求矩阵各元素平方
        sqDistances = sqDiffMat.sum(axis=1) # axis=1 求各行向量和，axis=0 求各列向量和
        distances = sqDistances**0.5
        sortedDistIndicies = distances.argsort() # 得到排序eg:[0,1,3,2]
        classCount = dict()
        for i in range(k):
            voteIlable = labels[sortedDistIndicies[i]] # 按距离从小到大得到各标签个数
            classCount[voteIlable] = classCount.get(voteIlable,0) + 1
        sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True) # 按标签从大到小排列
        return sortedClassCount[0][0]

    def datingClassTest(self, k):
        hoRotio = 0.1
        datamat, datalab = self.file2matrix(file_path)
        normat, ranges, minvals = self.autoNorm(datamat)
        m = normat.shape[0]
        testnum = int(hoRotio*normat.shape[0])
        errorcount = 0

        for i in range(testnum):
            classifyresult = self.classify0(normat[i, :], normat[testnum:m, :], datalab[testnum: m], k)
            print("the classify result of sample {} is {}, the real label is {}".
                  format(i, classifyresult, datalab[i]))
            if classifyresult != datalab[i]:
                errorcount += 1
        print("the total error rate is {}".format(errorcount/testnum))

    def predict(self, inx, k):
        datamat, datalab = self.file2matrix(file_path)
        normat, ranges, minvals = self.autoNorm(datamat) # 先归一化
        predict_result = self.classify0((inx - minvals)/ranges, normat, datalab, k) # 预测数据也要归一化
        predict_result = self.LIKE_MAP[predict_result]
        print("the predict result of {} is {}".format(inx, predict_result))

if __name__ == '__main__':
    hd = HelenDate()
    # mat, label = hd.file2matrix(file_path)
    # hd.showFigure(mat, label)
    # print(hd.autoNorm(mat))
    # hd.datingClassTest(5)
    hd.predict(np.array([40920, 8.326976, 0.953952]), 3)