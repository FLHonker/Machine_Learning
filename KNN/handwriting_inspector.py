import numpy as np
from os import path, listdir
import re

filepath = path.dirname(path.dirname(__file__))

class HandwritingInspector():
    # todo: 降低数值精度，减小资源消耗
    def file2vector(self, filename):
        """将数字矩阵转换为一维向量"""
        f = open(filename)
        vector = list() # 创建一个空矩阵
        classLabel = re.findall('(\d+)_\d+\.txt', filename)[0]
        for l in f:
            line = l.strip()
            for i in line:
                vector.append(int(i))
        vector = np.array(vector)
        return vector, classLabel

    def file2mat(self, directory):
        file_path = path.join(filepath, 'machinelearninginaction/Ch02/digits/{}'.format(directory))
        files = listdir(file_path)
        datamat = np.empty((0, 1024))
        labellist = list()
        for file in files:
            filename = path.join(file_path, file)
            vector, classLable = self.file2vector(filename)
            datamat = np.row_stack([datamat, vector]) # 一定不要忘了重新赋值
            labellist.append(classLable)
        labellist = np.array(labellist)
        return datamat, labellist

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

    def handwritingclasstest(self, k):
        trainmat, trainlabels = self.file2mat('trainingDigits')
        testmat, testlabels = self.file2mat('testDigits')
        testsize = testmat.shape[0]
        errorcount = 0
        for t in range(testsize):
            testsample = testmat[t, :]
            truelabel = testlabels[t]
            classifyresult = self.classify0(testsample, trainmat, trainlabels, k)
            print("the classify result of sample {} is {}, the real label is {}".
                  format(t, classifyresult, truelabel))
            if classifyresult != truelabel:
                errorcount += 1
        print("the total error rate is {}".format(errorcount/testsize))

    def predict(self):
        pass

if __name__ == '__main__':
    hi= HandwritingInspector()
    hi.handwritingclasstest(3)