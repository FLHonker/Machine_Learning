from numpy import *
import operator

class KNN():
    def createDataset(self):
        group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
        labels = ['A', 'A', 'B', 'B']
        return group, labels

    def classify0(self, inX, dataSet, labels, k):
        dataSetSize = dataSet.shape[0] # 求矩阵size(元素个数)
        diffMat = tile(inX, (dataSetSize, 1)) - dataSet # 将inX复制dataSetSize份后求差
        sqDiffMat = diffMat**2 # 求矩阵各元素平方
        sqDistances = sqDiffMat.sum(axis=1) # axis=1 求各行向量和，axis=0 求各列向量和
        distances = sqDistances**0.5
        print(distances)
        sortedDistIndicies = distances.argsort() # 得到排序eg:[0,1,3,2]
        print(sortedDistIndicies)
        classCount = dict()
        for i in  range(k):
            voteIlable = labels[sortedDistIndicies[i]] # 按距离从小到大得到各标签个数
            classCount[voteIlable] = classCount.get(voteIlable,0) + 1
        sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True) # 按标签从大到小排列
        return sortedClassCount[0][0]

if __name__ == '__main__':
    knn = KNN()
    g,l = knn.createDataset()
    print(knn.classify0([1,3], g, l, 2))