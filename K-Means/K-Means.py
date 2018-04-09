import numpy as np
import random

class KMEANS():

    def __init__(self):
        pass

    def load_data(self, filename):
        datamat = []
        file = open(filename)
        for l in file:
            line = l.strip().split('\t')
            line = list(map(float, line))
            datamat.append(line)
        return np.mat(datamat)

    def calc_dist(self, vecA, vecB):
        return np.sqrt(np.sum(np.power(vecA-vecB, 2), 1)) # np.sum一定要指定axis, 默认为0

    def rand_seed(self, k, datamat):
        m, n = datamat.shape
        indexes = random.sample(range(m), k)
        rand_seeds = datamat[indexes,:]
        return rand_seeds

    def kMeans(self, datamat, k, threshold=0.0001, distmeans=calc_dist, createseed=rand_seed):

        m, n = datamat.shape
        rand_seeds = createseed(self, k, datamat)
        mean, cluster = dict(), dict()
        mean = {i: rand_seeds[i,:] for i in range(k)}
        changed = True
        while changed:
            changed = False
            old_mean = mean.copy()
            print(old_mean)
            cluster = {i: np.empty((0, n)) for i in range(k)} # 每次循环将cluster清空，重新分组
            for i in range(m):
                mindist, minindex = np.Inf, 0
                for j in range(k):
                    dist = distmeans(self, datamat[i,:], mean[j])
                    if dist < mindist:
                        mindist, minindex = dist, j
                cluster[minindex] = np.row_stack((cluster[minindex], datamat[i,:]))
            mean = {i: np.mat(np.mean(cluster[i], 0)) for i in range(k)}
            for i in range(k): # 如果任何一簇的改变大于阈值，则认为未收敛，反之则收敛
                if distmeans(self, old_mean[i], mean[i]) > threshold:
                    changed = True
        return mean, {i: np.mat(cluster[i]) for i in range(k)}

if __name__ == '__main__':
    kmeans = KMEANS()
    datamat = kmeans.load_data('data/testSet.txt')
    mean, cluster = kmeans.kMeans(datamat, 4)
    print(mean)
    print(cluster)


