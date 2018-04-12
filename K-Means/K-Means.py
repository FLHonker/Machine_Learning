import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

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
        return np.sqrt(np.sum(np.power(vecA-vecB, 2), 1))  # np.sum一定要指定axis, 默认为0

    def rand_seed(self, k, datamat):
        m, n = datamat.shape
        indexes = random.sample(range(m), k)
        rand_seeds = datamat[indexes,:]
        return rand_seeds

    def kMeans(self, datamat, k, threshold=0.0001, distmeans=calc_dist, createseed=rand_seed):

        m, n = datamat.shape
        rand_seeds = createseed(self, k, datamat)
        mean, cluster = list(), list()
        mean = [rand_seeds[i,:] for i in range(k)]
        changed = True
        while changed:
            changed = False
            old_mean = mean.copy()
            # print(old_mean)
            cluster = [np.empty((0, n)) for _ in range(k)]  # 每次循环将cluster清空，重新分组
            for i in range(m):
                mindist, minindex = np.Inf, 0
                for j in range(k):  # 将点划分到与其最近的一个质心
                    dist = distmeans(self, datamat[i,:], mean[j])
                    if dist < mindist:
                        mindist, minindex = dist, j
                cluster[minindex] = np.row_stack((cluster[minindex], datamat[i,:]))
            mean = [np.mat(np.mean(cluster[i], 0)) for i in range(k)]
            for i in range(k):  # 如果任何一簇的改变大于阈值，则认为未收敛，反之则收敛
                if distmeans(self, old_mean[i], mean[i]) > threshold:
                    changed = True
        return mean, [np.mat(cluster[i]) for i in range(k)]

    def calc_sse(self, mean, cluster):
        sse = 0
        for i in range(len(mean)):
            sse += np.sum(np.power(cluster[i]-mean[i], 2))
        return sse

    def bi_kmeans(self, datamat, k, distmeans=calc_dist):
        """选取二分后sse改变最大的簇进行二分，直至达到指定的size"""
        mean, cluster = self.kMeans(datamat, 2)
        while len(cluster) < k:
            bestsse, bestindex, bestmean, bestcluster = 0, 0, list(), list()
            for i in range(len(cluster)):
                before_sse = self.calc_sse([np.mean(cluster[i], 0)], [cluster[i]])
                current_mean, current_cluster = self.kMeans(cluster[i], 2)
                current_sse = self.calc_sse(current_mean, current_cluster)
                red_sse = before_sse - current_sse
                if red_sse > bestsse: # 找到二分后sse改变最大的簇进行二分
                    bestsse = red_sse
                    bestindex = i
                    bestmean = current_mean
                    bestcluster = current_cluster
            mean.pop(bestindex)
            mean.extend(bestmean)
            cluster.pop(bestindex)
            cluster.extend(bestcluster)
        return mean, cluster

    def plot_result(self, cluster, mean):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(cluster)):
            markers = MarkerStyle().markers
            marker = random.choice(list(markers.keys()))
            ax.scatter(cluster[i][:, 0].A.T, cluster[i][:, 1].A.T, marker=marker)
            ax.scatter(mean[i][0, 0], mean[i][0, 1], marker='+', s=500)
        plt.show()



if __name__ == '__main__':
    kmeans = KMEANS()
    datamat = kmeans.load_data('data/testSet.txt')
    mean, cluster = kmeans.kMeans(datamat, 4)
    print(mean)
    kmeans.plot_result(cluster, mean)
    # print(cluster)
    mean, cluster = kmeans.bi_kmeans(datamat, 4)
    print(mean)
    kmeans.plot_result(cluster, mean)
    # print(cluster)

