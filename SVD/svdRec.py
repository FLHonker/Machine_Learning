import numpy as np
from numpy import linalg as la
def loadExData():
    return np.mat([[1,1,1,0,0],
                   [2,2,2,0,0],
                   [1,1,1,0,0],
                   [5,5,5,0,0],
                   [1,1,0,2,2],
                   [0,0,0,3,3],
                   [0,0,0,1,1]])

def loadExData2():
    return np.mat([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])

def euclid_sim(inA, inB):
    """欧氏距离"""
    return 1/(1+la.norm(inA-inB)) # 计算欧式距离

def pears_sim(inA, inB):
    """皮尔逊相关系数"""
    if len(inA) < 3:
        return 1
    return 0.5+0.5*np.corrcoef(inA, inB, rowvar=False)[0,1]  # 相关系数范围（-1，1）,按照列来计算方差

def cos_sim(inA, inB):
    """余弦定理"""
    num = float(inA.T * inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5*(num/denom)

def standEst(datamat, user, sim_method, item):
    """根据相似度计算用户对于某个item的评分"""
    n = np.shape(datamat)[1]
    score, total_sim = 0, 0
    for i in range(n):
        relevent_score = datamat[user, i]
        if relevent_score == 0: # 选取不为0的计算相似度
            continue
        index1 = set(np.nonzero(datamat[:, item])[0].tolist()) # 注意，索引不要弄错了，第二个维度为固定值，所以均为0
        index2 = set(np.nonzero(datamat[:, i])[0].tolist())
        overlap = list(index1 & index2)
        # 只截取两向量重叠部分计算相似度，因为0表示数据缺失（未品尝过该菜肴）不代表评分为0
        if not overlap:
            print("no overlapping")
            continue
        sim = sim_method(datamat[overlap, item], datamat[overlap, i])
        score += sim*relevent_score
        total_sim += sim
    return score/total_sim if total_sim != 0 else 0

def svdEst(datamat, user, sim_method, item):
    """根据相似度计算用户对于某个item的评分"""
    n = np.shape(datamat)[1]
    score, total_sim = 0, 0
    u,sigma,vt = np.linalg.svd(datamat)  # d(m,n) = u(m,m)*lam(m,n)*v(n,n) ~ u(m,k)*lam(k,k)*v(k,n)
    sig, i = sigma_transfer(sigma)
    # 选取前k行比较相似度
    transformed_data = sig * vt[:i+1, :] # d(k,m) = ut(k,m) * d(m,n) ~ lam(k,k) * v(k,n)
    for i in range(n):
        relevent_score = datamat[user, i]
        if relevent_score == 0: # 选取不为0的计算相似度
            continue
        index1 = set(np.nonzero(transformed_data[:, item])[0].tolist()) # 注意，索引不要弄错了，第二个维度为固定值，所以均为0
        index2 = set(np.nonzero(transformed_data[:, i])[0].tolist())
        overlap = list(index1 & index2)
        # 只截取两向量重叠部分计算相似度，因为0表示数据缺失（未品尝过该菜肴）不代表评分为0
        if not overlap:
            print("no overlapping")
            continue
        sim = sim_method(transformed_data[overlap, item], transformed_data[overlap, i])
        score += sim*relevent_score
        total_sim += sim
    return score/total_sim if total_sim != 0 else 0

def sigma_transfer(sigma):
    sig = sigma ** 2
    sig_sum = np.sum(sig)
    sum, i = 0, 0
    for i in range(len(sig)):
        sum += sig[i]
        if sum / sig_sum > 0.9:
            break
    return np.mat(np.eye(i+1) * sigma[:i+1]), i

def recommend(datamat, user, N=3, sim_method=cos_sim, est_method=standEst):
    unrated_items = np.nonzero(datamat[user,:].A == 0)[1]
    if len(unrated_items) == 0:
        print("you've rated everything")
    item_scores = []
    for item in unrated_items:
        estimated_score = est_method(datamat, user, sim_method, item)
        item_scores.append((item, estimated_score))
    return sorted(item_scores, key=lambda x:x[1], reverse=True)[:N]

def load_data3(file):
    f = open(file)
    data = list()
    for l in f:
        row = [int(i) for i in l.strip()]
        data.append(row)
    return np.mat(data)

def printmat(data, thresh):
    for i in range(32):
        string = ''
        for j in range(32):
            if data[i, j] > thresh:
                string += '1'
            else:
                string += '0'
        print(string + '\n')

def img_compress(data, numsv=3, thresh=0.9):
    print("**********origin data*************\n")
    printmat(data, thresh)
    u,s,v = np.linalg.svd(data)
    sigma = np.mat(np.eye(numsv) * s[:numsv])
    data_est = u[:,:numsv] * sigma * v[:numsv,:]
    print("*************after compress************\n")
    printmat(data_est, thresh)




if __name__ == "__main__":
    # data = loadExData2()
    # print(euclid_sim(data[:,0], data[:,4]))
    # print(pears_sim(data[:,0], data[:,0]))
    # print(cos_sim(data[:,0], data[:,0]))
    # print(standEst(data, 5, pears_sim, 1))
    # data[0,1] = data[0,0] = data[1,0] = data[2,0] = 4
    # data[3,3] = 2
    # print(data)
    # data = loadExData2()
    # u, sigma, vt = np.linalg.svd(data)
    # print(sigma)
    # sig = sigma ** 2
    # sig_sum = sum(sig)
    # # print(sig)
    # print(recommend(data, 1, est_method=svdEst))
    # sum = 0
    # for i in range(len(sig)):
    #     sum += sig[i]
    #     if sum/sig_sum > 0.9:
    #         break
    data = load_data3('data/0_5.txt')
    img_compress(data)



