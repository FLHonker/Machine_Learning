import numpy as np
import matplotlib.pyplot as plt
def load_data(filename, delim='\t'):

    fr = open(filename)
    string_arr = [line.strip().split(delim) for line in fr]
    data = [list(map(float, line)) for line in string_arr]

    return np.mat(data)

def pac(datamat, top_nfeat=9999999):
    """
    data(m,n);(data.T*data)(n,n);eig_vects(n,n);reg_eig_vects(n,k)
    data_transfer(m,k) = data(m,n)*reg_eig_vects(n,k)  # 降维后的数据
    data_recon(m,n) = data_transfer(m,k) * reg_eig_vects.T(k,n)  # 重构(还原)后的数据
    """
    means = np.mean(datamat, axis=0)  # data(m,n)
    mean_removed = datamat - means
    print(mean_removed)
    covmat = np.cov(mean_removed, rowvar=False)  # 每一列作为一个变量(某个特征的所有取值), 按列求方差
    eig_vals, eig_vects = np.linalg.eig(np.mat(covmat))
    # print(eig_vals)
    # print(eig_vects)
    eig_sort = np.argsort(eig_vals)  # 排序值
    eig_sort = eig_sort[:-(top_nfeat+1):-1]  # 以-1为步长，-(top_nfeat+1)为上界
    reg_eig_vects = eig_vects[:,eig_sort]  # 列向量为特征向量
    # print(reg_eig_vects)
    low_data = mean_removed * reg_eig_vects
    reconmat = low_data * reg_eig_vects.T + means  # 还原后的数据
    return low_data, reconmat, reg_eig_vects, eig_vals

def plot_data(init_data, recon_data):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(init_data[:,0].flatten().A, init_data[:,1].flatten().A)
    ax.scatter(recon_data[:, 0].flatten().A, recon_data[:, 1].flatten().A)
    plt.show()

    # print(covmat)
    # print(1/1000*mean_removed.T * mean_removed)

def replace_nan_with_mean():
    """对于每个特征，将Nan设为其余值的均值"""
    datamat = load_data('data/secom.data', ' ')
    m = datamat.shape[1]
    for i in range(m):
        mean = np.mean(datamat[np.nonzero(~np.isnan(datamat[:,i]))[0], i])
        datamat[np.nonzero(np.isnan(datamat[:,i]))[0], i] = mean
    return datamat

def plot_var(datamat):
    low_data, reconmat, reg_eig_vects, eig_vals = pac(datamat)
    print(reg_eig_vects)
    full_data = datamat * reg_eig_vects
    var_list = list()
    full_var = np.sum(np.var(full_data, axis=0))
    for i in range(20): # 前20个主成分所对应的方差百分比
        var = np.var(full_data[:,i], axis=0)[0,0]
        var_list.append(var/full_var*100)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(list(range(1, 21)), var_list)
    plt.show()




if __name__ == '__main__':
    # data = load_data('data/testSet.txt')
    # print(data.shape)
    # low_data, reconmat, reg_eig_vects, eig_vals = pac(data, 1)
    # print(reg_eig_vects)
    # print(eig_vals)
    # print(low_data.shape)
    # print(reconmat.shape)
    # plot_data(data, reconmat)
    data = replace_nan_with_mean()
    plot_var(data)