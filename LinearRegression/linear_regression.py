import numpy as np
import matplotlib.pyplot as plt
from numpy import Inf

def load_data(filename):
    f = open(filename, 'r')
    data, label = list(), list()
    for l in f:
        line = l.split('\t') # 应将字符串转为数字
        data_array = [float(i) for i in line[:-1]]
        label_array = float(line[-1])
        data.append(data_array)
        label.append(label_array)
    return np.mat(data), np.mat(label).T

def standard_regression(in_mat, out_mat):
    """
    标准线性回归
    :param in_mat:
    :param out_mat:
    :return:
    """
    xTx = in_mat.T * in_mat
    if np.linalg.det(xTx) == 0: # 求矩阵行列式
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * in_mat.T * out_mat
    return ws

def plot_data(in_mat, out_mat, ws=np.zeros((1,1)), predict_out=None):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(in_mat[:,1].flatten().A, out_mat[:,0].flatten().A)
    sorted_in_mat = np.sort(in_mat, 0)
    if ws.any():
        predict_out_mat = sorted_in_mat * ws
    else:
        sort_index = np.argsort(in_mat[:,1], 0)
        predict_out_mat = predict_out[sort_index][:,0,:]
    ax.plot(sorted_in_mat[:,1].flatten().A[0], predict_out_mat[:,0].flatten().A[0])
    plt.show()

def cal_corrcoef(ws, in_mat, out_mat):
    """
    计算预测值与实际值的相关系数cov(X,Y)/D(X)*D(Y);
    cov(X,Y):协方差E((X-EX)(Y-EY));
    D(X):标准差
    :param ws:
    :param in_mat:
    :param out_mat:
    :return:
    """
    predict_label = in_mat * ws
    corrcoef = np.corrcoef(predict_label.T, out_mat.T) # 转化为行向量
    print(corrcoef)

def lwlr(testpoint, in_mat, out_mat, k=1):
    """
    根据输入点与所有样本点的距离计算权重矩阵；
    根据权重矩阵得到局部加权后的w;
    根据w得到输入点的预测值
    :param testpoint: 输入点
    :param in_mat: 输入矩阵
    :param out_mat: 输出矩阵
    :param k: 参数
    :return: 输出值
    """
    size = np.shape(in_mat)[0]
    weights = np.mat(np.eye((size)))
    for i in range(size):
        diff_mat = testpoint - in_mat[i,:]
        weights[i,i] = np.exp(diff_mat*diff_mat.T/(-2*k**2))
    xTx = in_mat.T * weights * in_mat
    if np.linalg.det(xTx) == 0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * in_mat.T * weights * out_mat
    return testpoint * ws

def weights_regression(in_mat, out_mat, k=1):
    """局部加权回归"""
    size = np.shape(in_mat)[0]
    predict_out = np.mat(np.zeros((size,1)))
    for i in range(size):
        predict_out[i,0] = lwlr(in_mat[i,:], in_mat, out_mat, k=k)
    return predict_out

def rssError(out_mat, predict_out):
    return ((out_mat.A-predict_out.A)**2).sum() # .A将矩阵转换为array

def ridge_regres(in_mat, out_mat, lam=0.2):
    """岭回归"""
    xTx = in_mat.T * in_mat
    denom = xTx + np.eye(np.shape(in_mat)[1]) * lam
    if np.linalg.det(denom) == 0:
        print("This matrix is singular, cannot do inverse")
    ws = denom.I * (in_mat.T * out_mat)
    return ws

def norm_data(x_mat):
    """
    标准化，输出标签无需处理
    :param x_mat:
    :return:
    """
    x_mean = np.mean(x_mat, 0) # 均值
    x_var = np.var(x_mat, 0) # 方差
    x_mat = (x_mat - x_mean)/x_var
    return x_mat

def ridge_test(x_mat, y_mat):
    """
    为避免特征量级对结果的影响，首先对其标准化
    :param in_mat:
    :param out_mat:
    :return:
    """
    x_mat = norm_data(x_mat)
    num = 30
    wmat = np.zeros((num, x_mat.shape[1]))
    for i in range(num):
        wmat[i,:] = ridge_regres(x_mat, y_mat, np.exp(i-10)).T
    return wmat

def plot_weights(ridge_weights):
    fig = plt.figure()
    ax =fig.add_subplot(111)
    ax.plot(ridge_weights)
    plt.show()

def stage_wise(x_mat, y_mat, step=0.005, iter=1000):
    x_mat = norm_data(x_mat)
    feature = x_mat.shape[1]
    weights = np.zeros((iter, feature))
    best_weights = np.zeros((1, feature))
    m_weights = np.zeros((1, feature))
    best_diff = Inf
    for i in range(iter):
        for j in range(feature):
            for sign in [-1, 1]:
                wstest = best_weights.copy()
                wstest[0, j] += step * sign
                predict_out = x_mat * wstest.T
                diff = rssError(y_mat, predict_out)
                if diff < best_diff:
                    print(diff)
                    best_diff = diff
                    m_weights = wstest # 为什么不是m_weights[0, j] = wstest[0. j]
        best_weights = m_weights.copy()
        weights[i,:] = best_weights
    return weights


if __name__ == '__main__':
    x_mat, y_mat = load_data('data/abalone.txt')
    # ws = standard_regression(in_mat, out_mat)
    # print(ws)
    # plot_data(in_mat, out_mat, ws=ws)
    # cal_corrcoef(ws, in_mat, out_mat)
    # 在预测数据上的结果,K越小,靠近预测点的权重越iio大,越容易造成过拟合
    # predict_out = weights_regression(in_mat[100:199], out_mat[0:99], k=0.1)
    # print(predict_out)
    # print(rssError(in_mat[100:199], predict_out))
    # plot_data(in_mat, out_mat, predict_out=predict_out)
    weights = stage_wise(x_mat, y_mat)
    print(weights)
    plot_weights(weights)
