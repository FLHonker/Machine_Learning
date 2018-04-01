import numpy as np

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

def standard_regression(data_mat, label_mat):
    xTx = data_mat.T * data_mat
    if np.linalg.det(xTx) == 0: # 求矩阵行列式
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * data_mat.T * label_mat
    return ws

def plot_data(ws, data_mat, label_mat):
    pass

if __name__ == '__main__':
    data_mat, label_mat = load_data('data/ex0.txt')
    print(data_mat)
    ws = standard_regression(data_mat, label_mat)
    print(ws)

