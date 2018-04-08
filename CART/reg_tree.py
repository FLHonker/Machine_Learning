import numpy as np
from pprint import pprint

class RegTree():

    def __init__(self):
        pass

    def load_data(self, filename):
        data_mat = []
        file = open(filename, 'r')
        for l in file:
            line = l.strip().split('\t')
            data_line = list(map(float, line))
            data_mat.append(data_line)
        return np.mat(data_mat)

    def bsplit_datamat(self, datamat, feature, value):
        """
        根据属性值对数据集进行划分
        :param datamat: 数据集
        :param feature: 划分属性
        :param value: 属性值
        :return: 划分后的数据集
        """
        # nonzero返回满足条件的索引，索引按照维度进行排列
        # eg:a = np.mat([[1,2,3],[4,5,6],[7,8,9]]), np.nonzero(a[:,2]>3) = (array([1, 2]), array([0, 0]))
        # 元祖第一个元素表示行索引，第二个表示列索引，由于条件限定列为2，故返回的结果中列索引均为0
        mat0 = datamat[np.nonzero(datamat[:, feature] > value)[0], :]
        mat1 = datamat[np.nonzero(datamat[:, feature] <= value)[0], :]
        return mat0, mat1

    def standard_regression(self, in_mat, out_mat):
        """
        标准线性回归
        :param in_mat:
        :param out_mat:
        :return:
        """
        xTx = in_mat.T * in_mat
        if np.linalg.det(xTx) == 0:  # 求矩阵行列式
            print("This matrix is singular, try to increase the size of ops")
            return
        ws = xTx.I * in_mat.T * out_mat
        return ws

    def calc_leaf(self, datamat):
        """求叶节点"""
        m, n = datamat.shape
        in_mat = np.mat(np.ones((m, n)))
        in_mat[:, 1:] = datamat[:, 0:-1] # 增加一列全为1的项(wx+b. 相当于增加常数项)
        out_mat = datamat[:, -1]
        return self.standard_regression(in_mat, out_mat)

    def calc_err(self, datamat):
        """求数据集输出值的总误差"""
        m, n = datamat.shape
        in_mat = np.mat(np.ones((m, n))) # 注np.ones returns array, not mat
        in_mat[:, 1:] = datamat[:, 0:-1]  # 增加一列全为1的项(wx+b. 相当于增加常数项)
        out_mat = datamat[:, -1]
        ws = self.standard_regression(in_mat, out_mat)
        try:
            out_predict = in_mat * ws
            err = sum(np.power(out_mat-out_predict, 2))
            return err
        except TypeError as e:
            print(e)
            return np.Inf

    def choose_best_split(self, datamat, ops=(1,4)):
        """
        选取最佳分割属性及分割值，与分类树不同之处在于所有属性一直都可以被选择
        :param datamat: 数据集
        :param ops: 参数，1表示方差改变的下限, 2表示子集元素个数
        :return: 选取最佳分割属性及分割值
        """
        tols, toln = ops[0], ops[1]
        if len(set(datamat[:,-1].T.tolist()[0])) == 1:# 如果输出值相同则退出(不分割)
            return None, self.calc_leaf(datamat)
        m, n = np.shape(datamat)
        S = self.calc_err(datamat) # 整个数据集的总方差
        bestS, bestIndex, bestVal = np.Inf, 0, 0
        for feat in range(n-1):
            for val in set(datamat[:, feat].T.tolist()[0]): # 最佳分割值从现有值中选
                mat0, mat1 = self.bsplit_datamat(datamat, feat, val)
                s = self.calc_err(mat0) + self.calc_err(mat1)
                if s < bestS:
                    bestS = s
                    bestIndex = feat
                    bestVal = val
        if S - bestS < tols: # 如果最佳分割的方差变化小于阈值则退出(不分割)
            return None, self.calc_leaf(datamat)
        mat0, mat1 = self.bsplit_datamat(datamat, bestIndex, bestVal)
        if (np.shape(mat0)[0]<toln) or (np.shape(mat1)[0]<toln): # 如果分割后的子集数量太少则退出(不分割)
            return None, self.calc_leaf(datamat)
        return bestIndex, bestVal

    def get_error(self, tree, ltmat, rtmat, testdata):
        unmerge_error = sum(np.power(ltmat[:, -1] - tree['left'], 2)) + \
                        sum(np.power(rtmat[:, -1] - tree['right'], 2))
        merge_error = sum(np.power(testdata[:, -1] - 0.5 * (tree['left'] + tree['right']), 2))

        return unmerge_error, merge_error

    def create_tree(self, datamat, ops=(100,4)):
        feat, val = self.choose_best_split(datamat, ops)
        if feat == None:
            return val
        retTree = dict()
        retTree['feature'] = feat # 最佳特征
        retTree['value'] = val # 分割值
        lmat, rmat = self.bsplit_datamat(datamat, feat, val)
        retTree['left'] = self.create_tree(lmat, ops) # 左树
        retTree['right'] = self.create_tree(rmat, ops) # 右树
        return retTree

    def pre_prune(self, datamat, testdata,  ops=(100,4)):
        """
        预剪枝，时间开销小，但分支未完全展开，容易造成欠拟合
        :param datamat:
        :param testdata:
        :param ops:
        :return:
        """
        feat, val = self.choose_best_split(datamat, ops)
        if feat == None:
            return val
        retTree = dict()
        retTree['feature'] = feat # 最佳特征
        retTree['value'] = val # 分割值
        lmat, rmat = self.bsplit_datamat(datamat, feat, val) # 训练集的左右数据集
        retTree['left'] = self.calc_leaf(lmat)
        retTree['right'] = self.calc_leaf(rmat)
        ltmat, rtmat = self.bsplit_datamat(testdata, feat, val) # 测试集的左右数据集
        unmerge_error, merge_error = self.get_error(retTree, ltmat, rtmat, testdata)
        if unmerge_error < merge_error: # 如果不剪枝的误差小于剪枝的误差，则不剪枝
            retTree['left'] = self.pre_prune(lmat, ltmat, ops) # 左树
            retTree['right'] = self.pre_prune(rmat, rtmat, ops) # 右树
        else:
            print("merge")

        return retTree

    def after_prune(self, tree, testdata):
        """
        后剪枝，时间开销较大，但可避免欠拟合，泛化性能往往优于预剪枝
        :param tree: 
        :param testdata:
        :return:
        """
        ltmat, rtmat = self.bsplit_datamat(testdata, tree['feature'], tree['value'])
        if type(tree['left']) == dict: # 如果左边为树，则递归地对左边剪枝
            tree['left'] = self.after_prune(tree['left'], ltmat)
        if type(tree['right']) == dict: # 如果右边为树，则递归地对右边剪枝
            tree['right'] = self.after_prune(tree['right'], rtmat)
        # 如果剪枝后左右两边均为叶节点，则尝试合并，否则维持原样（如果树的子树没有合并，则该树必然也不会进行合并）
        if type(tree['left']) != dict and type(tree['right']) != dict:
            unmerge_error, merge_error = self.get_error(tree, ltmat, rtmat, testdata)
            if merge_error < unmerge_error:  # 如果合并后测试误差小于原误差则合并，否则保持原样
                print("merge")
                tree = 0.5 * (tree['left'] + tree['right'])
        return tree


if __name__ == '__main__':
    rt = RegTree()
    datamat = rt.load_data('data/exp2.txt')
    tree = rt.create_tree(datamat, (1,10))
    pprint(tree)
    # testdata = cart.load_data('data/ex2test.txt')
    # prune_tree = cart.after_prune(tree, testdata)
    # pprint(prune_tree)
    # pre_prune = cart.pre_prune(datamat, testdata)
    # pprint(pre_prune)

