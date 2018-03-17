from math import log
import numpy as np
from pprint import pprint
import json
from os import path
from copy import deepcopy

filepath = path.dirname(path.dirname(__file__))


class Tree():
    def calc_ShannonEnt(self, dataSet):
        # 计算香农熵
        # 统计数据集中每个label出现的次数，存储到dict中
        numEntries = len(dataSet)
        labelCounts = dict()
        for featVec in dataSet:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts:
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/numEntries
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

    def calc_GiniImpurity(self, dataSet):
        # 计算基尼指数
        size = len(dataSet)
        label_dict = dict()
        for i in dataSet:
            label = i[-1]
            count = label_dict.get(label, 0)
            label_dict[label] = count + 1
        GiniImpurity = 1
        for l, c in label_dict.items():
            prob = float(c/size)
            GiniImpurity -= prob ** 2
        return GiniImpurity

    def split_Dataset(self, dataset, axis):
        # 根据特征的值对数据集进行划分
        splited_dict = dict() #特征划分后的子节点
        size = dataset.shape[1]
        for featVec in dataset:
            current_set = splited_dict.get(featVec[axis], np.empty((0, size-1))) # size - 1 当前子节点划分
            new_sample = np.append(featVec[:axis], featVec[axis+1:]) # 剔除掉选取的特征值
            splited_dict[featVec[axis]] = np.row_stack([current_set, new_sample])
        return splited_dict

    def get_best_feature(self, dataset, feature_map): # feature_map = {0:'a',1:'b'...}
        feature_num = dataset.shape[1] - 1 # 特征数
        datasize = dataset.shape[0]
        baseshannon = self.calc_ShannonEnt(dataset) # 基准熵
        bestfeature = feature_map[0] # 最佳特征初始化
        bestInfoGain = 0 # 最佳信息增益初始化
        bestfeature_dict = dict() # 最佳子节点初始化
        for i in range(feature_num):
            feature = feature_map[i]
            feature_dict = self.split_Dataset(dataset, i)
            # print(feature_dict)
            set_shannon = 0 # 子节点信息熵初始化
            for subset in feature_dict.values():
                pro = subset.shape[0]/datasize
                set_shannon += pro * (self.calc_ShannonEnt(subset))
            InfoGain = baseshannon - set_shannon # 子节点信息增益
            if InfoGain > bestInfoGain:
                bestInfoGain = InfoGain
                bestfeature = feature
                bestfeature_dict = feature_dict
        return bestfeature, bestfeature_dict

    def max_count_label(self, labels):
        label_dict = {}
        for l in labels:
            label_dict[l] = label_dict.get(l, 0) + 1
        label_dict = sorted(label_dict.items(), key=lambda x:x[1], reverse=True)
        return label_dict[0][0] #取样例最多的一个类，如果数量一样，默认取第一个

    def get_feature_map(self, feature_map, best_feature_index):
        # 剔除最佳特征后得到特征映射
        print("before:{} ,index:{}".format(feature_map, best_feature_index))
        feature_map.pop(best_feature_index)
        new_feature_map = dict()
        for k, v in feature_map.items():
            if k > best_feature_index:
                new_feature_map[k-1] = v
            else:
                new_feature_map[k] = v
        print("after:{}".format(new_feature_map))
        return new_feature_map

    def gen_tree(self, dataset, feature_map):
        """
        基线条件：
        1. 样本类别相同
        2. 特征映射为空
        3. 样本特征相同
        """
        labels = [l[-1] for l in dataset]
        if len(set(labels)) == 1:
            return labels[0]
        if not feature_map:
            return self.max_count_label(labels)
        same = True
        for key in feature_map:
            if len(set(dataset[:,key])) != 1:
                same = False
        if same:
            return self.max_count_label(labels)

        tree = dict()
        bestfeature, feature_dict = self.get_best_feature(dataset, feature_map)
        tree[bestfeature] = dict()
        bestfeature_index = list(filter(lambda x: x[1] == bestfeature, feature_map.items()))[0][0]
        new_feature_map = self.get_feature_map(feature_map, bestfeature_index)
        for k, v in feature_dict.items():
            map = deepcopy(new_feature_map) # 涉及到可变对象的迭代一定要注意! 在迭代过程中对象可能已改变
            tree[bestfeature][k] = self.gen_tree(v, map) # {'feature':{'value':{'feature':'value'...
        return tree

    def classify(self, tree, feature_map, data):
        feature = list(tree.keys())[0]
        feature_index = list(filter(lambda x: x[1] == feature, feature_map.items()))[0][0]
        class_ = tree[feature][str(data[feature_index])]
        if type(class_) != dict:
            return class_
        else:
            return self.classify(class_, feature_map, data)

    def load_data(self, file):
        p = path.join(filepath, file)
        data_set = np.empty([0,5])
        with open(p, 'r') as f:
            for l in f:
                data = l.strip().split('\t')
                data_set = np.row_stack((data_set, data))
        return data_set

    def save_tree(self, tree):
        with open('tree.txt','w') as f:
            f.write(json.dumps(tree))


if __name__ == '__main__':
    tree = Tree()
    data = tree.load_data('machinelearninginaction/Ch03/lenses.txt') # 训练集必须包括特征所有可能的值
    feature_map = {0:'age', 1:'prescript', 2:'astigmatic', 3:"tearRate"}
    t = tree.gen_tree(data, feature_map) # feature_map发生了变化
    tree.save_tree(t)
    # print(t)
    # print(tree.classify(t, {0:'no surfacing', 1:'flippers'}, ['no','no']))