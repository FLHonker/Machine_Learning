# coding: utf-8

import matplotlib.pyplot as plt
from copy import deepcopy
import json

class PlotTree():
    def __init__(self, tree):
        self.tree = tree
        self.decision_node = dict(boxstyle="sawtooth", fc="0.8")
        self.leaf_node = dict(boxstyle="round4", fc="0.8")
        self.arrow_args = dict(arrowstyle="<-")
        self.tree_width = float(self.tree_scale(tree)[0])
        self.tree_height = float(self.tree_scale(tree)[1])

    def plot_node(self, nodetxt, middletxt, childpt, parentpt, nodetype):
        self.figure.annotate(nodetxt, xy=parentpt, xycoords="axes fraction",
                                xytext=childpt, textcoords="axes fraction",
                                va="center", ha="center", bbox=nodetype,
                                arrowprops=self.arrow_args)
        xmid = (parentpt[0] + childpt[0]) / 2.0
        ymid = (parentpt[1] + childpt[1]) / 2.0
        self.figure.text(xmid, ymid, middletxt)


    def plot_tree(self, mytree, parentpt):
        feature = list(mytree.keys())[0]
        num_of_branch = len(list(mytree[feature].keys()))
        x_interval = float(1/self.tree_width/(num_of_branch-1)) # 将子节点间的最大间隔设为1,均分；由于图幅默认为(1,1),缩放
        y_interval = float(1 / self.tree_height)
        child_list = [(parentpt[0]-1/self.tree_width/2+i*x_interval, parentpt[1]- y_interval) for i in range(num_of_branch)]
        for k, v in mytree[feature].items():
            pt = child_list.pop()
            if type(v) == str:
                self.plot_node(v, k, pt, parentpt, self.leaf_node)
            else:
                self.plot_node(list(v.keys())[0], k, pt, parentpt, self.decision_node)
                self.plot_tree(mytree[feature][k], pt)

    def run(self, tree, parentpt):
        first_feature = list(tree.keys())[0]
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.figure = plt.subplot(111, frameon=False, **axprops)
        self.figure.annotate(
                             first_feature,xy=parentpt,xycoords="axes fraction",
                             xytext=parentpt, textcoords="axes fraction",
                             va="center", ha="center", bbox=self.decision_node,
                             ) # 父节点初始化
        self.plot_tree(tree, parentpt)
        plt.show()

    def tree_scale(self, tree):
        # 不采用递归得到树的广度及深度

        leafnum = 0
        depth = 1
        feature = list(tree.keys())[0]
        trees = list(tree[feature].values())
        while trees:
            tree_copy = deepcopy(trees) # 深复制
            for i, t in enumerate(tree_copy):
                if type(t) != dict:
                    leafnum += 1
                    trees.remove(t)
                else:
                    trees.remove(t)
                    feature = list(t.keys())[0]
                    trees.extend(list(t[feature].values()))
            depth += 1

        return leafnum, depth



if __name__ == '__main__':
    # tree = {'no surfacing': {'yes': {'flippers': {'yes': 'true', 'no': 'false'}}, 'no': 'false'}}
    with open('tree.txt', 'r') as f:
        tree = json.loads(f.read())
    pt = PlotTree(tree)
    print(pt.tree_scale(tree))
    pt.run(tree, (0.5,1))
    # pt.figure.imshow((1,1))