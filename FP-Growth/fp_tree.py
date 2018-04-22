import json

class treeNode():

    def __init__(self, nameValue, numOccure, parentNode):
        self.name = nameValue
        self.count = numOccure
        self.nodeLink = None
        self.parent = parentNode
        self.children = dict()

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print('  '*ind + self.name + ' ' + str(self.count))
        for child in self.children.values():
            child.disp(ind+1)

class FP_Tree():

    def load_data(self, file=None):
        if file:
            data = []
            f = open(file)
            for l in f:
                line = l.split()
                data.append(line)
        else:
            data = [['r', 'z', 'h', 'j', 'p'],
                    ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                    ['z'],
                    ['r', 'x', 'n', 'o', 's'],
                    ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                    ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
        return data

    def creat_set(self, dataset):
        retdict = {}
        for trans in dataset:
            retdict[json.dumps(trans)] = 1
        return retdict

    def createTree(self, dataset, minsup = 1):
        headertable = {}
        for trans in dataset:  # 统计数据集中每个元素出现的次数
            trans = json.loads(trans)
            for item in trans:
                headertable[item] = headertable.get(item, 0) + dataset[json.dumps(trans)]  # 第一次扫描整个数据集
        for k in list(headertable.keys()):  # 剔除掉不满足支持度的元素
            if headertable[k] < minsup:
                headertable.pop(k)
        freq_item_set = set(headertable.keys())  # 满足支持度的元素所构成的集合
        if len(freq_item_set) == 0:
            return None, None
        for k in headertable:
            headertable[k] = [headertable[k], None]  # None表示该元素的所有相似节点
        retTree = treeNode('Null Set', 1, None)  # 初始的空节点
        for transet, count in dataset.items():  # 对数据集中的每条记录剔除不满足支持度的元素，并按照次数进行排序(如果元素次数相同怎么排序？)
            localD = dict()
            transet = json.loads(transet)  # 先将项集排序？？
            for item in transet:  # 第二次扫描整个数据集
                if item in freq_item_set:
                    localD[item] = headertable[item][0]
            if len(localD) > 0:
                ordered_items = [v[0] for v in sorted(localD.items(), key=lambda i: i[1], reverse=True)]
                self.update_tree(ordered_items, retTree, headertable, count)
        return retTree, headertable

    def update_tree(self, items, inTree, headertable, count):
        """
        递归更新fp_tree
        首先对于项集第一个元素，如果该元素在tree的子节点中，则将子节点对应的计数加1
        如果不在，则将该元素作为tree的子节点，并更新headertable
        对于项集中剩下的元素，以tree的子节点为tree，递归地更新tree
        :param items: 项集
        :param inTree: fp_tree
        :param headertable: 头指针表
        :return: None
        """

        if items[0] in inTree.children:
            inTree.children[items[0]].inc(count)
        else:
            inTree.children[items[0]] = treeNode(items[0], 1, inTree)
            if headertable[items[0]][1] == None:
                headertable[items[0]][1] = inTree.children[items[0]]  # headertable存储元素个数及该元素对应的节点
            else:
                self.update_Header(headertable[items[0]][1], inTree.children[items[0]])
        if len(items) > 1:
            self.update_tree(items[1::], inTree.children[items[0]], headertable, count)

    def update_Header(self, nodeToTest, targetNode):
        while (nodeToTest.nodeLink != None):  # 递归的寻找节点的最后一个相似节点
            nodeToTest = nodeToTest.nodeLink
        nodeToTest.nodeLink = targetNode

    def ascend_tree(self, leafNode, prefixPath):

        if leafNode.parent != None:
            prefixPath.append(leafNode.name)
            self.ascend_tree(leafNode.parent, prefixPath)

    def find_prefix_path(self, basepat, treeNode):
        """找到treenode所有父节点构成的前缀路径路径"""
        condpats = {}
        while treeNode != None:
            prefixpath = []
            self.ascend_tree(treeNode, prefixpath)
            if len(prefixpath) > 1:
                condpats[json.dumps(prefixpath[1:])] = treeNode.count
            treeNode = treeNode.nodeLink

        return condpats

    def minetree(self, intree, headertable, minsup, prefix, freq_item_list):
        """
        对headertabel中的每个频繁单项，生成其前缀路径
        以前缀路径作为数据集，生成headertabel和intree
        将headertabe中的每个频繁项加入频繁项集中
        """

        bigL = [v[0] for v in sorted(headertable.items(), key=lambda p:p[1][0])]
        for basepat in bigL:
            newfreqset = prefix.copy()
            newfreqset.add(basepat)
            freq_item_list.append(newfreqset)
            cond_patt_bases = self.find_prefix_path(basepat, headertable[basepat][1])
            my_condtree, myhead = self.createTree(cond_patt_bases, minsup)
            if myhead != None:
                self.minetree(my_condtree, myhead, minsup, newfreqset, freq_item_list)


if __name__ == '__main__':

    # rootNode = treeNode('pymaid', 9, None)
    # rootNode.children = {'eye': treeNode('eye', 13, None)}
    # rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
    # rootNode.disp()
    ft = FP_Tree()
    data = ft.load_data(file='data/kosarak.dat')
    dataset = ft.creat_set(data)
    retTree, headertable = ft.createTree(dataset, minsup=100000)
    retTree.disp()
    print(headertable)
    # print(ft.find_prefix_path('t', headertable['t'][1]))
    freqitems = list()
    ft.minetree(retTree, headertable, 100000, set([]), freqitems)
    print(freqitems)


