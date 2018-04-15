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

    def load_data(self):
        data = [['r', 'z', 'h', 'j', 'p'],
                ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                ['z'],
                ['r', 'x', 'n', 'o', 's'],
                ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
        return data

    def createTree(self, dataset, minsup = 1):
        headertable = {}
        for trans in dataset:  # 统计数据集中每个元素出现的次数
            for item in trans:
                headertable[item] = headertable.get(item, 0) + 1
        for k in list(headertable.keys()):  # 剔除掉不满足支持度的元素
            if headertable[k] < minsup:
                headertable.pop(k)
        freq_item_set = set(headertable.keys())  # 满足支持度的元素所构成的集合
        if len(freq_item_set) == 0:
            return None, None
        for k in headertable:
            headertable[k] = [headertable[k], None]  # None表示该元素的所有相似节点
        retTree = treeNode('Null Set', 1, None)  # 初始的空节点
        for transet in dataset:  # 对数据集中的每条记录剔除不满足支持度的元素，并按照次数进行排序(如果元素次数相同怎么排序？)
            localD = dict()
            transet = sorted(transet)  # 先将项集排序？？
            for item in transet:
                if item in freq_item_set:
                    localD[item] = headertable[item][0]
            if len(localD) > 0:
                ordered_items = [v[0] for v in sorted(localD.items(), key=lambda i: i[1], reverse=True)]
                self.update_tree(ordered_items, retTree, headertable)
        return retTree, headertable

    def update_tree(self, items, inTree, headertable):
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
            inTree.children[items[0]].inc(1)
        else:
            inTree.children[items[0]] = treeNode(items[0], 1, inTree)
            if headertable[items[0]][1] == None:
                headertable[items[0]][1] = inTree.children[items[0]]  # headertable存储元素个数及该元素对应的节点
            else:
                self.update_Header(headertable[items[0]][1], inTree.children[items[0]])
        if len(items) > 1:
            self.update_tree(items[1::], inTree.children[items[0]], headertable)

    def update_Header(self, nodeToTest, targetNode):
        while (nodeToTest.nodeLink != None):  # 递归的寻找节点的最后一个相似节点
            nodeToTest = nodeToTest.nodeLink
        nodeToTest.nodeLink = targetNode

if __name__ == '__main__':

    rootNode = treeNode('pymaid', 9, None)
    rootNode.children = {'eye': treeNode('eye', 13, None)}
    rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
    rootNode.disp()
    ft = FP_Tree()
    data = ft.load_data()
    retTree, headertable = ft.createTree(data, minsup=3)
    retTree.disp()
    print(headertable)


