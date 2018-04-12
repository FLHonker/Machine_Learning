import numpy as np
import json

class Apriori():

    def __init__(self):
        pass

    def load_data(self):
        return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

    def creat_c1(self, dataset):
        c1 = list()
        for d in dataset:
            c1.extend(d)
        c1 = [[i] for i in sorted(list(set(c1)))]
        return c1

    def creat_ck(self, dataset, k):
        """生成以k个数据为元素的数据集合"""
        ck = list()
        for i in range(len(dataset)):
            for j in range(i+1, len(dataset)):
                L1 = sorted(dataset[i][:k-2])
                L2 = sorted(dataset[j][:k-2])
                if L1 == L2: # 前面k-2个元素相同时合并；eg:[1, 2], [2, 3] ==> [1, 2, 3]
                    ck.append(list(set(dataset[i]) | set(dataset[j])))
        return ck

    def scanD(self, Data, Ck, minsupport):
        count = dict()
        for i in range(len(Ck)):
            for j in range(len(Data)):
                if set(Ck[i]).issubset(set(Data[j])):
                    count[json.dumps(Ck[i])] = count.get(json.dumps(Ck[i]), 0) + 1
        support, support_data = dict(), list()
        num_items = len(Data)
        for i in range(len(Ck)):
            key = json.dumps(Ck[i])
            support_i = count.get(key, 0)/num_items
            if support_i >= minsupport:
                support[key] = support_i
                support_data.append(Ck[i])
        return support, support_data

    def apriori(self, dataset, minsupport=0.75):
        support, support_data = dict(), list()
        ck = self.creat_c1(dataset)
        s, sd = self.scanD(dataset, ck, minsupport)
        support.update(s)
        support_data.append(sd)
        k = 2
        while sd:
            ck = self.creat_ck(ck, k)
            s, sd = self.scanD(dataset, ck, minsupport)
            support.update(s)
            support_data.append(sd)
            k += 1
        return support_data, support

    def generate_rules(self, support_data, support, minconf=0.7):

        big_rule_list = list()
        for i in range(1, len(support_data)):
            if support_data[i]:
                for freqset in support_data[i]:
                    freqset = [[i] for i in freqset]
                    if i > 1:
                        self.rules_from_conseq(freqset, support, big_rule_list, minconf)
                    else:
                        self.calc_conf(freqset, support, big_rule_list, minconf)

        return big_rule_list

    def calc_conf(self, freqset, support, big_rule_list, minconf):
        data = list()
        _ = list(map(lambda x:data.extend(x), freqset))
        data = list(set(data))
        for d in freqset:
            rd = list(set(data) - set(d))
            conf = support[json.dumps(data)] / support[json.dumps(d)]
            if conf > minconf:
                print('{} --> {}, conf: {}'.format(d, rd, conf))
                big_rule_list.append((d, rd, conf))

    def rules_from_conseq(self, freqset, support, big_rule_list, minconf):
        self.calc_conf(freqset, support, big_rule_list, minconf)
        l = len(freqset)
        for k in range(2, l):
            freqset = self.creat_ck(freqset, k)
            self.calc_conf(freqset, support, big_rule_list, minconf)


if __name__ == '__main__':
    ap = Apriori()
    data = ap.load_data()
    # c1 = ap.creat_c1(data)
    # print(c1)
    support_data, support = ap.apriori(data, 0.5)
    print(support,support_data)
    rules = ap.generate_rules(support_data ,support, minconf=0.5)
    print(rules)
