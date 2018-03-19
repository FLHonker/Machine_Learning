import re
from os import path
from bayes import NaiveBayes
import numpy as np
from math import log
import random
import json

filepath = path.dirname(path.dirname(__file__))


class SpamFilter(NaiveBayes):

    def __init__(self):
        # todo:改进init函数，使之通用性更强
        super(SpamFilter, self).__init__()
        self.test_set, self.test_class, self.train_set, self.train_class = self.choose_set()

    def load_DataSet(self, *args, **kwargs):
        file_map = {'s':{'location':'machinelearninginaction/Ch04/email/spam/', 'class':1},
                    'h':{'location':'machinelearninginaction/Ch04/email/ham/', 'class':0}}
        data_set = []
        class_list = []
        for k in file_map:
            for i in range(1,26): # spam与ham各25个样本
                file = open(path.join(filepath, file_map[k]['location']+'{}.txt'.format(i)), 'r')
                text = file.read()
                parsed_list = self.parse_text(text)
                data_set.append(parsed_list)
                class_list.append(file_map[k]['class'])
                file.close()

        return data_set, class_list

    def parse_text(self, text):

        word_list = re.split(r'\W*', text)  # \W匹配所有的字母/数字/下划线以外的字符
        return [i.lower() for i in word_list if len(i) > 2]

    def choose_set(self):
        """
        共有50个样本，采用留出法，选取40个训练集，10个测试集
        为保证数据分布的一致性，训练集及测试集中 spam:ham=1
        """
        randindex1 = random.sample(range(int(1/2*len(self.data_set))), 5) # 从指定范围内随机选取若干个互不相同的元素
        randindex2 = random.sample(range(int(1/2*len(self.data_set)), len(self.data_set)),5)
        randindex1.extend(randindex2)
        test_set = [self.data_set[i] for i in randindex1]
        test_class = [self.class_vec[i] for i in randindex1]
        train_index = set(range(len(self.data_set))) - set(randindex1) # 求差集
        train_set = [self.data_set[i] for i in train_index]
        train_class = [self.class_vec[i] for i in train_index]
        # return self.word2mat(test_set), test_class, self.word2mat(train_set), train_class
        test_set = self.word2mat(test_set)
        train_set = self.word2mat(train_set)
        return test_set, test_class, train_set, train_class

    def sample_test(self):
        p0_vec, p1_vec, p_spam = self.train_nb(self.train_set, self.train_class) # 可根据情况省略

        error_count = 0
        for i in range(len(self.test_set)):
            classify_result = self.classify(self.test_set[i], p0_vec, p1_vec, p_spam)
            if classify_result != self.test_class[i]:
                error_count += 1
        print("the error count is {}".format(float(error_count/len(self.test_class))))

    def test(self, input):
        with open('models/bayes_model.json', 'r') as f:
            model = json.load(f)
        input_vec = self.word2vec(input)
        classify_result = self.classify(input_vec, np.array(model['p0_vec']), np.array(model['p1_vec']), model['p'])

        print("the classify result of {} is {}".format(input, classify_result))

if __name__ == '__main__':
    # list_, class_ = load_DataSet()
    # voc = create_Vocalblist(list_)
    # # print(voc)
    # # print(word2vec(voc, list_[0]))
    # mat = word2mat(voc, list_)
    # # print(mat)
    # print(train_nb(mat, class_))
    # print(classify_nb(['love','peace']))
    nb = SpamFilter()
    print(nb.voc)
    nb.sample_test()
    nb.test(['hope','I','good']) # 长度小于3的词会被过滤掉



