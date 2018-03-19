import numpy as np
from math import log
import json

class NaiveBayes():

    def __init__(self):

        self.data_set, self.class_vec = self.load_DataSet()
        self.voc = self.create_Vocalblist()

    def load_DataSet(self):
        postingList=np.array([['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                     ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                     ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                     ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                     ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                     ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']])
        classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
        return postingList,classVec

    def create_Vocalblist(self):
        vocabset = list()
        for doc in self.data_set:
            vocabset.extend(list(doc))
        return list(set(vocabset))

    def word2vec(self, input):
        returnvec = [0]*len(self.voc) # list和array不要混用
        for word in input:
            if word in self.voc:
                returnvec[self.voc.index(word)] += 1
            else:
                print("word {} is not in my dictionary".format(word))
        return returnvec

    def word2mat(self, inputmat):
        returnmat = np.empty((0, len(self.voc))) # list和array不要混用
        for line in inputmat:
            l = [0] * len(self.voc)
            for w in line:
                if w in self.voc:
                    l[self.voc.index(w)] += 1
                else:
                    print("word {} is not in my dictionary".format(w))
            returnmat = np.row_stack((returnmat, l))
        return returnmat

    def train_nb(self, trainmatrix, trainCategory):
        num_train_doc = trainmatrix.shape[0]
        num_doc_words = len(trainmatrix[0])
        p_abusive = sum(trainCategory) / float(num_train_doc) #类别1的概率：p(c1)
        p0_num = np.ones(num_doc_words, dtype=float) # 拉普拉斯修正，此处简化，将分子初始值设为1，分母为2
        p1_num = np.ones(num_doc_words, dtype=float)
        p0_denom = 2.0
        p1_denom = 2.0
        for i in range(num_train_doc):
            if trainCategory[i] == 1:
                p1_num += trainmatrix[i] # 整个向量相加 =》向量每个单词相加
                p1_denom += sum(trainmatrix[i]) # 类别1中，单词表中所有单词总词数
            else:
                p0_num += trainmatrix[i]
                p0_denom += sum(trainmatrix[i])
        p1_vec = list(map(log,p1_num/p1_denom)) # 类别1中每个单词/类别1中词典总词数 = 类别1时每个单词出现的概率 p(w|c1)
        p0_vec = list(map(log,p0_num/p0_denom)) # p(w|c0) 注意，只有array才能对每个元素进行操作
        model = {
            'p0_vec': list(p0_vec),
            'p1_vec': list(p1_vec),
            'p': p_abusive
        }
        with open('models/bayes_model_2.json', 'w') as f:
            json.dump(model, f)
        return np.array(p0_vec), np.array(p1_vec), p_abusive

    def classify(self, input, p0_vec, p1_vec, pc):
        # input_vec = np.array(self.word2vec(input)) # 根据输入文档得到词向量
        p1 = sum(input*p1_vec) + log(pc) # 只有array才能相乘
        p0 = sum(input*p0_vec) + log(1-pc)
        return 1 if p1 > p0 else 0

    def test(self, input):
        trainmatrix = self.word2mat(self.data_set)
        trainCategory = self.class_vec
        p0_vec, p1_vec, p1 = self.train_nb(trainmatrix, trainCategory)
        input = self.word2vec(input)
        print(self.classify(input, p0_vec, p1_vec, p1))


if __name__ == '__main__':
    # list_, class_ = load_DataSet()
    # voc = create_Vocalblist(list_)
    # # print(voc)
    # # print(word2vec(voc, list_[0]))
    # mat = word2mat(voc, list_)
    # # print(mat)
    # print(train_nb(mat, class_))
    # print(classify_nb(['love','peace']))
    nb = NaiveBayes()
    nb.test(['love','my','dalmation'])
