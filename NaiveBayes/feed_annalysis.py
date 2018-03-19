import feedparser
from spam_filter import SpamFilter
import re, json

class FeedAnalysis(SpamFilter):

    def __init__(self, feed1, feed0, load_model):

        self.load_model = load_model
        if not load_model:
            print("train model")
            self.data_set, self.class_vec = self.load_DataSet(feed1, feed0)
            self.voc = self.create_Vocalblist()
            self.test_set, self.test_class, self.train_set, self.train_class = self.choose_set()

    def calc_most_freq(self, voc, fulltext):
        freq_dict = dict()
        for w in voc:
            freq_dict[w] = fulltext.count(w)
        sorted_freq = sorted(freq_dict.items(), key=lambda x:x[1], reverse=True)
        return sorted_freq[:30]

    def load_DataSet(self, feed1, feed0):
        data_set = []
        class_list = []
        source_map = {json.dumps(feed1):1, json.dumps(feed0):0}
        length = min(len(feed1['entries']), len(feed0['entries']))
        for s in source_map:
            for i in range(length):
                text = json.loads(s)['entries'][i]['summary']
                parsed_list = self.parse_text(text)
                data_set.append(parsed_list)
                class_list.append(source_map[s])
        return data_set, class_list

    def create_Vocalblist(self):
        # 去除出现频率最高的30个词
        vocablist = list()
        for doc in self.data_set:
            vocablist.extend(list(doc))
        voc_set = list(set(vocablist)) # set是无序的，此处顺序会发生改变
        top30_words = self.calc_most_freq(voc_set, vocablist)
        for w in top30_words:
            voc_set.remove(w[0])
        with open('models/voc.json', 'w') as f:
            json.dump(voc_set, f)
        return voc_set

    def get_top_words(self):
        if not self.load_model:
            voc = self.voc
            p0_vec, p1_vec, p_spam = self.train_nb(self.train_set, self.train_class)
            top_sf = [(voc[i], p0_vec[i]) for i in range(len(p0_vec)) if p0_vec[i] > -4.8]
            top_ny = [(voc[i], p1_vec[i]) for i in range(len(p1_vec)) if p1_vec[i] > -4.8]
        else:
            voc = json.load(open('models/voc.json', 'r'))
            with open('models/bayes_model_2.json', 'r') as f:
                model = json.load(f)
            top_sf = [(voc[i],model['p0_vec'][i]) for i in range(len(model['p0_vec'])) if model['p0_vec'][i] > -4.8]
            top_ny = [(voc[i],model['p1_vec'][i]) for i in range(len(model['p1_vec'])) if model['p1_vec'][i] > -4.8]
        sorted_sf = sorted(top_sf, key=lambda x: x[1], reverse=True)
        for w in sorted_sf:
            print(w[0])
        print("SF----------SF-----------SF-------------SF")

        sorted_ny = sorted(top_ny, key=lambda x: x[1], reverse=True)

        for w in sorted_ny:
            print(w[0])
        print("NY----------NY-----------NY-------------NY")


if __name__ == '__main__':
    # feed1 = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    # feed0 = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    with open('models/feed1.json', 'r') as f:
        feed1 = json.load(f)
    with open('models/feed0.json', 'r') as f:
        feed0 = json.load(f)
    fa = FeedAnalysis(feed1, feed0, True)
    # fa.sample_test()
    fa.get_top_words()
