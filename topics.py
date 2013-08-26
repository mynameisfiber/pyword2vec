import word2vec as wv
from sklearn import cluster
import numpy as np

import re
from operator import itemgetter
from collections import defaultdict
from itertools import tee
import cPickle

from pprint import pprint
import logging

logging.basicConfig(level=logging.INFO)

def iter_pairs(dom):
    a,b = tee(dom, 2)
    b.next()
    while True:
        yield a.next(), b.next()

class TopicModel(object):
    def __init__(self, vectors, n_clusters, word_weights=None):
        self.vectors = vectors
        self.n_clusters = n_clusters
        self.word_weights = word_weights or defaultdict(lambda x : 1)
        self._kmean = cluster.MiniBatchKMeans(n_clusters=n_clusters, compute_labels=True)
        self.word_to_cluster = None

    def train(self):
        logging.info("Starting training")
        words, bare_vectors = zip(*self.vectors.iteritems())
        bare_vectors = np.asarray(bare_vectors)
        self._kmean.fit(bare_vectors)
        self.word_to_cluster = dict(zip(words, self._kmean.labels_))
        logging.info("Training complete")

    def save(self, fd):
        cPickle.dump(self, fd, -1)

    @classmethod
    def load(cls, fd):
        return cPickle.load(fd)

    def classify_word(self, word):
        return self.word_to_cluster.get(word)

    def _score_words(self, words):
        N = len(words)
        weights = [self.word_weights[w] for w in words if self.word_weights.get(w)]
        if len(weights):
            score_boost = np.log1p(sum(weights) / float(len(weights)))
            return N / score_boost
        else:
            return 0

    def classify_text(self, text):
        logging.info("Starting text classification")
        # TODO: right now we only consider a word from an article ONCE
        # regardless of how many times it shows up... this should change
        word_to_cluster = {}

        for word in text.split():
            cluster = self.classify_word(word)
            if cluster:
                word_to_cluster[word] = cluster
        for w1, w2 in iter_pairs(text.split()):
            word = w1 + "_" + w2
            cluster = self.classify_word(word)
            if cluster:
                word_to_cluster[word] = cluster

        cluster_to_word = defaultdict(list)
        for word, cluster in word_to_cluster.iteritems():
            cluster_to_word[cluster].append(word)

        cluster_scored = map(lambda x : (x[0], self._score_words(x[1])), cluster_to_word.iteritems())
        cluster_scored.sort(key=itemgetter(1))
        logging.critical("Classification complete")
        return cluster_scored


def clean_text(text):
    text = text.lower()
    #for num, num_word in ((1, " one "), (2, " two "), (3, " three "), (4, " four "), (5, " five "), (6, " six "), (7, " seven "), (8, " eight "), (9, " nine "), (0, " zero ")):
    #    text.replace(str(num), num_word)
    text, nchanges = re.subn(r"([a-z])[^a-z ]+([a-z])", r"\1\2", text)
    text, nchanges = re.subn(r"[^a-z ]+", r" ", text)
    text, nchanges = re.subn(r"[ ]+", r" ", text)
    return text

if __name__ == "__main__":
    article = clean_text(open("/mnt/data/train/article1", "r").read())

    try:
        tm = TopicModel.load(open("topicmodel.pkl", "w+"))
    except:
        weights = wv.load_weights("/home/micha/data/wiki_data/vectors_phrases-size:200-window:5.bin")
        vectors = wv.load_vector("/home/micha/data/wiki_data/wiki_phrase_vocab_20130805")

        tm = TopicModel(vectors, 45, weights)
        tm.train()
        tm.save(open("topicmodel.pkl", "w+"))
    classification = tm.classify_text(article)

    print "Topic weights:"
    pprint(classification)
