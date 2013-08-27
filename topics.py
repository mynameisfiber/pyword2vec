import word2vec as wv
from sklearn import cluster
import numpy as np
from nltk.corpus import stopwords

import time
import re
from operator import itemgetter
from collections import defaultdict
from itertools import tee
import cPickle

from pprint import pprint
import logging

logging.basicConfig(level=logging.INFO)
stop_words = set(stopwords.words("english"))

if not hasattr(globals(), "profile"):
    def profile(fxn):
        return fxn

def iter_pairs(dom):
    a,b = tee(dom, 2)
    b.next()
    while True:
        yield a.next(), b.next()

class TopicModel(object):
    def __init__(self, vectors, n_clusters, word_weights=None, kmean_parameters={}):
        self.vectors = vectors
        self.n_clusters = n_clusters
        self.word_weights = word_weights or defaultdict(lambda x : 1)
        self._kmean_parameters = {"n_clusters": n_clusters, "compute_labels": True}
        self._kmean_parameters.update(kmean_parameters)
        self._kmean = cluster.MiniBatchKMeans(**self._kmean_parameters)
        self.word_to_cluster = None

    def train(self, remove_stopwords=True):
        start = time.time()
        logging.info("Starting training")

        if remove_stopwords:
            for word in stop_words:
                self.vectors.pop(word, None)

        words, bare_vectors = zip(*self.vectors.iteritems())
        bare_vectors = np.asarray(bare_vectors)
        self._kmean.fit(bare_vectors)
        self.word_to_cluster = dict(zip(words, self._kmean.labels_))
        logging.info("Training complete: %fs", time.time() - start)

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

    @profile
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
    text, nchanges = re.subn(r"([a-z])[^a-z ]+([a-z])", r"\1\2", text)
    text, nchanges = re.subn(r"[^a-z ]+", r" ", text)
    text = " ".join(word for word in text.split(" ") if word and word not in stop_words)
    return text

if __name__ == "__main__":
    articles = []
    for i in range(4):
        content = open("/mnt/data/test/article-%d.txt" % i, "r").read()
        articles.append(clean_text(content))

    vectors = wv.load_vector("/mnt/data/wiki_data/vectors_phrases-size:200-window:5.bin")
    weights = wv.load_weights("/mnt/data/wiki_data/wiki_phrase_vocab_20130805")

    tm = TopicModel(vectors, 45, weights, kmean_parameters={"batch_size" : 500, "n_init" : 100})
    tm.train()

    for i, article in enumerate(articles):
        result = tm.classify_text(article)
        print "Article %d - category %d" % (i, result[-1][0]) 
