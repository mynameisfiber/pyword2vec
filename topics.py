import word2vec as wv
from sklearn import cluster
import numpy as np

import re
from operator import itemgetter
from collections import defaultdict
from itertools import tee

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
            cluster_to_word[cluster].append(cluster)

        cluster_scored = map(lambda x : (x[0], self._score_words(x[1])), cluster_to_word.iteritems())
        cluster_scored.sort(key=itemgetter(1))
        logging.critical("Classification complete")
        return cluster_scored


def clean_text(text):
    text, nchanges = re.subn(r"([a-z])[^a-z ]+([a-z])", r"\1\2", text.lower())
    text, nchanges = re.subn(r"[^a-z ]+", r" ", text)
    text, nchanges = re.subn(r"[ ]+", r" ", text)
    return text

if __name__ == "__main__":
    article = clean_text(open("/mnt/data/train/article1", "r").read())
    weights = wv.load_weights("/mnt/data/wiki-article-pages/wiki_vocab_20130805")
    vectors = wv.load_vector("/mnt/data/wiki-article-pages/vectors_phrases-size:600-window:5.bin")

    tm = TopicModel(vectors, 45, weights)
    tm.train()
    classification = tm.classify_text(article)

    print "Topic weights:"
    pprint(classification)
