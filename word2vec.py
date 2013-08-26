#!/usr/bin/env python2.7

import numpy as np
from collections import Counter
from progressbar import ProgressBar, ETA, Bar

NORMALIZE_VECTORS = True

def _read_until(fd, sentinal):
    result = ""
    while True:
        tmp = fd.read(1)
        if tmp == sentinal:
            return result
        result += tmp

def load_weights(filename):
    weights = dict(line.strip().split() for line in open(filename))
    for word, c in weights.iteritems():
        weights[word] = int(c)
    return weights

def load_vector(filename):
    with open(filename, "rb") as fd:
        vocab_size, layer1_size = map(int, fd.readline().strip().split())
        vectors = {}
        pbar = ProgressBar(maxval=vocab_size, widgets=["Loading Vectors", Bar(), ETA()]).start()
        for i in pbar(xrange(vocab_size)):
            try:
                word = _read_until(fd, " ")
            except EOFError:
                break
            vector = np.fromfile(fd, dtype=np.float32, count=layer1_size)

            # clear out trailing newline
            fd.read(1) 

            # normalize the vector
            if NORMALIZE_VECTORS:
                vector /= np.sqrt(vector.dot(vector))

            vectors[word] = vector
        return vectors

def cosdistance(A, B):
    A_norm = B_norm = 1
    if not NORMALIZE_VECTORS:
        A_norm = np.sqrt(np.sum(A * A))
        B_norm = np.sqrt(np.sum(B * B))
    if A_norm == 0 or B_norm == 0:
        return -1.0 * np.Infinity
    return np.dot(A, B) / (A_norm * B_norm)

def closest_vector(needle, haystack, N=1):
    if N == 1:
        return list(max(haystack.iteritems(), key=lambda (key, vector) : cosdistance(needle, vector)))
    else:
        closest = Counter({word:cosdistance(needle, vector) for word, vector in haystack.iteritems()})
        if not N:
            return closest.items()
        else:
            return closest.most_common(N)

def analogy(A, B, vectors, N=1):
    y = vectors[A[1]] - vectors[A[0]]

    if B[0] is not None:
        y += vectors[B[0]]
    if B[1] is not None:
        y -= vectors[B[1]]

    y /= np.sqrt(y.dot(y))

    words = A + B
    # We ask for 4 more results than we should because we filter out the words
    # from the query... this is trimmed out with the final fancy list slice
    return [(k,v) for k,v in closest_vector(y, vectors, N+4) if k not in words][:N]
