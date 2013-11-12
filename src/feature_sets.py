import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from scipy import sparse
from scipy.sparse import dok_matrix
from stemming.porter2 import stem
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from util import *


def get_featureset(name, stem=True, tf_idf=True, stopwords=False, norm='l2', use_idf=1, smooth_idf=1, sublinear_tf=1, binary=False):
    data = parse(name)

    if stopwords:
        data = data * load('stopword_matrix', make_stopword_matrix)
    if stem:
        data = data * load('stem_matrix', make_stem_matrix)
    if tf_idf:
        data = TfidfTransformer(use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf).fit_transform(data)
    if norm:
        normalize(data, norm, copy=False)
    if binary:
        data.data[:] = 1

    return data

def make_stopword_matrix():
    words = load('text_vectorizer').get_feature_names()

    matrix = sparse.eye(len(words), format='dok')
    for word_id, word in enumerate(words):
        if word in ENGLISH_STOP_WORDS:
            matrix[word_id,word_id] = 0

    return matrix.tocsr()

def make_stem_matrix():
    words = load('text_vectorizer').get_feature_names()

    # stem all words
    stems = defaultdict(list)
    for word_id, word in enumerate(words):
        stems[stem(word)].append(word_id)

    # make matrix
    matrix = dok_matrix( (len(words), len(stems)) )

    for stem_id, s in enumerate(stems):
        for word_id in stems[s]:
            matrix[word_id, stem_id] = 1.

    return matrix.tocsr()

def parse(s):
    # remove spaces if any
    if ' ' in s:
        s = s.replace(' ', '')
    return _parse(s).copy()

def _apply(fun, items):
    assert len(items) > 0
    return reduce(fun, items[1:], items[0])

def _parse(s):
    text = load('text_features')
    function = re.compile(r'^(\w*)\(([^)]*)\)$')

    plus = lambda x,y: x+y
    times = lambda x,y: x*y

    # replace some strings
    if s == 'body':
        s = 'h1+h2+h3+img+a+other'
    elif s == 'other':
        s = 'body'

    # apply functions
    if function.match(s):
        name, param = function.match(s).group(1, 2)

        if param == 'all':
            param = ','.join(text)
        items = map(_parse, param.split(','))
        return _apply({'max': maximum, 'sum': plus}[name], items)

    # addition and multplicatoin
    if '+' in s:
        items = map(_parse, s.split('+'))
        return _apply(plus, items)

    if '*' in s:
        items = map(_parse, s.split('*'))
        return _apply(times, items)

    # try to parse any numbers
    try:
        return float(s)
    except ValueError:
        pass

    # return corresponding dataset
    return text[s]
