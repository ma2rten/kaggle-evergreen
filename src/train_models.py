"""
Train all the individual models
"""

from util import *
from feature_sets import get_featureset
from data import get_labels

from DATASETS import DATASETS, LDA_DATASETS

from sklearn.cross_validation import KFold
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.linear_model import LogisticRegression

import gensim


def main():
    scores = load('scores', {})

    PREPROCESSING = {
        'raw': lambda: get_featureset(dataset, tf_idf=False),
        'tfidf': lambda: get_featureset(dataset),
        'nostem': lambda: get_featureset(dataset, stem=False),
        'svd50': lambda: SVD(50).fit_transform(get_featureset(dataset)),
        'svd100': lambda: SVD(100).fit_transform(get_featureset(dataset)),
        'lda': lambda: get_lda(dataset, 100),
    }

    for dataset in DATASETS:
        for method in PREPROCESSING:
            # name of feature set
            name = method + ':' + dataset

            # check if we calulated this score before
            if name in scores:
                continue

            # we can only do iton some datasets
            if method == 'lda':
                if 'dataset' not in LDA_DATASETS:
                    continue

            print name

            # get the data
            data = PREPROCESSING[method]()

            # train model
            scores[name] = get_scores(data)

            # save it
            save('scores', scores)


def get_scores(data):
    labels = get_labels()

    scores = []

    for train_idx, test_idx in KFold(len(labels), 10):
        score = predict(data[train_idx], labels[train_idx], data[test_idx])
        scores.append(score)

    score = predict(data[:len(labels)], labels, data[len(labels):])
    scores.append(score)

    return np.hstack(scores)


def predict(train, labels, test):
    model = LogisticRegression()
    model.fit(train, labels)

    return model.predict_proba(test)[:, -1]


def get_lda(name, n):
    data = get_featureset(name, tf_idf=False, norm=None, stem=False,
                          stopwords=True)

    corpus = gensim.matutils.Scipy2Corpus(data)
    lda = LdaModel(corpus, num_topics=n)
    data = gensim.matutils.corpus2csc(lda[corpus]).T

    return data


if __name__ == "__main__":
    main()
