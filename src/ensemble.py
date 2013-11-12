from util import *
from data import get_labels

from sklearn.metrics import roc_auc_score
from scipy.optimize import nnls
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import KFold

def auc(s):
    labels = get_labels()
    v = roc_auc_score(labels, s[:len(labels)])
    return round(v, 5)

def weighted(data, labels):
    weights, _ = nnls(data[:len(labels)], labels)
    return data.dot(weights)

def weight_selected(data, labels):
    weights, _ = nnls(data[:len(labels)], labels)
    return data[:,weights > 0].mean(axis=1)

def main():
    scores = load('scores')
    labels = get_labels()

    scores_with_raw = np.vstack(scores.values()).T
    scores_without_raw = np.vstack([scores[n] for n in scores if ('raw:' not in n)]).T

    print 'Best Model:',
    print max([(auc(scores[name]), name) for name in scores])
    print 
    print auc(scores_with_raw.mean(axis=1)),
    print 'Simple Average'
    print auc(weighted(scores_with_raw, labels)),
    print 'Weighted'
    print auc(weight_selected(scores_with_raw, labels)),
    print 'Weight selected'
    print 
    print auc(scores_without_raw.mean(axis=1)),
    print 'Simple Average (without raw)'
    print auc(weighted(scores_without_raw, labels)),
    print 'Weighted (without raw)'
    print auc(weight_selected(scores_without_raw, labels)),
    print 'Weight selected (without raw)'
    print 

    final = weight_selected(scores_without_raw, labels)
    submit(final[len(labels):])

if __name__ == "__main__":
    main()
