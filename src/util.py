import numpy as np

_cache = {}  # in-memory cache for load function


def file(filename, mode='r'):
    """ open file with project root as base (not current dictionary) """
    import os
    try:
        name = os.path.dirname(__file__) + '/../' + filename
        return open(name, mode)

    except IOError:
        return open('../'+filename, mode)


def save(name, data):
    """ save object to disk for caching """
    import cPickle as pickle

    with file('generated/'+name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load(name, default=None, cache=True):
    """ load object cached from disk """
    import cPickle as pickle

    # check if data is already in memory cache
    global _cache

    if name in _cache:
        return _cache[name]

    # try to load data
    try:
        with file('generated/'+name, 'rb') as f:
            data = pickle.load(f)

    # fallback to default
    except IOError:
        if hasattr(default, '__call__'):
            default = default()
            save(name, default)
        return default

    # cache in memory
    if cache:
        _cache[name] = data

    return data


def submit(predictions):
    """ generate final submission file """
    test = load('test')
    assert len(predictions) == len(test)
    f = file('submission.csv', 'w')
    f.write('urlid,label\n')
    for item, p in zip(test, predictions):
        f.write(item['urlid']+','+str(p)+'\n')
    f.close()


def sparse_maximum(A, B):
    """
    like np.maximum for sparse matrices
    note: may not handle nans well
    """
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)


def maximum(A, B):
    """
    like np.maximum, but works with both sparse and dense matrices
    note: may not handle nans well for sparse matrices
    """
    from scipy.sparse import issparse, csr_matrix

    if issparse(A) or issparse(B):
        return sparse_maximum(csr_matrix(A), csr_matrix(B))
    else:
        return np.maximum(A, B)
