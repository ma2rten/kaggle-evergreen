import json, re

from util import *
from collections import Counter
from operator import itemgetter
from sklearn.feature_extraction import DictVectorizer
from parse_html import clean_string, TAGS


def main():
    # load data
    path = 'generated/extracted_text'
    data = map(json.loads, file(path))

    # count word for every tag
    tags = TAGS + ['boilerplate', 'boilerpipe']
    counts_per_tag = {}

    for tag in tags:
        counts = map(count, get(tag, data))
        counts_per_tag[tag] = counts

    total = sum_up(counts_per_tag, len(data))

    # vectorize
    v = DictVectorizer()
    v.fit([total])

    features = {}
    for tag in tags:
        features[tag] = v.transform(counts_per_tag[tag])

    save('text_features', features)
    save('text_vectorizer', v)


def tokenize(string):
    string = re.sub(r'[0-9]', '0', string)
    words = re.split(r'\W+', string)

    return map(None, words)


def count(texts):
    counter = Counter()

    for text in texts:
        words = tokenize(text)
        counter.update(words)

    return counter


def get(tag, items):
    for item in items:
        yield item[tag] if (tag in item) else []

    total = Counter()


def sum_up(counts, n):
    tags = list(counts)
    total = Counter()

    for i in xrange(n):
        words = set()
        for tag in tags:
            words.update(set(counts[tag][i]))

        total.update(words)

    return total

if __name__ == "__main__":
    main()
