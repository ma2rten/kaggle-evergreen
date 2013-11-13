import re, datetime, json

from unidecode import unidecode
from boilerpipe.extract import Extractor
from bs4 import BeautifulSoup

from util import *
from data import get_test, get_train


# html tags of interest
TAGS = ['title', 'h1', 'h2', 'h3', 'meta-description', 'meta-keywords',
        'img', 'a', 'other']


def main():
    data = get_train() + get_test()

    f = file('generated/extracted_text', 'w')

    for i, item in enumerate(data):
        # status update
        if (i % 500) == 0:
            print i, datetime.datetime.now().time()

        #  parse file
        data = {}
        soup = boil_soup(item['urlid'])

        # given boilerplate
        data['boilerplate'] = [item['title'], item['body']]

        # extract text
        extractor = Extractor(extractor='ArticleExtractor', html=unicode(soup))
        data['boilerpipe'] = [extractor.getText()]

        # remove non-text tags
        for tag in ['script', 'style']:
            for el in soup.find_all(tag):
                el.extract()

        # extract text for each tag
        for tag in TAGS:
            items = []
            for el in soup.find_all(tag):
                el.extract()

                if tag == 'img':
                    try:
                        items.append(el['alt'])
                    except KeyError:
                        pass
                    try:
                        items.append(el['title'])
                    except KeyError:
                        pass
                else:
                    items.append(el.text)

            data[tag] = items

        # extract meta tags
        meta = soup.find_all('meta')
        for el in meta:
            prop = el.get('property') if el.get('property') else el.get('name')
            if not prop:
                continue
            prop = prop.lower()
            try:
                s = unicode(el['content'])
            except:
                continue

            data['meta-'+prop] = s.split(u',') if prop == 'keywords' else [s]

        # preprocess string
        for item in data:
            data[item] = map(clean_string, data[item])
            data[item] = filter(None, data[item])

        print >>f, json.dumps(data)

    f.close()


def clean_string(s):
    s = unicode(s)
    s = unidecode(s).lower()
    s = re.sub(r"\s+", ' ', s)
    return s.strip()


def boil_soup(urlid, parser="lxml"):
    filename = 'data/raw_content/' + urlid

    with file(filename, 'rb') as f:
        html = f.read()

        for parser in ["lxml", "xml", "html5lib"]:
            soup = BeautifulSoup(html, parser)
            if soup.body:
                return soup

        return BeautifulSoup(html)


if __name__ == "__main__":
    main()
