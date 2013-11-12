import csv, json, re
from collections import defaultdict, Counter

from unidecode import unidecode
from util import *

def extract_domain(url):
    # extract domains
    domain = url.lower().split('/')[2]
    domain_parts = domain.split('.')
        
    # e.g. co.uk 
    if domain_parts[-2] not in ['com', 'co']:
        return '.'.join(domain_parts[-2:])
    else:
        return '.'.join(domain_parts[-3:])

def load_data(filename):
    csv_file_object = csv.reader(file(filename, 'rb'), delimiter='\t') 

    header = csv_file_object.next()

    data=[]

    for row in csv_file_object:
        # make dictionary
        item = {}
        for i in range(len(header)):
            item[header[i]] = row[i]
        
        # url
        item['real_url'] = item['url'].lower()
        item['domain'] = extract_domain(item['url'])
        item['tld'] = item['domain'].split('.')[-1]

        # parse boilerplate
        boilerplate = json.loads(item['boilerplate'])
        for f in ['title', 'url', 'body']:
            item[f] = boilerplate[f] if (f in boilerplate) else u''
            item[f] = unidecode(item[f]) if item[f] else ''
        
        del item['boilerplate']

        # label
        if 'label' in item:
            item['label'] = item['label'] == '1'
        else:
            item['label'] = '?'

        data.append(item)

    return data

def get_train():
    return load('train', lambda: load_data('data/train.tsv'))

def get_test():
    return load('test', lambda: load_data('data/test.tsv'))

def get_labels():
    return np.array([item['label'] for item in get_train()])
