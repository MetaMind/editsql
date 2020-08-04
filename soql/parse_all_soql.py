import os
import json
import re
import traceback
from parse_soql import *
import random


def random_split():
    if random.random() < 0.98:
        return '_1'
    else:
        return '_2'


def replace_value(query):
    query = re.sub("\'[^']+\'", 'value', query)
    # query = re.sub('true|false', 'value', query)
    query = re.sub(r'\b([0-9]+.[0-9]+)\b', 'value', query)
    query = re.sub(r'\b([0-9])\b', 'value', query)
    
    return query.lower()


with open('annotations.json', 'r') as f:
    annotations = json.load(f)

db_id = 'salesforce'
dev = []
train = []

for annotation in annotations:
    database_id = db_id + random_split()
    annotation['query'] = annotation['query'].replace('count()', 'count(id)')
    soql = annotation['query']
    db_file = os.path.join('database', database_id, 'database.json')
    schema = get_schema(db_file)
    try:
        soql = get_soql(schema, soql)
    except KeyError as e:
        print (soql)
        print (e)
        print ('---------')
        continue
    entry = {
        'db_id': database_id,
        'query': annotation['query'],
        'query_toks': word_tokenize(annotation['query'].replace('\'', '\"').lower()),
        'sql': soql,
        'question': annotation['question'],
        'question_toks': word_tokenize(annotation['question'])
    }

    query_no_value = replace_value(annotation['query'])
    entry['query_toks_no_value'] = word_tokenize(query_no_value)

    if database_id[-1] == '1':
        train.append(entry)
    else:
        dev.append(entry)


with open('train.json', 'w') as f:
    json.dump(train, f, indent=2, sort_keys=True, separators=(',', ': '))


with open('dev.json', 'w') as f:
    json.dump(dev, f, indent=2, sort_keys=True, separators=(',', ': '))


with open('dev_gold.txt', 'w') as f:
    for e in dev:
        f.write(e['query'] + '\t' + e['db_id'] + '\n')

with open('train_db_ids.txt', 'w') as f:
    for e in train:
        f.write(e['db_id'] + '\n')

with open('dev_db_ids.txt', 'w') as f:
    for e in dev:
        f.write(e['db_id'] + '\n')
