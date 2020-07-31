import os
import json
import re
from collections import defaultdict


with open('annotations.json', 'r') as f:
    annotations = json.load(f)

questions = defaultdict(list)
queries = []

for annotation in annotations:
    from_cols = re.findall('FROM ([\w]+)', annotation['query'])
    assert len(from_cols) == 1
    from_col = from_cols[0]
    assert from_col == annotation['sObject']
    questions[annotation['question']].append(annotation)
    assert len(questions[annotation['question']]) == 1, '{}'.format(annotation['question'])
    queries.append(annotation['query'])

with open('dev_gold.txt', 'w') as f:
    for query in queries:
        f.write(query + '\t' + 'salesforce' + '\n')
