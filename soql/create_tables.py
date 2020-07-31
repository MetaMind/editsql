from collections import defaultdict
import os
import re
import sys
import json
from nltk import tokenize, word_tokenize



def create_database_schemas(db_dir, table_schemas):
    for root, dirs, files in os.walk(db_dir):
        if 'schema.txt' in files:
            table_names_original = set()
            table_names = []
            column_names_original = []
            column_names = []
            column_types = []

            with open(os.path.join(root, 'schema.txt'), 'r') as f:
                for line in f:
                    line = line.strip().split(' ')
                    assert len(line) == 3
                    table_names_original.add(line[0])
            table_names_original = list(table_names_original)

            with open(os.path.join(root, 'schema.txt'), 'r') as f:    
                for line in f:
                    line = line.strip().split(' ')
                    assert len(line) == 3

                    table_id = table_names_original.index(line[0])

                    column_names_original.append([table_id, line[1]])
                    tks = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', line[1].replace('__c', ''))).replace('_', ' ').split()
                    tks = ' '.join(tks).lower().replace('.', '')
                    column_names.append([table_id, tks])
                    column_types.append(line[2])
                
                db_name = root.split('/')[-1]
                
                for table_name in table_names_original:
                    tks = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', table_name.replace('__c', ''))).replace('_', ' ').split()
                    tks = ' '.join(tks).lower()
                    table_names.append(tks)
                
                
                schema = {
                    'db_id': db_name,
                    'table_names_original': table_names_original,
                    'table_names': table_names,
                    'column_names_original': column_names_original,
                    'column_names': column_names,
                    'column_types': column_types
                }

                table_schemas.append(schema)
                with open(os.path.join(root, 'database.json'), 'w') as db_schema_file:
                    json.dump(schema, db_schema_file, sort_keys=True, indent=2, separators=(',', ': '))
    return table_schemas


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ('Usage: python create_tables.py database/ tables.json')
        sys.exit()
    input_dir = sys.argv[1]
    tables_file = sys.argv[2]
    
    table_schemas = []
    table_schemas = create_database_schemas(input_dir, table_schemas)

    with open(tables_file, 'w') as output_file:
        json.dump(table_schemas, output_file, sort_keys=True, indent=2, separators=(',', ': '))
