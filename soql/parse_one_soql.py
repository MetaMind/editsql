import os
import json
from parse_soql import *


soql = "SELECT count FROM Account WHERE Type WHERE Name = 'bharat'"
db_id = "salesforce"
table_file = "tables.json"

db_schema_filename = os.path.join('database', db_id, 'database.json')
schema = get_schema(db_schema_filename)

soql = get_soql(schema, soql)

print (soql)