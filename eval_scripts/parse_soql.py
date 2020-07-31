############################################################
# Assumptions:
# > The SOQL query is correct
# > All fields are accompanied by the sObject
# > Only child to parent relationships are included
# > Aliases are not supported yet
#
# val: number(float) / string(str) / soql(dict)
# field_unit: (agg_id, object_id, field_id)
# object_unit: (object_type, field_unit)
# cond_unit: (not_op, op_id, field_unit, val)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# soql = {
#     'select': [field_unit1, field_unit2, ...]
#     'from': {'object_unit': object_unit}
#     'where': condition
#     'group_by': [field_unit1, field_unit2, ...]
#     'order_by': ('asc'/'desc', [field_unit1, field_unit2, ...])
#     'limit': None/limit_value
#     'nulls': None/'first'/'last'
# }
############################################################

import json
from nltk import word_tokenize

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'nulls')

WHERE_OPS = ('not', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'includes', 'excludes')
AGG_OPS = ('none', 'count', 'count_distinct', 'max', 'min', 'sum', 'avg')
TABLE_TYPE = {
    'child': "child",
    'parent': "parent"
}

COND_OPS = ('and', 'or')
ORDER_OPS = ('desc', 'asc')
NULLS_OPS = ('first', 'last')


class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema):
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema):
        idMap = {'': '__all__'}
        id = 1
        for key, vals in schema.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = "__" + key.lower() + "." + val.lower() + "__"
                id += 1

        for key in schema:
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1

        return idMap


def get_schema(fpath):
    with open(fpath, 'r') as f:
        data = json.load(f)

    schema = {}
    db_id = data['db_id']
    column_names_original = data['column_names_original']
    table_names_original = data['table_names_original']

    for i, table_name in enumerate(table_names_original):
        table = str(table_name.lower())
        cols = [str(col.lower()) for t_id, col in column_names_original if t_id == i]
        schema[table] = cols


    return Schema(schema)


def tokenize(string):
    string = str(string)
    string = string.replace("\'", "\"")
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    vals = {}
    for i in range(len(quote_idxs) - 1, -1, -2):
        idx1 = quote_idxs[i - 1]
        idx2 = quote_idxs[i]
        val = string[idx1:idx2 + 1]
        key = "__val_{}_{}__".format(idx1, idx2)
        string = string[:idx1] + key + string[idx2 + 1:]
        vals[key] = val
    
    toks = [word.lower() for word in word_tokenize(string)]

    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]
    
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx - 1]
        if pre_tok in prefix:
            toks = toks[:eq_idx - 1] + [pre_tok + "="] + toks[eq_idx + 1:]
    
    return toks


def parse_col(toks, start_idx, schema, table_name):
    tok = toks[start_idx]
    
    if '.' in tok:
        assert tok.count('.') == 1
        key = '.'.join([table_name, tok])
        return start_idx + 1, schema.idMap[key]
    
    assert table_name is not None

    if tok in schema.schema[table_name]:
        key = table_name + '.' + tok
        return start_idx + 1, schema.idMap[key]
    
    assert False, "Error col: {}".format(toks[start_idx])


def parse_col_unit(toks, start_idx, schema, table_name):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1
        if toks[idx] == ')':
            isBlock = False
            idx += 1
            agg_id = AGG_OPS.index("none")
            col_id = schema.idMap['']
            return idx, (agg_id, col_id)

    
    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1
        if toks[idx] != ')':
            idx, col_id = parse_col(toks, idx, schema, table_name)
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        return idx, (agg_id, col_id)
    
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, schema, table_name)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1
    
    return idx, (agg_id, col_id)


def parse_table_unit(toks, start_idx, schema):
    idx = start_idx
    len_ = len(toks)
    key = toks[idx]
    idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, schema, table_name=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] == 'select':
        idx, val = parse_soql(toks, idx, schema)
    
    elif "\"" in toks[idx]:
        val = toks[idx]
        idx += 1
    
    

    elif toks[idx] in ('true', 'false'):
        val = toks[idx]
        idx += 1
    
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except:
            end_idx = idx
            while end_idx < len_ and toks[end_idx] not in (',', ')') \
                and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS:
                    end_idx += 1
            
            idx, val = parse_col_unit(toks[start_idx:end_idx], 0, schema, table_name)
            idx = end_idx
    
    if isBlock:
        assert toks[idx] == ')'
        idx += 1
    
    return idx, val


def parse_condition(toks, start_idx, schema, table_name=None):
    idx = start_idx
    len_ = len(toks)
    conds = []

    while idx < len_:
        idx, col_unit = parse_col_unit(toks, idx, schema, table_name)
        not_op = False
        if toks[idx] == 'not':
            not_op = True
            idx += 1
        
        assert idx < len_ and toks[idx] in WHERE_OPS
        op_id = WHERE_OPS.index(toks[idx])
        idx += 1
        val = None
        idx, val = parse_value(toks, idx, schema, table_name)
        
        conds.append((not_op, op_id, col_unit, val))

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (')', ';')):
            break
        
        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1
    
    return idx, conds


def parse_select(toks, start_idx, schema, table_name):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == 'select'

    idx += 1
    col_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
            assert toks[idx] == '('
        idx, col_unit = parse_col_unit(toks, idx, schema, table_name)
        col_units.append((agg_id, col_unit))
        if idx < len_ and toks[idx] == ',':
            idx += 1
    
    return idx, col_units


def parse_from(toks, start_idx, schema):
    assert 'from' in toks[start_idx:]

    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    idx, table_unit, table_name = parse_table_unit(toks, idx, schema)
    return idx, table_unit, table_name


def parse_where(toks, start_idx, schema, table_name):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []
    
    idx += 1
    idx, conds = parse_condition(toks, idx, schema, table_name)
    return idx, conds


def parse_group_by(toks, start_idx, schema, table_name):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units
    
    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (')', ';')):
        idx, col_unit = parse_col_unit(toks, idx, schema, table_name)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1
        else:
            break
        
    return idx, col_units

def parse_order_by(toks, start_idx, schema, table_name):
    idx = start_idx
    len_ = len(toks)
    col_units = []
    order_type = 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, col_units
    
    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, schema, table_name)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1
        else:
            break

    return idx, (order_type, col_units)


def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 1
        return idx + 1, int(toks[idx])
    
    return idx, None


def parse_nulls(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'nulls':
        idx += 1
        return idx + 1, toks[idx]
    
    return idx, None


def parse_soql(toks, start_idx, schema):
    isBlock = False
    len_ = len(toks)
    idx = start_idx

    soql = {}
    if toks[idx] == '(':
        isBlock = True
        idx += 1
    
    from_end_idx, table_unit, table_name = parse_from(toks, idx, schema)
    soql['from'] = {'table_name': table_name, 'table_unit': table_unit}

    _, select_col_units = parse_select(toks, idx, schema, table_name)
    idx = from_end_idx
    soql['select'] = select_col_units

    idx, where_conds = parse_where(toks, idx, schema, table_name)
    soql['where'] = where_conds

    idx, group_col_units = parse_group_by(toks, idx, schema, table_name)
    soql['group_by'] = group_col_units

    idx, order_col_units = parse_order_by(toks, idx, schema, table_name)
    soql['order_by'] = order_col_units

    idx, nulls_type = parse_nulls(toks, idx)
    soql['nulls'] = nulls_type

    idx, limit_val = parse_limit(toks, idx)
    soql['limit'] = limit_val

    if isBlock:
        assert toks[idx] == ')'
        idx += 1
    
    return idx, soql


def get_soql(schema, query):
    toks = tokenize(query)
    _, soql = parse_soql(toks, 0, schema)

    return soql


def load_data(fpath):
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data
