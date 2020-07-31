import os, sys
import json
import sqlite3
import traceback
import argparse

from parse_soql import *

# Flag to disable value evaluation
DISABLE_VALUE = True
# Flag to disable distinct in select evaluation
DISABLE_DISTINCT = True



HARDNESS = {
    "component1": ('where', 'group', 'order', 'limit', 'join', 'or', 'like'),
    "component2": ('except', 'union', 'intersect')
}


def condition_has_or(conds):
    return 'or' in conds[1::2]


def condition_has_like(conds):
    return WHERE_OPS.index('like') in [cond_unit[1] for cond_unit in conds[::2]]


def condition_has_sql(conds):
    for cond_unit in conds[::2]:
        val1, val2 = cond_unit[3], cond_unit[4]
        if val1 is not None and type(val1) is dict:
            return True
        if val2 is not None and type(val2) is dict:
            return True
    return False



def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')


def accuracy(count, total):
    if count == total:
        return 1
    return 0


def recall(count, total):
    if count == total:
        return 1
    return 0


def F1(acc, rec):
    if (acc + rec) == 0:
        return 0
    return (2. * acc * rec) / (acc + rec)


def get_scores(count, pred_total, label_total):
    if pred_total != label_total:
        return 0,0,0
    elif count == pred_total:
        return 1,1,1
    return 0,0,0


def eval_sel(pred, label):
    pred_sel = pred['select']
    label_sel = label['select']
    label_wo_agg = [unit[1] for unit in label_sel]
    pred_total = len(pred_sel)
    label_total = len(label_sel)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_sel:
        if unit in label_sel:
            cnt += 1
            label_sel.remove(unit)
        if unit[1] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[1])

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_where(pred, label):
    pred_conds = [unit for unit in pred['where'][::2]]
    label_conds = [unit for unit in label['where'][::2]]
    label_wo_agg = [unit[2] for unit in label_conds]
    pred_total = len(pred_conds)
    label_total = len(label_conds)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_conds:
        if unit in label_conds:
            cnt += 1
            label_conds.remove(unit)
        if unit[2] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[2])

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_group(pred, label):
    pred_cols = [unit[1] for unit in pred['group_by']]
    label_cols = [unit[1] for unit in label['group_by']]
    pred_total = len(pred_cols)
    label_total = len(label_cols)
    cnt = 0
    pred_cols = [pred.split(".")[1] if "." in pred else pred for pred in pred_cols]
    label_cols = [label.split(".")[1] if "." in label else label for label in label_cols]
    for col in pred_cols:
        if col in label_cols:
            cnt += 1
            label_cols.remove(col)
    return label_total, pred_total, cnt


def eval_having(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['group_by']) > 0:
        pred_total = 1
    if len(label['group_by']) > 0:
        label_total = 1

    pred_cols = [unit[1] for unit in pred['group_by']]
    label_cols = [unit[1] for unit in label['group_by']]
    if pred_total == label_total == 1 \
            and pred_cols == label_cols:
        cnt = 1

    return label_total, pred_total, cnt


def eval_order(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['order_by']) > 0:
        pred_total = 1
    if len(label['order_by']) > 0:
        label_total = 1
    if len(label['order_by']) > 0 and pred['order_by'] == label['order_by'] and \
            ((pred['limit'] is None and label['limit'] is None) or (pred['limit'] is not None and label['limit'] is not None)):
        cnt = 1
    return label_total, pred_total, cnt


def eval_and_or(pred, label):
    pred_ao = pred['where'][1::2]
    label_ao = label['where'][1::2]
    pred_ao = set(pred_ao)
    label_ao = set(label_ao)

    if pred_ao == label_ao:
        return 1,1,1
    return len(pred_ao),len(label_ao),0





def get_keywords(sql):
    res = set()
    if len(sql['where']) > 0:
        res.add('where')
    if len(sql['group_by']) > 0:
        res.add('group')
    if len(sql['order_by']) > 0:
        res.add(sql['order_by'][0])
        res.add('order')
    if sql['limit'] is not None:
        res.add('limit')

    # or keyword
    ao = sql['where'][1::2]
    if len([token for token in ao if token == 'or']) > 0:
        res.add('or')

    cond_units = sql['where'][::2]
    # not keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[0]]) > 0:
        res.add('not')

    # in keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('in')]) > 0:
        res.add('in')

    # like keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')]) > 0:
        res.add('like')

    return res


def eval_keywords(pred, label):
    pred_keywords = get_keywords(pred)
    label_keywords = get_keywords(label)
    pred_total = len(pred_keywords)
    label_total = len(label_keywords)
    cnt = 0

    for k in pred_keywords:
        if k in label_keywords:
            cnt += 1
    return label_total, pred_total, cnt


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['group_by']) > 0:
        count += 1
    if len(sql['order_by']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    

    ao = sql['where'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['where'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])

    return count



def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql['select'])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['group_by'])
    if len(sql['order_by']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['order_by'][1] if unit[1]] +
                            [unit[2] for unit in sql['order_by'][1] if unit[2]])
    # agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql['select']) > 1:
        count += 1

    # number of where conditions
    if len(sql['where']) > 1:
        count += 1

    # number of group by clauses
    if len(sql['group_by']) > 1:
        count += 1

    return count


class Evaluator:
    """A simple evaluator"""
    def __init__(self):
        self.partial_scores = None

    def eval_hardness(self, sql):
        count_comp1_ = count_component1(sql)
        count_comp2_ = 0
        count_others_ = count_others(sql)

        if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
            return "easy"
        elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
                (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
            return "medium"
        elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
                (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
                (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
            return "hard"
        else:
            return "extra"

    def eval_exact_match(self, pred, label):
        partial_scores = self.eval_partial_match(pred, label)
        self.partial_scores = partial_scores

        for _, score in partial_scores.items():
            if score['f1'] != 1:
                return 0
        if label['from']['table_unit']:
            label_table = label['from']['table_unit']
            pred_table = pred['from']['table_unit']
            return label_table == pred_table
        return 1

    def eval_partial_match(self, pred, label):
        res = {}

        label_total, pred_total, cnt, cnt_wo_agg = eval_sel(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['select'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['select(no AGG)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt, cnt_wo_agg = eval_where(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['where'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['where(no OP)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_group(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group(no Having)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_having(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_order(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['order'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_and_or(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['and/or'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}


        label_total, pred_total, cnt = eval_keywords(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['keywords'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        return res




def print_scores(scores, etype):
    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                     'group', 'order', 'and/or', 'keywords']

    print("{:20} {:20} {:20} {:20} {:20} {:20}".format("", *levels))
    counts = [scores[level]['count'] for level in levels]
    print("{:20} {:<20d} {:<20d} {:<20d} {:<20d} {:<20d}".format("count", *counts))

    if etype in ["all", "exec"]:
        print('=====================   EXECUTION ACCURACY     =====================')
        this_scores = [scores[level]['exec'] for level in levels]
        print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format("execution", *this_scores))

    if etype in ["all", "match"]:
        print('\n====================== EXACT MATCHING ACCURACY =====================')
        exact_scores = [scores[level]['exact'] for level in levels]
        print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format("exact match", *exact_scores))
        print('\n---------------------PARTIAL MATCHING ACCURACY----------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['acc'] for level in levels]
            print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores))

        print('---------------------- PARTIAL MATCHING RECALL ----------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['rec'] for level in levels]
            print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores))

        print('---------------------- PARTIAL MATCHING F1 --------------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['f1'] for level in levels]
            print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores))


def evaluate(gold, predict, db_dir, etype):
    with open(gold) as f:
        glist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]

    with open(predict) as f:
        plist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
    # plist = [("select max(Share),min(Share) from performance where Type != 'terminal'", "orchestra")]
    # glist = [("SELECT max(SHARE) ,  min(SHARE) FROM performance WHERE TYPE != 'Live final'", "orchestra")]
    evaluator = Evaluator()

    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                     'group', 'order', 'and/or', 'keywords']
    entries = []
    scores = {}

    for level in levels:
        scores[level] = {'count': 0, 'partial': {}, 'exact': 0.}
        scores[level]['exec'] = 0
        for type_ in partial_types:
            scores[level]['partial'][type_] = {'acc': 0., 'rec': 0., 'f1': 0.,'acc_count':0,'rec_count':0}

    eval_err_num = 0
    for p, g in zip(plist, glist):
        p_str = p[0]
        g_str, db = g
        db_name = db
        db = os.path.join(db_dir, db, "database.json")

        schema = get_schema(db)
        g_sql = get_soql(schema, g_str)
        hardness = evaluator.eval_hardness(g_sql)
        scores[hardness]['count'] += 1
        scores['all']['count'] += 1

        try:
            p_sql = get_soql(schema, p_str)
        except Exception as e:
            eval_err_num += 1
            print ('Error: Incorrect SOQL')
            print (p_str)
            print("eval_err_num:{}".format(eval_err_num))
            print ()
            continue

        # rebuild sql for value evaluation

        g_valid_col_units = build_valid_col_units(g_sql['from']['table_unit'], schema)
        g_sql = rebuild_sql_val(g_sql)
        g_sql = rebuild_sql_col(g_valid_col_units, g_sql)
        
        p_valid_col_units = build_valid_col_units(p_sql['from']['table_unit'], schema)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid_col_units, p_sql)

        if etype in ["all", "exec"]:
            exec_score = eval_exec_match(db, p_str, g_str, p_sql, g_sql)
            if exec_score:
                scores[hardness]['exec'] += 1

        if etype in ["all", "match"]:
            exact_score = evaluator.eval_exact_match(p_sql, g_sql)
            partial_scores = evaluator.partial_scores
            if exact_score == 0:
                print("{} pred: {}".format(hardness,p_str))
                print("{} gold: {}".format(hardness,g_str))
                print("")
            scores[hardness]['exact'] += exact_score
            scores['all']['exact'] += exact_score
            for type_ in partial_types:
                if partial_scores[type_]['pred_total'] > 0:
                    scores[hardness]['partial'][type_]['acc'] += partial_scores[type_]['acc']
                    scores[hardness]['partial'][type_]['acc_count'] += 1
                if partial_scores[type_]['label_total'] > 0:
                    scores[hardness]['partial'][type_]['rec'] += partial_scores[type_]['rec']
                    scores[hardness]['partial'][type_]['rec_count'] += 1
                scores[hardness]['partial'][type_]['f1'] += partial_scores[type_]['f1']
                if partial_scores[type_]['pred_total'] > 0:
                    scores['all']['partial'][type_]['acc'] += partial_scores[type_]['acc']
                    scores['all']['partial'][type_]['acc_count'] += 1
                if partial_scores[type_]['label_total'] > 0:
                    scores['all']['partial'][type_]['rec'] += partial_scores[type_]['rec']
                    scores['all']['partial'][type_]['rec_count'] += 1
                scores['all']['partial'][type_]['f1'] += partial_scores[type_]['f1']

            entries.append({
                'predictSQL': p_str,
                'goldSQL': g_str,
                'hardness': hardness,
                'exact': exact_score,
                'partial': partial_scores
            })

    for level in levels:
        if scores[level]['count'] == 0:
            continue
        if etype in ["all", "exec"]:
            scores[level]['exec'] /= scores[level]['count']

        if etype in ["all", "match"]:
            scores[level]['exact'] /= scores[level]['count']
            for type_ in partial_types:
                if scores[level]['partial'][type_]['acc_count'] == 0:
                    scores[level]['partial'][type_]['acc'] = 0
                else:
                    scores[level]['partial'][type_]['acc'] = scores[level]['partial'][type_]['acc'] / \
                                                             scores[level]['partial'][type_]['acc_count'] * 1.0
                if scores[level]['partial'][type_]['rec_count'] == 0:
                    scores[level]['partial'][type_]['rec'] = 0
                else:
                    scores[level]['partial'][type_]['rec'] = scores[level]['partial'][type_]['rec'] / \
                                                             scores[level]['partial'][type_]['rec_count'] * 1.0
                if scores[level]['partial'][type_]['acc'] == 0 and scores[level]['partial'][type_]['rec'] == 0:
                    scores[level]['partial'][type_]['f1'] = 1
                else:
                    scores[level]['partial'][type_]['f1'] = \
                        2.0 * scores[level]['partial'][type_]['acc'] * scores[level]['partial'][type_]['rec'] / (
                        scores[level]['partial'][type_]['rec'] + scores[level]['partial'][type_]['acc'])

    print_scores(scores, etype)


def eval_exec_match(db, p_str, g_str, pred, gold):
    """
    return 1 if the values between prediction and gold are matching
    in the corresponding index. Currently not support multiple col_unit(pairs).
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(p_str)
        p_res = cursor.fetchall()
    except:
        return False

    cursor.execute(g_str)
    q_res = cursor.fetchall()

    def res_map(res, val_units):
        rmap = {}
        for idx, val_unit in enumerate(val_units):
            key = tuple(val_unit[1]) if not val_unit[2] else (val_unit[0], tuple(val_unit[1]), tuple(val_unit[2]))
            rmap[key] = [r[idx] for r in res]
        return rmap

    p_val_units = [unit[1] for unit in pred['select']]
    q_val_units = [unit[1] for unit in gold['select']]
    return res_map(p_res, p_val_units) == res_map(q_res, q_val_units)


# Rebuild SQL functions for value evaluation
def rebuild_cond_unit_val(cond_unit):
    if cond_unit is None or not DISABLE_VALUE:
        return cond_unit

    not_op, op_id, col_unit, val = cond_unit
    if type(val) != dict:
        val = None
    else:
        val = rebuild_sql_val(val)

    return not_op, op_id, col_unit, val


def rebuild_condition_val(condition):
    if condition is None or not DISABLE_VALUE:
        return condition

    res = []
    for idx, it in enumerate(condition):
        if idx % 2 == 0:
            res.append(rebuild_cond_unit_val(it))
        else:
            res.append(it)
    return res


def rebuild_sql_val(sql):
    if sql is None or not DISABLE_VALUE:
        return sql

    sql['where'] = rebuild_condition_val(sql['where'])

    return sql


# Rebuild SQL functions for foreign key evaluation
def build_valid_col_units(table_unit, schema):
    col_ids = [table_unit[1]]
    prefixs = [col_id[:-2] for col_id in col_ids]
    valid_col_units= []
    for value in schema.idMap.values():
        if '.' in value and value[:value.index('.')] in prefixs:
            valid_col_units.append(value)
    return valid_col_units


def rebuild_col_unit_col(valid_col_units, col_unit):
    if col_unit is None:
        return col_unit

    agg_id, col_id = col_unit

    return agg_id, col_id


def rebuild_val_unit_col(valid_col_units, val_unit, kmap):
    if val_unit is None:
        return val_unit

    unit_op, col_unit1, col_unit2 = val_unit
    col_unit1 = rebuild_col_unit_col(valid_col_units, col_unit1, kmap)
    col_unit2 = rebuild_col_unit_col(valid_col_units, col_unit2, kmap)
    return unit_op, col_unit1, col_unit2


def rebuild_table_unit_col(valid_col_units, table_unit):
    if table_unit is None:
        return table_unit

    return table_unit
    


def rebuild_cond_unit_col(valid_col_units, cond_unit):
    if cond_unit is None:
        return cond_unit

    not_op, op_id, col_unit, val = cond_unit
    col_unit = rebuild_col_unit_col(valid_col_units, col_unit)
    return not_op, op_id, col_unit, val


def rebuild_condition_col(valid_col_units, condition):
    for idx in range(len(condition)):
        if idx % 2 == 0:
            condition[idx] = rebuild_cond_unit_col(valid_col_units, condition[idx])
    return condition


def rebuild_select_col(valid_col_units, sel):
    if sel is None:
        return sel
    new_list = []
    for it in sel:
        agg_id, col_unit = it
        new_list.append((agg_id, rebuild_col_unit_col(valid_col_units, col_unit)))
    return new_list


def rebuild_from_col(valid_col_units, from_):
    if from_ is None:
        return from_

    table_unit = from_['table_unit']
    from_['table_unit'] = rebuild_table_unit_col(valid_col_units, table_unit)
    from_['table_name'] = from_['table_unit'].strip('__')
    
    return from_


def rebuild_group_by_col(valid_col_units, group_by):
    if group_by is None:
        return group_by

    return [rebuild_col_unit_col(valid_col_units, col_unit) for col_unit in group_by]


def rebuild_order_by_col(valid_col_units, order_by):
    if order_by is None or len(order_by) == 0:
        return order_by

    direction, col_units = order_by
    new_col_units = [rebuild_col_unit_col(valid_col_units, col_unit) for col_unit in col_units]
    return direction, new_val_units


def rebuild_sql_col(valid_col_units, sql):
    if sql is None:
        return sql

    sql['select'] = rebuild_select_col(valid_col_units, sql['select'])
    sql['from'] = rebuild_from_col(valid_col_units, sql['from'])
    sql['where'] = rebuild_condition_col(valid_col_units, sql['where'])
    sql['group_by'] = rebuild_group_by_col(valid_col_units, sql['group_by'])
    sql['order_by'] = rebuild_order_by_col(valid_col_units, sql['order_by'])

    return sql



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', dest='gold', type=str)
    parser.add_argument('--pred', dest='pred', type=str)
    parser.add_argument('--db', dest='db', type=str)
    parser.add_argument('--table', dest='table', type=str)
    parser.add_argument('--etype', dest='etype', type=str)
    args = parser.parse_args()

    gold = args.gold
    pred = args.pred
    db_dir = args.db
    table = args.table
    etype = args.etype

    assert etype in ["all", "exec", "match"], "Unknown evaluation method"

    evaluate(gold, pred, db_dir, etype)
