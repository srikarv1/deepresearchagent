"""Lossless positional live metadata. Evidence bodies alone may be prefixes."""
from collections import Counter
import json

from adr.mlp.embedding_limits import token_count

EVIDENCE_FIELDS = ['id', 'url', 'title', 'text', 'retained']
BRANCH_FIELDS = ['id', 'goal', 'status']
RULE = 'metadata-table-prefix-v2'


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def encode_metadata(state):
    """Return independent mutable representation; preserve every record and its order."""
    data = {'_live_context_bounding': RULE + ': integer string fields index strings; evidence text is a prefix',
            'evidence_fields': EVIDENCE_FIELDS.copy(), 'branch_fields': BRANCH_FIELDS.copy(),
            'strings': [], 'budget': dict(state['budget']),
            'evidence': [[e[f] for f in EVIDENCE_FIELDS] for e in state['evidence']],
            'branches': [[b[f] for f in BRANCH_FIELDS] for b in state['branches']],
            'executed_subqueries': list(state['executed_subqueries'])}
    # Declare constant title/retention columns once. No record values are lost.
    defaults = {}
    for field in ('title', 'retained'):
        if data['evidence']:
            index = data['evidence_fields'].index(field)
            value = data['evidence'][0][index]
            if all(row[index] == value for row in data['evidence']):
                defaults[field] = value
                data['evidence_fields'].pop(index)
                for row in data['evidence']:
                    row.pop(index)
    if defaults:
        data['evidence_defaults'] = defaults
    url_index = data['evidence_fields'].index('url')
    prefix = 'url-sha256:'
    if data['evidence'] and all(row[url_index].startswith(prefix) for row in data['evidence']):
        data['url_prefix'] = prefix
        for row in data['evidence']:
            row[url_index] = row[url_index][len(prefix):]
    # References occur only in schema-declared string metadata cells, never body,
    # boolean or budget cells. Thus integers cannot collide with literal strings.
    cells = [(r, i) for r in data['evidence'] for i, field in enumerate(data['evidence_fields']) if field in ('id', 'url', 'title')]
    cells += [(r, i) for r in data['branches'] for i in (0, 1, 2)]
    cells += [(data['executed_subqueries'], i) for i in range(len(data['executed_subqueries']))]
    counts = Counter(r[i] for r, i in cells)
    pool = {}
    # First occurrence order; intern only if repeated-cell savings exceed entry cost.
    for r, i in cells:
        value = r[i]
        if value not in pool and counts[value] > 1:
            literal_cost = token_count(dumps(value))
            ref_cost = token_count(str(len(pool)))
            if counts[value] * (literal_cost - ref_cost) > literal_cost + 2:
                pool[value] = len(pool)
    data['strings'] = list(pool)
    for r, i in cells:
        if r[i] in pool:
            r[i] = pool[r[i]]
    return data


def decode_context(text):
    """Recover original state dict, except intentionally bounded body suffixes."""
    data = json.loads(text)
    marker = data.get('_live_context_bounding', '')
    if not marker:
        return data
    fields = data['evidence_fields']
    if marker.startswith('evidence-table-prefix-v1'):
        return {'budget': data['budget'], 'branches': data['branches'],
                'executed_subqueries': data['executed_subqueries'],
                'evidence': [dict(zip(fields, row)) for row in data['evidence']]}
    if not marker.startswith(RULE):
        raise ValueError('Unknown live metadata encoding')
    def resolve(value):
        return data['strings'][value] if type(value) is int else value
    evidence = []
    for row in data['evidence']:
        e = {**data.get('evidence_defaults', {}), **dict(zip(fields, row))}
        for f in ('id', 'url', 'title'):
            e[f] = resolve(e[f])
        e['url'] = data.get('url_prefix', '') + e['url']
        evidence.append(e)
    return {'budget': data['budget'], 'evidence': evidence,
            'branches': [dict(zip(data['branch_fields'], map(resolve, row))) for row in data['branches']],
            'executed_subqueries': list(map(resolve, data['executed_subqueries']))}
