"""Lossless development trace with optional local frozen-observation capture.

No provider, retrieval or embedding calls are added by logging.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from adr.mlp.embedding_limits import token_count

VERSION = 'research-representation-study-v1'


def sizes(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return {'characters': len(text), 'utf8_bytes': len(text.encode()), 'cl100k_tokens': token_count(text)}


class DevelopmentTrace:
    def __init__(self, path, question, budget, tokens, latency, candidate_context, *, canonical_source=None):
        token_count('')  # tokenizer must already be available, before research starts
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('x', encoding='utf-8')
        self.candidate_context = candidate_context
        self.question, self.budget = question, deepcopy(budget)
        self.tokens, self.latency = tokens, latency
        self.start = time.monotonic()
        self.seq = self.rounds = self.batches = self.searches = self.reads = self.planned_rounds = 0
        self.branches, self.records, self.queries = {}, {}, {}
        self.published = []
        self.canonical_source = canonical_source
        self.completed_results = {}
        self.event('session', version=VERSION, original_question=question,
                   nominal_budget=budget, evaluation=False, q051_development_exposed=True,
                   tokenizer='cl100k_base', gate_enabled=False)
        if canonical_source is not None:
            from adr.mlp.numerical_source_v2 import VERSION as numerical_version
            self.event('numerical_logging_enabled', feature_contract=numerical_version)

    def accounting(self):
        usage = deepcopy(self.tokens.get_totals())
        usage['total_tokens'] = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
        return {'search_dispatches': self.searches, 'read_attempts': self.reads,
                'tracked_tokens': usage, 'tracked_call_counts': deepcopy(self.latency.snapshot()),
                'elapsed_s': time.monotonic()-self.start, 'nominal_budget': self.budget,
                'candidate_dispatches': sum(q['status'] in ('dispatched','completed','failed') for q in self.queries.values()),
                'llm_calls': self.latency.snapshot().get('llm', 0),
                'published_rounds': self.rounds,
                'budget_usage': {'used_searches':self.searches,'used_reads':self.reads,
                    'used_tokens':usage['total_tokens'],'used_steps':len(self.queries)+self.rounds},
                'step_definition':'prior candidate proposals plus published rounds; planning searches separate'}

    def event(self, event_type, **data):
        self.seq += 1
        record = {'sequence': self.seq, 'event': event_type, 'elapsed_s': time.monotonic()-self.start, **data}
        self.stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True)+'\n')
        self.stream.flush()
        return self.seq

    def plan(self, queries, parent, depth, breadth):
        self.planned_rounds += 1
        ids = []
        for q in queries:
            bid = f'b{len(self.branches)+1:04d}'
            self.branches[bid] = {'id': bid, 'parent_id': parent, 'remaining_depth': depth,
                'breadth': breadth, 'tree_depth': 1 if parent is None else self.branches[parent]['tree_depth']+1,
                'round_id': self.planned_rounds, 'status': 'planned', 'query': q['query'], 'goal': q['researchGoal'],
                'created_order': len(self.branches)+1, 'evidence_ids': [], 'result_id': None}
            self.event('branch_created', branch=deepcopy(self.branches[bid]))
            ids.append(bid)
        return ids

    def activate(self, bid):
        self.branches[bid]['status'] = 'active'
        self.event('branch_active', branch_id=bid)

    def record(self, kind, value, bid, candidate=None):
        # Full payload appears once; repeated evidence acquisitions retain separate ownership edges.
        digest = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        rid = kind+'-'+digest
        if rid not in self.records:
            self.records[rid] = {'kind': kind, 'sizes': sizes(value), 'text_sizes': sizes('\n'.join(v for k,v in value.items() if k in ('raw_content','content','text','body') and isinstance(v,str))) if kind=='raw_page' and isinstance(value,dict) else sizes('')}
            self.event('record', record_id=rid, kind=kind, value=value, sizes=self.records[rid]['sizes'],
                       text_field_sizes={k:sizes(v) for k,v in value.items() if isinstance(v,str) and k in ('raw_content','content','text','body')}
                       if kind=='raw_page' and isinstance(value,dict) else {})
        self.event('record_acquired', record_id=rid, branch_id=bid, candidate_id=candidate)
        if bid and kind in ('raw_page', 'relevant_context', 'search_results'):
            self.branches[bid]['evidence_ids'].append(rid)
        return rid

    def raw(self, pages, bid, candidate):
        for page in pages:
            self.record('raw_page', page, bid, candidate)

    def finish(self, bid, result):
        b = self.branches[bid]
        b['status'] = 'processed' if result is not None else 'failed'
        if result is not None:
            self.completed_results[bid] = deepcopy({
                'goal': result.get('researchGoal', b['goal']),
                'findings': result['learnings'], 'followups': result['followUpQuestions'],
                'cited_insights': result.get('citations', {})})
            b['result_id'] = self.record('processed_result', result, bid)
            self.event('finding_sizes', branch_id=bid,
                       findings=[sizes(x) for x in result['learnings']],
                       followups=[sizes(x) for x in result['followUpQuestions']])
        b['completion_order'] = self.event('branch_processed', branch_id=bid,
              status=b['status'], result_id=b['result_id'], accounting=self.accounting())

    def publish(self, ids, legacy_round_id):
        self.rounds += 1
        for bid in ids:
            b = self.branches[bid]
            if b['status'] == 'processed':
                b['status'] = 'done'
                self.published.append(bid)
                b['publication_order'] = len(self.published)
        self.event('round_published', branch_ids=ids, available_completed=list(self.published),
                   legacy_round_id=legacy_round_id, accounting=self.accounting())

    def snapshot(self, bid, candidates):
        self.batches += 1
        batch = f'batch{self.batches:04d}'
        numerical = semantic_source = None
        if self.canonical_source is not None:
            from adr.mlp import numerical_source_v2 as numeric
            from adr.mlp.branch_observation_v2 import ObservationSource, PublishedBranch, Task
            accounting = self.accounting()
            numerical = numeric.capture(self.question, list(self.branches.values()),
                self.canonical_source(), self.budget, accounting['budget_usage'])
            available = set(self.published) | {bid}
            semantic_source = ObservationSource(current_task_id=bid,current_need=self.branches[bid]['goal'],
                tasks=tuple(Task(id=b['id'],parent_id=b['parent_id'],tree_depth=b['tree_depth']) for b in self.branches.values()),
                published_branches=tuple(PublishedBranch(task_id=b,**self.completed_results[b]) for b in self.published),
                executed_queries=tuple(q['query'] for q in self.queries.values() if q['branch_id'] in available and q['status']=='completed'))
        # Same-branch previously available material is allowed; unpublished siblings are excluded.
        available = set(self.published) | {bid}
        evidence = list(dict.fromkeys(r for b in self.branches.values() if b['id'] in available
                                      for r in b['evidence_ids']))
        raw = [r for r in evidence if self.records[r]['kind'] == 'raw_page']
        self.event('batch_snapshot', batch_id=batch, branch_id=bid,
            original_question_ref='session.original_question',
            branches=[{k:deepcopy(v) for k,v in b.items() if k not in ('evidence_ids','result_id')}
                      for b in self.branches.values()],
            completed_branch_ids=list(self.published),
            processed_result_ids=[self.branches[b]['result_id'] for b in self.published],
            evidence_ids=evidence,
            executed_candidate_ids=[q['id'] for q in self.queries.values()
                                    if q['branch_id'] in available and q['status']=='completed'],
            raw_state_sizes={k: sum(self.records[r]['sizes'][k] for r in raw)
                             for k in ('characters','utf8_bytes','cl100k_tokens')},
            raw_text_sizes={k: sum(self.records[r]['text_sizes'][k] for r in raw)
                            for k in ('characters','utf8_bytes','cl100k_tokens')},
            accounting=self.accounting())
        ids=[]
        for i, query in enumerate(candidates):
            cid=f'{batch}-c{i+1:03d}'
            q={'id':cid,'batch_id':batch,'branch_id':bid,'query':query,'status':'proposed'}
            self.queries[cid]=q
            self.event('candidate_proposed', **q)
            if numerical is not None:
                from adr.mlp.branch_observation_v2 import construct
                observation = construct(semantic_source, query)
                record = numeric.bind(numerical,batch_id=batch,candidate_id=cid,question=self.question,
                    candidate=query,context=observation.text,observation_version=observation.version)
                self.event('candidate_feature_snapshot', **record, context_before=observation.text,
                    observation_tokens=observation.tokens, observation_overflow=observation.overflow,
                    required_bypass_decision=observation.required_bypass_decision)
            ids.append(cid)
        return ids

    def decision(self, cid, decision):
        """Future gate adapter seam; disabled study runner never calls this."""
        if decision not in ('EXECUTE','SKIP'):
            raise ValueError('Unknown query decision')
        q = self.queries[cid]
        if q['status'] != 'proposed':
            raise ValueError('Decision must precede retrieval dispatch')
        q['decision'] = decision
        if decision == 'SKIP':
            q['status'] = 'skipped'
        self.event('candidate_decision',candidate_id=cid,branch_id=q['branch_id'],decision=decision)

    async def dispatch(self, operation, cid, *args):
        if self.queries[cid]['status'] == 'skipped':
            return ''
        token=self.candidate_context.set(cid)
        q=self.queries[cid];q['status']='dispatched'
        self.event('candidate_dispatched', candidate_id=cid, branch_id=q['branch_id'])
        try:
            result=await operation(*args)
            q['status']='completed'
            self.record('relevant_context', result, q['branch_id'],cid)
            return result
        except BaseException:
            q['status']='failed'
            raise
        finally:
            self.event('candidate_finished',candidate_id=cid,status=q['status'])
            self.candidate_context.reset(token)

    def search(self, query, bid, cid):
        self.searches+=1
        self.event('search_dispatch', query=query, branch_id=bid,candidate_id=cid,count=self.searches)

    def read(self, count, bid, cid):
        self.reads+=count
        self.event('read_dispatch',count=count,branch_id=bid,candidate_id=cid,total=self.reads)

    def close(self, status):
        for bid,b in self.branches.items():
            raw=list(dict.fromkeys(r for r in b['evidence_ids'] if self.records[r]['kind']=='raw_page'))
            self.event('branch_raw_totals',branch_id=bid,unique_records=len(raw),
                       sizes={k:sum(self.records[r]['sizes'][k] for r in raw)
                              for k in ('characters','utf8_bytes','cl100k_tokens')},
                       text_sizes={k:sum(self.records[r]['text_sizes'][k] for r in raw)
                                   for k in ('characters','utf8_bytes','cl100k_tokens')})
        self.event('session_end',status=status,accounting=self.accounting())
        self.stream.close()
