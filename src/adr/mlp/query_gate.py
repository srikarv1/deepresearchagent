"""Live pre-batch state bridge and append-only decision audit; no training."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import re

from adr.mlp.live_context import UnenforcedLiveBudget, UnenforcedLiveSnapshot, embedding_context
from adr.synthetic.schema import BranchState, BudgetState, EvidenceState, ResearchSnapshot, derive_features


def key(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def safe_text(text):
    """Deterministic privacy transform, applied BEFORE inference and logging."""
    text = str(text)
    for name,value in os.environ.items():
        if value and len(value) >= 6 and any(marker in name.upper() for marker in ('KEY','SECRET','TOKEN','PASSWORD')):
            text = text.replace(value, '[credential]')
    text = re.sub(r'(?i)\b(?:https?|bcp)://[^\s<>"\]]+', lambda m:'url-sha256:'+key(m[0]), text)
    text = re.sub(r'\b(?:sk-|sk_proj-|sk-proj-)[A-Za-z0-9_-]{8,}', '[credential]', text)
    return text


class QueryGate:
    def __init__(self, model, researcher, task, log_path, token_tracker, hooks, *, enforce_budget=False):
        self.model, self.researcher, self.task = model,researcher,task
        self.tokens, self.hooks = token_tracker,hooks
        self.enforce_budget = enforce_budget
        self.question = task.query.text
        if safe_text(self.question) != self.question:
            raise ValueError('Original question requires privacy redaction; exact-input gate cannot proceed')
        self.question_id = safe_text(task.query.id)
        self.history = []
        self.branches = {}
        self.prior_evidence = {}
        self.searches = self.reads = self.steps = self.batch = 0
        self.busy = False
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Never append a new run to an old decision stream.
        self.log = self.log_path.open('x',encoding='utf-8')
        self.event('gate_initialized', {'provenance':model.provenance, 'snapshot_policy':'committed-round-evidence/pre-batch-v1'})

    def event(self,event,record):
        # Records contain only the allowlisted state/decision fields built below.
        payload = {'event':event, 'question_id':self.question_id, **record}
        try:
            self.log.write(json.dumps(payload,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
            self.log.flush()
        except Exception:
            raise self.hooks.QueryGateError('Query gate audit write failed') from None

    def note_planning_query(self,query):
        self.history.append(safe_text(query))

    def remember_branch(self, goal, status, proposed_id):
        cleaned = safe_text(goal)
        existing = next((b for b in self.branches.values() if b.goal == cleaned), None)
        bid = existing.id if existing else safe_text(proposed_id)
        self.branches[bid] = BranchState(id=bid,goal=cleaned,status=status)

    def snapshot(self,conductor):
        if self.busy:
            raise ValueError('Concurrent outer query batches are unsupported')
        deep = self.researcher.deep_researcher
        logger = deep.trajectory_logger
        evidence = []
        for iid,e in sorted(logger._global_evidence.items(), key=lambda item:(item[1].retrieval_round,item[0])):
            if e.compressed:
                raise ValueError('Compressed evidence is incompatible with cumulative synthetic evidence semantics')
            item = EvidenceState(id=safe_text(iid),url='url-sha256:'+key(e.source_url),
                                 title='',text=safe_text(e.content),retained=e.was_retained)
            immutable = (item.url,item.title,item.text)
            if iid in self.prior_evidence and self.prior_evidence[iid] != immutable:
                raise ValueError('Live evidence body changed under an existing ID')
            self.prior_evidence[iid] = immutable
            evidence.append(item)
        if set(self.prior_evidence) != set(logger._global_evidence):
            raise ValueError('Live cumulative evidence history lost items')
        for goal in logger.trajectory.subquestions:
            bid = 'plan-'+key(goal)[:16]
            if not any(b.goal == safe_text(goal) for b in self.branches.values()):
                self.remember_branch(goal,'open',bid)
        statuses = {'open':'open','completed':'done','pruned':'dropped'}
        for round_ in logger.trajectory.rounds:
            for node in round_.frontier:
                if node.status not in statuses:
                    raise ValueError('Unknown live branch status')
                self.remember_branch(node.subquery,statuses[node.status],node.node_id)
        # A dispatched branch is active, not necessarily solved by a completed query.
        goal = conductor.researcher.query
        bid = 'active-'+key(goal)[:16]
        self.remember_branch(goal,'active',bid)
        totals = self.tokens.get_totals()
        budget = {f'max_{resource}':getattr(self.task.budget,f'max_{resource}')
                  for resource in ('steps','searches','reads','tokens')}
        budget.update(used_steps=self.steps+len(logger.trajectory.rounds),used_searches=self.searches,
                      used_reads=self.reads,used_tokens=int(totals['input_tokens']+totals['output_tokens']))
        snapshot_type = ResearchSnapshot if self.enforce_budget else UnenforcedLiveSnapshot
        return snapshot_type(evidence=evidence,executed_subqueries=list(self.history),
                                branches=list(self.branches.values()),
                                budget=(BudgetState if self.enforce_budget else UnenforcedLiveBudget)(**budget))

    async def dispatch(self,conductor,queries,scraped_data,query_domains):
        owns_batch = False
        stage = 'state_construction'
        candidate_ids = [f'{self.question_id}/b{self.batch+1:04d}/c{i:03d}' for i in range(1,len(queries)+1)]
        self.event('state_diagnostics', {'batch_index':self.batch+1, 'candidate_ids':candidate_ids,
                   'searches':self.searches,'reads':self.reads,'decisions':self.steps,
                   'research_usage':{k:self.tokens.get_totals().get(k) for k in ('input_tokens','output_tokens','cost')},
                   'evidence_count':len(self.researcher.deep_researcher.trajectory_logger._global_evidence),
                   'retained_count':sum(e.was_retained for e in self.researcher.deep_researcher.trajectory_logger._global_evidence.values()),
                   'pruned_count':sum(not e.was_retained for e in self.researcher.deep_researcher.trajectory_logger._global_evidence.values()),
                   'history_count':len(self.history),'branch_count':len(self.branches),
                   'budget_maxima':{f'max_{r}':getattr(self.task.budget,f'max_{r}') for r in ('steps','searches','reads','tokens')},
                   'original_bytes':None,'original_tokens':None,'metadata_bytes':None,'metadata_tokens':None,
                   'raw_features':None,'snapshot_status':'not_constructed'})
        try:
            if scraped_data or conductor.researcher.vector_store or any('mcpretriever' in r.__name__.lower() for r in conductor.researcher.retrievers):
                raise ValueError('Gate supports fresh standard web-query retrieval only')
            if not all(isinstance(q,str) and q.strip() for q in queries):
                raise ValueError('Invalid proposed queries')
            snapshot = self.snapshot(conductor)
            self.busy = True
            owns_batch = True
            self.batch += 1
            batch_id = f'b{self.batch:04d}'
            candidates = list(queries)
            if any(safe_text(q) != q for q in candidates):
                raise ValueError('Candidate requires privacy redaction; exact-input gate cannot proceed')
            stage = 'context_bounding'
            def diagnostics(sizes):
                try:
                    features = derive_features(self.question,snapshot).values
                except (ValueError, TypeError):
                    features = None
                self.event('context_diagnostics', {'batch_id':batch_id,'candidate_ids':candidate_ids,
                    **sizes, 'retained_count':sum(e.retained for e in snapshot.evidence),
                    'pruned_count':sum(not e.retained for e in snapshot.evidence),
                    'history_count':len(snapshot.executed_subqueries),'branch_count':len(snapshot.branches),
                    'budget':snapshot.budget.model_dump(),
                    'raw_features':features})
            _, bounding = embedding_context(snapshot, on_diagnostics=diagnostics)
            self.event('context_bounding', {'batch_id':batch_id, **bounding})
            stage = 'frozen_inference'
            p,context,raw,normalized = await self.model.predict_batch(self.question,snapshot,candidates)
            if len(p) != len(queries):
                raise ValueError('Gate prediction count mismatch')
            stage = 'decision_logging'
            snapshot_hash = key(context)
            self.event('snapshot', {'batch_id':batch_id,'snapshot_sha256':snapshot_hash,'context_before':context,
                                    'original_question':self.question,'pre_decision_budget':snapshot.budget.model_dump()})
            decisions = []
            for index,(q,prob) in enumerate(zip(candidates,p),1):
                if not 0 <= prob <= 1:
                    raise ValueError('Gate probability outside [0,1]')
                record = {'decision_id':f'{self.question_id}/{batch_id}/c{index:03d}', 'batch_id':batch_id,
                          'snapshot_sha256':snapshot_hash,'candidate_search_query':q,
                          'raw_features':raw,'normalized_features':normalized,'probability_execute':prob,
                          'threshold':.5,'decision':'EXECUTE' if prob >= .5 else 'SKIP',
                          'retrieval_invoked':False,'retrieval_path_dispatched':False,
                          'pre_decision_budget':snapshot.budget.model_dump()}
                self.event('proposed',record)
                decisions.append(record)
            # All inference finishes before ANY candidate retrieval starts.
            async def execute(index,record):
                if record['decision']=='SKIP':
                    self.event('skipped',record)
                    return ''
                token = self.hooks.active_decision.set(record)
                record['retrieval_path_dispatched'] = True
                try:
                    return await conductor._process_sub_query(queries[index],scraped_data,query_domains)
                finally:
                    self.hooks.active_decision.reset(token)
                    self.event('finished',record)
            stage = 'retrieval_dispatch'
            results = await asyncio.gather(*(execute(i,r) for i,r in enumerate(decisions)), return_exceptions=True)
            failures = [r for r in results if isinstance(r,BaseException)]
            if failures:
                raise failures[0]
            # Commit history in proposal order, never task-completion order.
            self.history.extend(r['candidate_search_query'] for r in decisions if r['retrieval_invoked'])
            self.steps += len(decisions)
            if not any(r['decision']=='EXECUTE' for r in decisions):
                self.event('empty_batch',{'batch_id':batch_id,'proposed_count':len(decisions),'policy':'return_empty_context_no_override'})
            return results
        except self.hooks.QueryGateError:
            raise
        except Exception as error:
            self.event('gate_error',{'error_type':type(error).__name__,'batch_index':self.batch,'stage':stage})
            raise self.hooks.QueryGateError('Frozen query gate failed at '+stage+' ('+type(error).__name__+'); no fallback decision') from None
        finally:
            if owns_batch:
                self.busy = False

    async def close(self):
        self.log.close()
        await self.model.close()
