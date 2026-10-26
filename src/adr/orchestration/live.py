"""Fail-open local policy at existing dispatch/publication hooks.

Reuses DevelopmentTrace's operational ledger, without recording raw pages or
claiming evaluation runs are development data. Existing hooks own publication.
The disabled control uses identical observational hooks and normal dispatch.
"""
import asyncio
from copy import deepcopy
import hashlib
import json
from adr.development.trace import DevelopmentTrace
from adr.mlp.branch_observation_v2 import ObservationSource, PublishedBranch, Task
from .redundancy import decide, VERSION, THRESHOLD


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


class PublicationTrace(DevelopmentTrace):
    def __init__(self,*args,policy,evaluation=False,**kwargs):
        self.policy=policy;self.evaluation=evaluation;self.invalid=None;self.source=None
        self.last_ids=[];self.audit_errors=0
        super().__init__(*args,**kwargs)

    def event(self,event_type,**data):
        if event_type=='session':
            data.update(version=VERSION,evaluation=self.evaluation,gate_enabled=self.policy!='always_execute',policy=self.policy)
            data.pop('q051_development_exposed',None)
        try:return super().event(event_type,**data)
        except Exception:
            # Audit failure disables pruning, never repeats a retrieval.
            self.audit_errors+=1;self.invalid='audit_write_failed';return self.seq

    def raw(self,*args):pass

    def record(self,kind,value,bid,candidate=None):
        if kind!='processed_result':return ''
        semantic={k:deepcopy(value[k]) for k in ('researchGoal','learnings','followUpQuestions','citations') if k in value}
        return super().record(kind,semantic,bid,candidate)

    def plan(self,queries,parent,depth,breadth):
        try:return super().plan(queries,parent,depth,breadth)
        except Exception:
            self.invalid='malformed_plan'
            return [f'unknown-{self.planned_rounds}-{i}' for i in range(len(queries))]

    def activate(self,bid):
        try:super().activate(bid)
        except Exception:self.invalid='malformed_activation'

    def finish(self,bid,result):
        try:super().finish(bid,result)
        except Exception:self.invalid='malformed_result'

    def publish(self,ids,legacy_round_id):
        try:super().publish(ids,legacy_round_id)
        except Exception:self.invalid='malformed_publication'

    def snapshot(self,bid,candidates):
        # Exactly the same source semantics as prospective numerical capture:
        # published results, plus completed queries in published/current branch.
        self.source=None
        try:
            if self.invalid:raise ValueError('Invalid ledger')
            available=set(self.published)|{bid}
            self.source=ObservationSource(current_task_id=bid,current_need=self.branches[bid]['goal'],
                tasks=tuple(Task(id=b['id'],parent_id=b['parent_id'],tree_depth=b['tree_depth']) for b in self.branches.values()),
                published_branches=tuple(PublishedBranch(task_id=b,**deepcopy(self.completed_results[b])) for b in self.published),
                executed_queries=tuple(q['query'] for q in self.queries.values() if q['branch_id'] in available and q['status']=='completed'))
        except Exception:self.invalid=self.invalid or 'malformed_snapshot'
        try:self.last_ids=super().snapshot(bid,candidates)
        except Exception:
            self.invalid='snapshot_logging_failed';self.batches+=1;self.last_ids=[]
            for i,q in enumerate(candidates):
                cid=f'batch{self.batches:04d}-c{i+1:03d}';self.last_ids.append(cid)
                self.queries[cid]=dict(id=cid,batch_id=f'batch{self.batches:04d}',branch_id=bid,query=q,status='proposed')
        return list(self.last_ids)

    def close(self,status):
        self.event('session_end',status=status,accounting=self.accounting(),raw_evidence_logged=False)
        try:self.stream.close()
        except Exception:self.audit_errors+=1


class RedundancyGate:
    def __init__(self,trace,hooks,policy):
        self.trace,self.hooks,self.policy=trace,hooks,policy
        self.searches=self.reads=self.steps=0
        self.decisions=[]

    def note_planning_query(self,query):pass # planning searches are not completed candidate history

    def event(self,event,record):
        # note_retrieval already emits this through active_trace and updates
        # the decision record; do not duplicate the same dispatch event.
        if event!='retrieval_invoked':self.trace.event(event,**record)

    async def dispatch(self,conductor,queries,scraped_data,query_domains):
        source=self.trace.source # immutable snapshot shared by ALL candidates
        ids=list(self.trace.last_ids)
        snapshot_id=digest(source.model_dump()) if source else None
        decisions=[]
        for i,q in enumerate(queries):
            try:
                result,obs=decide(source,q,self.policy)
                if self.trace.invalid or scraped_data or getattr(conductor.researcher,'vector_store',None):
                    result.update(decision='EXECUTE',fail_open=True,reason=self.trace.invalid or 'unsupported_retrieval_state')
                result.update(candidate_id=ids[i],decision_id=ids[i],batch_id=self.trace.queries[ids[i]]['batch_id'],
                    snapshot_id=snapshot_id,observation_digest=digest(obs.text) if obs else None,
                    candidate_query=q,retrieval_invoked=False,retrieval_path_dispatched=False,
                    selected_result_ids=[self.trace.branches[b]['result_id'] for b in result['selected_branch_ids']],
                    accounting=self.trace.accounting())
            except Exception:
                result=dict(policy=self.policy,version=VERSION,decision='EXECUTE',score=0.0,threshold=THRESHOLD,
                    fail_open=True,reason='decision_construction_failed',components={},selected_branch_ids=[],selected_result_ids=[],
                    candidate_id=ids[i] if i<len(ids) else f'unknown-{i}',batch_id=f'batch{self.trace.batches:04d}',snapshot_id=snapshot_id,
                    candidate_query=q,retrieval_invoked=False,retrieval_path_dispatched=False)
            decisions.append(result)
        # Persist all decisions before ANY retrieval. Logging failure => whole
        # batch fail-open; do not let earlier logged SKIPs suppress retrieval.
        for r in decisions:self.event('heuristic_decision',r)
        if self.trace.invalid:
            for r in decisions:r.update(decision='EXECUTE',fail_open=True,reason=self.trace.invalid)
            for r in decisions:self.event('heuristic_decision_override',r)
        async def execute(i,r):
            cid=r['candidate_id']
            try:
                if cid in self.trace.queries:self.trace.decision(cid,r['decision'])
            except Exception:
                r.update(decision='EXECUTE',fail_open=True,reason='decision_ledger_failed')
            if self.trace.invalid:
                r.update(decision='EXECUTE',fail_open=True,reason=self.trace.invalid)
            if r['decision']=='SKIP':
                self.event('skipped',r);return ''
            r['retrieval_path_dispatched']=True
            token=self.hooks.active_decision.set(r)
            try:
                if cid in self.trace.queries:
                    return await self.trace.dispatch(conductor._process_sub_query,cid,queries[i],scraped_data,query_domains)
                return await conductor._process_sub_query(queries[i],scraped_data,query_domains)
            finally:
                self.hooks.active_decision.reset(token);self.event('heuristic_finished',r)
        # Exceptions FROM research propagate normally; never retry a call after
        # it may already have produced external side effects.
        results=await asyncio.gather(*(execute(i,r) for i,r in enumerate(decisions)),return_exceptions=True)
        self.steps+=len(decisions);self.decisions.extend(decisions)
        for r in results:
            if isinstance(r,BaseException):raise r
        return results

    def metrics(self):
        n=len(self.decisions);skip=sum(r['decision']=='SKIP' for r in self.decisions)
        return dict(policy=self.policy,policy_version=VERSION,candidate_count=n,skip_count=skip,
            execute_count=n-skip,skip_rate=skip/n if n else 0,fail_open_count=sum(r['fail_open'] for r in self.decisions),
            error_fail_open_count=sum(r['fail_open'] and r['reason'] not in ('no_published_research','no_eligible_answer_statement','insufficient_redundancy_confidence') for r in self.decisions),
            audit_errors=self.trace.audit_errors,searches=self.trace.searches,reads=self.trace.reads,
            no_added_model_embedding_search_calls=True)
