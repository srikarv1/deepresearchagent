"""Opt-in observation ledger for a future v2 gate; no model or provider calls."""
from copy import deepcopy
from adr.synthetic.schema import BranchState, EvidenceState
from adr.mlp.live_context import UnenforcedLiveBudget, UnenforcedLiveSnapshot
from adr.mlp.query_gate import safe_text, key
from adr.synthetic_v2.schema import SourceState, CompletedResult

class LiveObservationV2:
    def __init__(self, budget, token_tracker):
        self.budget=budget
        self.tokens=token_tracker
        self.branches={}; self.cards=[]; self.evidence=[]; self.history=[]
        self.round_sequence=0; self.committed_rounds=0
        self.searches=0;self.reads=0;self.decisions=0
        self.batch_source=None;self.batch_queries=[];self.issued=set()

    def plan_round(self, queries):
        self.round_sequence+=1
        ids=[]
        for i,q in enumerate(queries):
            bid=f'r{self.round_sequence}-b{i+1}'
            self.branches[bid]=BranchState(id=bid,goal=safe_text(q['researchGoal']),status='open')
            ids.append(bid)
        return ids

    def start_branch(self,bid):
        if any(b.status=='active' for b in self.branches.values()):
            raise ValueError('v2 observation requires outer concurrency=1')
        self.branches[bid].status='active'

    def finish_branch(self,bid,successful=True):
        # Operational status only. Findings remain unpublished until round commit.
        self.branches[bid].status='done' if successful else 'dropped'

    def publish_round(self,ids,results,evidence):
        if self.batch_source is not None:
            raise ValueError('Cannot publish results during a candidate batch')
        additions=[]
        for bid,result in zip(ids,results,strict=True):
            if result is None:
                self.branches[bid].status='dropped'
                continue
            additions.append(CompletedResult(branch_id=bid,goal=self.branches[bid].goal,
                findings=[safe_text(s) for s in result['learnings']],
                followups=[safe_text(s) for s in result['followUpQuestions']]))
        # No dependency on asynchronous completion order or mutable producer values.
        pages=[EvidenceState(id=safe_text(i),url='url-sha256:'+key(e.source_url),title='',
                             text=safe_text(e.content)[:2000],retained=e.was_retained)
               for i,e in sorted(evidence.items(),key=lambda item:(item[1].retrieval_round,item[0]))]
        if any(not p.retained for p in pages):
            raise ValueError('v2 no-pruning observation received pruned evidence')
        self.cards.extend(deepcopy(additions));self.evidence=pages;self.committed_rounds+=1

    def note_search(self,query):
        self.searches+=1
        if self.batch_source is not None:
            self.issued.add(query)

    def begin_batch(self,bid,queries):
        if self.batch_source is not None:
            raise ValueError('Overlapping v2 candidate batches')
        totals=self.tokens.get_totals()
        maxima={f'max_{r}':getattr(self.budget,f'max_{r}') for r in ('steps','searches','reads','tokens')}
        budget=UnenforcedLiveBudget(**maxima,used_steps=self.decisions+self.committed_rounds,
            used_searches=self.searches,used_reads=self.reads,
            used_tokens=int(totals['input_tokens']+totals['output_tokens']))
        source=SourceState(current_branch_id=bid,state=UnenforcedLiveSnapshot(
            evidence=deepcopy(self.evidence),branches=deepcopy(list(self.branches.values())),
            executed_subqueries=list(self.history),budget=budget),completed_results=deepcopy(self.cards))
        self.batch_source=source;self.batch_queries=list(queries);self.issued=set()
        return source.model_copy(deep=True)

    def inputs(self,question,candidate):
        if self.batch_source is None:
            raise ValueError('No frozen pre-decision v2 batch')
        return self.batch_source.inputs(question,candidate)

    def end_batch(self):
        # Attempts count globally; text history contains issued candidate searches,
        # in proposal order, including searches that returned empty/failed results.
        self.history.extend(safe_text(q) for q in self.batch_queries if q in self.issued)
        self.decisions+=len(self.batch_queries)
        self.batch_source=None;self.batch_queries=[];self.issued=set()
