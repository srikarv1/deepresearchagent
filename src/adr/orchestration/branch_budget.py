"""Hierarchy-aware depth budget: one depth-2 expansion per root lineage.

No semantic-saturation claim: planner order is an explicit deterministic priority.
All roots and their depth-2 work remain. Only future grandchildren are suppressed.
No child outcomes/costs, embeddings, models, or retrieval are policy inputs.
"""
from collections import Counter
from adr.development.trace import DevelopmentTrace
from .live import PublicationTrace

VERSION='hierarchical-depth-budget-one-per-root-v1'
RULE='expand all depth-1 branches; at depth 2 permit one expansion per root in planner order; terminal depth stops naturally'


class DepthBudget:
    def __init__(self):self.used={};self.decisions={}

    def decide(self,bid,remaining_depth,branches,published):
        if bid in self.decisions:return dict(self.decisions[bid])
        r=dict(policy_version=VERSION,branch_id=bid,decision='EXPAND',reason='fail_open',
            rule=RULE,threshold=1,fail_open=False,depth=None,root_id=None,root_expansions_used=None,
            descendants_avoided=None,descendant_accounting='retrospective_only_not_available_live')
        try:
            b=branches[bid];d=b['tree_depth'];r['depth']=d
            if type(remaining_depth) is not int or remaining_depth<1 or type(d) is not int or d<1:raise ValueError()
            # Frozen policy is designed for the locked depth-three experiment.
            if d+remaining_depth!=4:raise ValueError()
            if bid not in published or b['status']!='done':raise ValueError()
            if d==1:
                if b['parent_id'] is not None:raise ValueError()
                r.update(reason='preserve_root_expansion',root_id=bid)
            elif d==2:
                root=b['parent_id'];parent=branches[root]
                if parent['tree_depth']!=1 or parent['parent_id'] is not None or root not in published:raise ValueError()
                used=self.used.get(root,0);r.update(root_id=root,root_expansions_used=used)
                if used>=1:r.update(decision='STOP',reason='root_deepening_budget_exhausted')
                else:
                    r['reason']='first_depth2_expansion_in_root';self.used[root]=used+1
            else:r.update(decision='STOP',reason='terminal_depth')
        except Exception:r.update(decision='EXPAND',reason='malformed_or_unpublished_state',fail_open=True)
        self.decisions[bid]=dict(r)
        return r


class BranchTrace(PublicationTrace):
    """Reuse the operational ledger/publication hooks, no query gating/MLP."""
    def event(self,event_type,**data):
        if event_type=='session':
            data.update(version=VERSION,evaluation=self.evaluation,gate_enabled=False,
                branch_expansion_policy=VERSION,raw_evidence_logged=False)
            data.pop('q051_development_exposed',None)
        try:return DevelopmentTrace.event(self,event_type,**data)
        except Exception:
            self.audit_errors+=1;self.invalid='audit_write_failed';return self.seq


class BranchExpansion:
    def __init__(self,trace):self.trace=trace;self.policy=DepthBudget();self.records=[]

    def failed_open(self,bid,depth):
        r=dict(policy_version=VERSION,branch_id=bid,remaining_depth=depth,decision='EXPAND',
            reason='controller_exception',fail_open=True,rule=RULE,threshold=1,descendants_avoided=None)
        self.records.append(r);self.trace.event('branch_expansion_decision',**r)

    def allow(self,bid,remaining_depth):
        t=self.trace
        if t.invalid:
            self.failed_open(bid,remaining_depth);return True
        r=self.policy.decide(bid,remaining_depth,t.branches,t.published)
        b=t.branches.get(bid,{})
        card=t.completed_results.get(bid,{})
        r.update(remaining_depth=remaining_depth,parent_id=b.get('parent_id'),
            result_id=b.get('result_id'),published_branch_ids=list(t.published),
            signals=dict(published_results=len(t.published),findings=len(card.get('findings',[])),
                followups=len(card.get('followups',[])),result_characters=sum(len(x) for x in card.get('findings',[])),
                planner_order=b.get('created_order')),
            budget_state=t.accounting(),publication_order=b.get('publication_order'))
        t.event('branch_expansion_decision',**r)
        if t.invalid:
            r.update(decision='EXPAND',fail_open=True,reason='audit_write_failed')
            t.event('branch_expansion_override',**r)
        self.records.append(r)
        return r['decision']=='EXPAND'

    def metrics(self):
        candidates=[r for r in self.records if r.get('remaining_depth',1)>1]
        stopped=sum(r['decision']=='STOP' and not r['fail_open'] for r in candidates)
        return dict(policy_version=VERSION,expansion_opportunities=len(candidates),
            expansions_stopped=stopped,expansions_allowed=len(candidates)-stopped,
            expansion_stop_rate=stopped/len(candidates) if candidates else 0,
            fail_open_count=sum(r['fail_open'] for r in self.records),
            completed_branches=len(self.trace.published),
            depth_counts=dict(Counter(str(b['tree_depth']) for b in self.trace.branches.values())),
            candidates=len(self.trace.queries),searches=self.trace.searches,reads=self.trace.reads,
            descendants_avoided=None,descendant_costs_used_by_policy=False)
