"""Local lexical redundancy, not semantic entailment or a trained classifier.

Fit TF-IDF only to individual findings/insights in the frozen selected packet.
For each statement, multiply cosine similarity by candidate unigram coverage;
use the maximum SINGLE-statement product. Goals, followups and history cannot
by themselves license SKIP. Numeric qualifiers must occur in that statement.
Explicit uncertainty/absence wording vetoes a statement. These fixed guards are
conservative syntactic safeguards, not a claim to understand every qualifier.
"""
import json
import math
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from adr.mlp.branch_observation_v2 import construct, VERSION as OBSERVATION_VERSION, MAX_TOKENS
from adr.mlp.embedding_limits import token_count

VERSION = 'lexical-redundancy-single-statement-v1'
THRESHOLD = 0.22
POLICIES = ('always_execute', 'conservative')
# No synonyms, stemming, domain terms, label hints, or additional learned calls.
UNCERTAIN = re.compile(r'\b(?:unclear|unknown|unresolved|uncertain|inconclusive|insufficient|lack(?:s|ing)?|no evidence|not established|not demonstrated)\b', re.I)
NUMBERS = re.compile(r'\d+(?:\.\d+)?')
# Preserve literal answer dimensions: a reported value is not a protocol or a
# population distribution. Singular/plural variants only; no semantic expansion.
DIMENSIONS = ('protocol', 'method', 'distribution', 'share', 'variance', 'percentile',
              'uncertainty', 'comparison', 'cost', 'threshold', 'sensitivity')

def dimensions(text):
    return {d for d in DIMENSIONS if re.search(r'\b'+d+r's?\b',text,re.I)}


def decide_view(context, candidate, policy='conservative'):
    if policy not in POLICIES:
        raise ValueError('Unknown frozen redundancy policy')
    result=dict(policy=policy,version=VERSION,observation_version=OBSERVATION_VERSION,
        decision='EXECUTE',score=0.0,threshold=THRESHOLD if policy=='conservative' else None,
        fail_open=False,reason='baseline',context_tokens=None,components={},witness=None)
    def bypass(reason):
        result.update(fail_open=policy!='always_execute',reason=reason)
        return result
    if policy=='always_execute':return result
    try:
        result['context_tokens']=token_count(context)
        if result['context_tokens']>MAX_TOKENS:return bypass('observation_overflow')
        view=json.loads(context)
        if set(view)!={'current_need','recent_queries','branches'} or not isinstance(view['current_need'],str):return bypass('malformed_observation')
        if not isinstance(candidate,str) or not candidate.strip():return bypass('invalid_candidate')
        if not isinstance(view['recent_queries'],list) or not all(isinstance(q,str) for q in view['recent_queries']):return bypass('malformed_history')
        if not isinstance(view['branches'],list):return bypass('malformed_branches')
        if not view['branches']:return bypass('no_published_research')
        if len(view['branches'])>3 or len(view['recent_queries'])>6:return bypass('malformed_window')
        docs=[];refs=[]
        for i,b in enumerate(view['branches']):
            if set(b)!={'goal','findings','followups','cited_insights','relation','tree_depth'}:return bypass('malformed_branch')
            if not isinstance(b['goal'],str) or b['relation'] not in ('parent','ancestor','other') or type(b['tree_depth']) is not int or b['tree_depth']<1:return bypass('malformed_branch')
            if not all(isinstance(b[k],list) and all(isinstance(t,str) for t in b[k]) for k in ('findings','followups')):return bypass('malformed_statements')
            if not isinstance(b['cited_insights'],dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in b['cited_insights'].items()):return bypass('malformed_insights')
            for kind,items in [('finding',b['findings']),('insight',list(b['cited_insights']))]:
                for j,text in enumerate(items):
                    if text not in docs:docs.append(text);refs.append(dict(branch_index=i,kind=kind,index=j))
        analyzer=TfidfVectorizer(stop_words='english').build_analyzer()
        terms=set(analyzer(candidate))
        if not terms or not docs:return bypass('empty_lexical_evidence')
        v=TfidfVectorizer(lowercase=True,stop_words='english',ngram_range=(1,2),sublinear_tf=True,smooth_idf=True,norm='l2')
        m=v.fit_transform(docs);cos=(m@v.transform([candidate]).T).toarray().ravel()
        coverage=[len(terms&set(analyzer(d)))/len(terms) for d in docs]
        values=[float(c)*t for c,t in zip(cos,coverage)]
        eligible=[not UNCERTAIN.search(d) and set(NUMBERS.findall(candidate))<=set(NUMBERS.findall(d)) and dimensions(candidate)<=dimensions(d) for d in docs]
        allowed=[i for i in range(len(docs)) if eligible[i]]
        hist=max([len(terms&set(analyzer(q)))/max(1,len(terms|set(analyzer(q)))) for q in view['recent_queries']]+[0.0])
        result['components']={'prior_query_jaccard':hist,'unvetoed_max_score':max(values),'eligible_statements':len(allowed),'statements':len(docs)}
        if not allowed:return bypass('no_eligible_answer_statement')
        j=max(allowed,key=lambda i:(values[i],-i));score=values[j]
        if not math.isfinite(score):return bypass('nonfinite_score')
        result.update(score=score,witness=refs[j])
        result['components'].update(answer_cosine=float(cos[j]),candidate_unigram_coverage=coverage[j])
        if score>=THRESHOLD:result.update(decision='SKIP',reason='high_single_statement_overlap')
        else:return bypass('insufficient_redundancy_confidence')
        return result
    except Exception:
        return bypass('scoring_failed')


def decide(source, candidate, policy='conservative'):
    try:
        obs=construct(source,candidate)
        r=decide_view(obs.text,candidate,policy)
        r.update(selected_branch_ids=list(obs.selected_task_ids),branch_selection_scores=list(obs.scores),context_tokens=obs.tokens)
        return r,obs
    except Exception:
        return dict(policy=policy,version=VERSION,observation_version=OBSERVATION_VERSION,
            decision='EXECUTE',score=0.0,threshold=THRESHOLD if policy=='conservative' else None,
            fail_open=policy!='always_execute',reason='observation_construction_failed',components={},witness=None,
            selected_branch_ids=[],branch_selection_scores=[],context_tokens=None),None
