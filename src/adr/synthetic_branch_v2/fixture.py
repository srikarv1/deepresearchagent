"""Hand-authored fictional archive fixture only; not production training data."""
import json
from adr.synthetic_branch_v2.source import World, compile_world, digest
from adr.synthetic_branch_v2.supervision import Annotation, Review, finalize

QUERY='Find the humidity recommendation in the fictional North Archive silver-gelatin storage survey.'
FINDING='The fictional North Archive survey recommends 35–45% relative humidity for its silver-gelatin collection; it does not study nitrate film.'


def fixture_world():
    events=[]
    def plan(specs):
        events.append(dict(kind='plan',tasks=specs,planning_queries=['archive literature overview'],research_tokens=1000))
    def run(tid,query,findings,followups,executed=True):
        events.extend([
            dict(kind='batch',task_id=tid,candidates=[dict(id=tid+'-query',query=query)]),
            dict(kind='end_batch',executed_ids=[tid+'-query'] if executed else [],pages=[],read_attempts=2 if executed else 0,research_tokens=2000),
            dict(kind='process',task_id=tid,findings=findings,followups=followups,
                 cited_insights={f:'https://north-archive.invalid/survey/'+tid for f in findings},research_tokens=500)])
    plan([dict(id='humidity',goal='Find silver-gelatin storage humidity recommendations'),
          dict(id='catalog',goal='Find accession catalogue identifier practices')])
    run('humidity',QUERY,[FINDING],['Find independent replication in another archive'])
    run('catalog','Find published accession catalogue practices',['The fictional catalogue uses yearly identifiers.'],[])
    events.append(dict(kind='publish',task_ids=['humidity','catalog']))
    plan([dict(id=i,goal=g) for i,g in [('packing','Find shipping crate padding evidence'),
        ('staff','Find staffing schedules'),('lighting','Find display lighting guidance'),('access','Find access request procedures')]])
    for tid,query,finding in [
        ('packing','Find published comparisons of archive shipping crate padding','The example shipping guide compares foam and paper padding.'),
        ('staff','Find published archive staffing schedules','The example operations report documents weekend staffing gaps.'),
        ('lighting','Find published display lighting guidance','The example exhibition guide limits display duration.'),
        ('access','Find published access request procedures','The example access guide describes appointment booking.')]:
        run(tid,query,[finding],['Find independent published evaluations'])
    events.append(dict(kind='publish',task_ids=['packing','staff','lighting','access']))
    plan([dict(id='followup',parent_id='humidity',goal='Find archive humidity recommendations and their limits')])
    events.extend([
        dict(kind='batch',task_id='followup',candidates=[dict(id='late',query=QUERY),
             dict(id='ambiguous',query='Find further general archive storage advice')]),
        dict(kind='end_batch',executed_ids=[],pages=[],read_attempts=0,research_tokens=0),
        dict(kind='drop',task_id='followup')])
    return World.model_validate(dict(question_id='offline-archive-fixture',
        original_question='What published guidance could inform storage and access planning for a small photographic archive?',events=events))


def overflow_world():
    world=fixture_world().model_dump()
    world['events'][-4]['tasks'][0]['goal']='archive need ' * 5000
    return World.model_validate(world)


def fixture_annotations(packets):
    late=json.loads(packets['late']['context_before'])
    index=next(i for i,b in enumerate(late['branches']) if FINDING in b['findings'])
    witness=dict(kind='finding',branch_index=index,text=FINDING)
    specs=[dict(candidate_id='humidity-query',label='EXECUTE',category='unresolved_need',
        rationale='The observation has no published result answering this specific survey query.',pair_id='humidity-pair'),
        dict(candidate_id='late',label='SKIP',category='direct_answer',
        rationale='The selected survey finding directly supplies the requested recommendation, with its collection qualifier.',
        witnesses=[witness],decisive_witness=witness,pair_id='humidity-pair'),
        dict(candidate_id='ambiguous',label='AMBIGUOUS',category='uncertain_marginal_value',
        rationale='The broad query could uncover a useful new topic or duplicate the visible guidance; the packet cannot distinguish these outcomes.')]
    annotations=[Annotation(**s,packet_digest=digest(packets[s['candidate_id']])) for s in specs]
    reviews={a.candidate_id:Review(annotation_digest=digest(a.model_dump()),reviewer='hand-authored-offline-fixture-review',
        verdict='SUPPORTED',explanation=a.rationale) for a in annotations if a.label!='AMBIGUOUS'}
    return annotations,reviews


def main():
    packets,audit,execution=compile_world(fixture_world())
    annotations,reviews=fixture_annotations(packets)
    accepted,rejected=finalize(packets,audit,execution,annotations,reviews)
    _,overflow_audit,_=compile_world(overflow_world())
    print(json.dumps({'accepted':len(accepted),'rejected':rejected,
        'overflow_fixture':{k:overflow_audit['late'][k] for k in ('tokens','overflow','required_bypass_decision')},
        'observations':{k:{f:a[f] for f in ('tokens','published_count','history_count','branch_count','selected_task_ids')} for k,a in audit.items()}},indent=2))

if __name__=='__main__':
    main()
