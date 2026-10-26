"""Offline synthetic-v2 compiler/fixture only. No API flag or provider construction."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict

from adr.synthetic.validate import load_heldout, near_duplicate
from adr.synthetic_v2.schema import TeacherTrajectory, SourceState, SCHEMA_VERSION, FEATURE_VERSION
from adr.mlp.observation_v2 import VERSION
from adr.synthetic_v2.action_space import check_followups


def compile_trajectory(trajectory, heldout):
    trajectory=TeacherTrajectory.model_validate(trajectory.model_dump())
    if not heldout:
        raise ValueError('Held-out registry is required')
    if any(near_duplicate(trajectory.original_question,q['text']) for q in heldout):
        raise ValueError('Held-out overlap')
    for state in trajectory.states:
        check_followups(state.source.completed_results)
    question_id=hashlib.sha256(trajectory.original_question.encode()).hexdigest()[:24]
    rows=[];pairs=defaultdict(list)
    for index,item in enumerate(trajectory.states):
        for c in item.candidates:
            inputs=item.source.inputs(trajectory.original_question,c.candidate_subquery)
            view=json.loads(inputs['context_before'])
            visible=[f for card in view['prior_results'] for f in card['findings']]
            if c.supporting_finding is not None and c.supporting_finding not in visible:
                raise ValueError('Supervision references a finding absent from bounded observation')
            row={'schema_version':SCHEMA_VERSION,'observation_version':VERSION,
                 'example_id':hashlib.sha256(f'{trajectory.original_question}/{item.state_id}/{c.candidate_id}'.encode()).hexdigest()[:24],
                 'question_id':question_id,'trajectory_id':question_id+'-trajectory',
                 'state_id':item.state_id,'state_index':index,'source':item.source.model_dump(),
                 'publications_before':[p.model_dump() for p in item.publications_before],
                 **inputs,**c.model_dump()}
            rows.append(row)
            if c.counterfactual_pair_id:pairs[c.counterfactual_pair_id].append(row)
    for pair in pairs.values():
        if len(pair)!=2 or [r['label'] for r in pair]!=['EXECUTE','SKIP'] or pair[0]['state_index']>=pair[1]['state_index'] or pair[0]['candidate_subquery']!=pair[1]['candidate_subquery'] or not pair[1]['supporting_finding']:
            raise ValueError('Counterfactual requires two chronological opposite labels, identical query, visible later finding')
        witness=pair[1]['supporting_finding']
        earlier=json.loads(pair[0]['context_before'])
        if any(witness in f for card in earlier['prior_results'] for f in card['findings']):
            raise ValueError('Counterfactual decision-changing finding was already visible earlier')
    return rows


def fixture():
    """Complete offline decision trace; no unobserved searches or invented steps."""
    question='What published evidence supports replacing conventional lamps with LEDs in a community workshop?'
    counts=[3,3,3,3,2,2,2,2]
    questions=['Find published comparisons of LED and conventional lamp power use.',
               'Find published measurements of LED lamp startup delay.']
    findings=['Published measurements establish lower LED power use.',
              'Bench measurements show immediate LED startup in the tested workshop fixtures.']
    states=[];history=[];prior_decisions=0
    for i,count in enumerate(counts):
        branches=[{'id':f'b{j}','goal':('Establish LED power use and startup delay' if j==0 else f'Find independent long-term tests for lighting need {j}'),
                   'status':'done' if j<i else ('active' if j==i else 'open')} for j in range(i+5)]
        cards=[{'branch_id':f'b{j}','goal':branches[j]['goal'],
                'findings':findings if j==0 else [f'Published tests partially address lighting need {j}.'],
                'followups':[f'Find independent long-term tests for lighting need {j+1}.']} for j in range(i)]
        pages=[{'id':f'e{j}','url':f'url-sha256:{j:064x}','title':'',
                'text':'Laboratory lighting measurements report power use and startup tests; longer-duration durability remains uncertain.',
                'retained':True} for j in range(i)]
        source={'current_branch_id':f'b{i}','completed_results':cards,'state':{'branches':branches,
            'executed_subqueries':list(history),'evidence':pages,
            'budget':{'max_steps':32,'max_searches':64,'max_reads':128,'max_tokens':400000,
                      'used_steps':prior_decisions+i,'used_searches':i*10,'used_reads':i*20,'used_tokens':i*65000}}}
        candidates=[]
        for k in range(count):
            paired=i<2 and k<2
            skip=i==1 and paired
            candidates.append({'candidate_id':f'c{k+1}',
                'candidate_subquery':questions[k] if paired else f'Find published measurements of workshop lighting need {i} under operating condition {k}.',
                'label':'SKIP' if skip else 'EXECUTE',
                'decision_category':'already_established' if skip else 'unanswered_information_need',
                'reason':'Visible measurements answer this comparison for the tested fixtures.' if skip else 'This information need remains unanswered in the visible observation.',
                'counterfactual_pair_id':f'pair{k+1}' if paired else None,
                'supporting_finding':findings[k] if skip else None})
        states.append({'state_id':f's{i+1}','source':source,'candidates':candidates,
            'publications_before':[] if i==0 else [{'task_ids':[f'b{i-1}'],'evidence_ids':[f'e{i-1}']}]})
        history.extend(c['candidate_subquery'] for c in candidates if c['label']=='EXECUTE')
        prior_decisions+=count
    return TeacherTrajectory.model_validate({'original_question':question,'states':states})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',action='store_true',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--heldout',type=Path,default=Path('data/heldout/drb_baseline.json'))
    args=parser.parse_args()
    rows=compile_trajectory(fixture(),load_heldout(args.heldout))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'source.json').write_text(fixture().model_dump_json(indent=2)+'\n')
    (args.output/'examples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (args.output/'manifest.json').write_text(json.dumps({'schema_version':SCHEMA_VERSION,'observation_version':VERSION,
        'feature_version':FEATURE_VERSION,'fixture_only':True,'not_training_data':True,'examples':len(rows),
        'source_sha256':hashlib.sha256((args.output/'source.json').read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps({'examples':len(rows),'context_tokens':[r['observation_diagnostics']['tokens'] for r in rows]}))

if __name__=='__main__':main()
