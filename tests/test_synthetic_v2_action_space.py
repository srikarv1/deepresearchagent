import pytest
from pathlib import Path
from adr.synthetic_v2.action_space import check_retrieval
from adr.synthetic_v2.generate import fixture, compile_trajectory
from adr.synthetic_v2.schema import TeacherTrajectory
from adr.synthetic.validate import load_heldout

@pytest.mark.parametrize('text',[
 'Run a six-hour experiment','Conduct a pilot','Deploy a sensor array',
 'Instrument hardware','Collect new measurements','Simulate a thermal event',
 'Model the cooling system ourselves','Calculate energy costs','Estimate lifecycle overhead',
 'Analyze existing logs','Synthesize available aging logs','Create a comparative table',
 'Draft an SOP','Derive operational SOP guidance','Formulate a research question',
 'Negotiate with suppliers','Procure equipment','Build a prototype','Implement a controller',
 'Field-validate the published result','Replicate the experiment',
])
def test_imperatives_rejected(text):
    with pytest.raises(ValueError,match='retrieval query'):check_retrieval(text)

@pytest.mark.parametrize('text',[
 'Find published experiments comparing cooling systems',
 'Retrieve published analyses of aging tests',
 'Search for SOP evaluations and simulation studies',
 'Find reports of field validation of sensors',
 'Analysis of published battery experiments',
 'Experimental measurements of cooling performance review',
 'Run a literature search for maintenance reports',
 'Find published estimates of lifetime costs',
 'Retrieve studies that instrument containers during field experiments',
])
def test_searches_about_actions_accepted(text):check_retrieval(text)


def test_nonretrieval_followup_rejected_without_changing_live_schema():
    d=fixture().model_dump()
    for s in d['states'][1:]:s['source']['completed_results'][0]['followups']=['Deploy a pilot ourselves']
    trajectory=TeacherTrajectory.model_validate(d)
    with pytest.raises(ValueError,match='retrieval query'):
        compile_trajectory(trajectory,load_heldout(Path('data/heldout/drb_baseline.json')))


def test_partial_quote_is_not_exact_finding():
    d=fixture().model_dump();d['states'][1]['candidates'][0]['supporting_finding']='lower LED power use'
    with pytest.raises(ValueError,match='absent'):
        compile_trajectory(TeacherTrajectory.model_validate(d),load_heldout(Path('data/heldout/drb_baseline.json')))


def test_counterfactual_query_must_match_exactly():
    d=fixture().model_dump();d['states'][1]['candidates'][0]['candidate_subquery']+=' published trials'
    with pytest.raises(ValueError,match='identical query'):
        compile_trajectory(TeacherTrajectory.model_validate(d),load_heldout(Path('data/heldout/drb_baseline.json')))
