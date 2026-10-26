import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from adr.synthetic_v2.events import encode_fixture
from adr.synthetic_v2.generate import fixture, compile_trajectory
from adr.synthetic_v2.visibility import VisibleEventTrajectory, expand_visible, visibility_views
from adr.synthetic.validate import load_heldout


def visible_fixture():
    trajectory=fixture();events=encode_fixture(trajectory).model_dump()
    for state,view in zip(events['states'],visibility_views(trajectory),strict=True):
        state['visible_findings']=view['visible_findings']
    return events


def test_authoritative_visibility_and_identical_model_inputs():
    events=VisibleEventTrajectory.model_validate(visible_fixture())
    trajectory,views=expand_visible(events)
    assert trajectory==fixture()
    assert views==visibility_views(fixture())
    assert views[0]['visible_findings']==[]
    assert views[-1]['diagnostics']['omitted_result_cards']==3
    heldout=load_heldout(Path('data/heldout/drb_baseline.json'))
    assert compile_trajectory(trajectory,heldout)==compile_trajectory(fixture(),heldout)


@pytest.mark.parametrize('state_index',[0,7])
def test_future_and_evicted_visibility_declarations_rejected(state_index):
    d=visible_fixture()
    d['states'][state_index]['visible_findings'].append(d['states'][1]['visible_findings'][0])
    with pytest.raises(ValueError,match='locally bounded view'):
        expand_visible(VisibleEventTrajectory.model_validate(d))


def test_missing_visible_finding_rejected():
    d=visible_fixture();d['states'][1]['visible_findings'].pop()
    with pytest.raises(ValueError,match='locally bounded view'):
        expand_visible(VisibleEventTrajectory.model_validate(d))


@pytest.mark.asyncio
async def test_smoke4_is_single_call_no_retry_and_persists_local_view(tmp_path,monkeypatch):
    path=Path('runs/synthetic-v2-smoke/teacher-smoke-4-20260923/run_smoke.py')
    spec=importlib.util.spec_from_file_location('smoke4_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.HERE=tmp_path
    monkeypatch.setenv('OPENAI_API_KEY','offline-key')
    monkeypatch.setenv('OPENAI_BASE_URL','https://offline.invalid/openai/v1/')
    monkeypatch.setattr('sys.argv',['run_smoke.py','--allow-api'])
    calls=[];raw=json.dumps(visible_fixture())
    class Client:
        def __init__(self):self._client=self
        def with_options(self,**kw):assert kw=={'max_retries':0};return self
        async def close(self):pass
        async def complete(self,*a,**kw):
            calls.append(1)
            return NS(text=raw,raw={'choices':[{'finish_reason':'stop'}],'usage':{}},usage=NS(model_dump=lambda:{}))
    def build(cfg):assert cfg['model']=='gpt-5-mini' and cfg['max_output_tokens']==32000;return Client()
    monkeypatch.setattr(m,'build_llm',build)
    await m.main()
    assert json.loads((tmp_path/'validation.json').read_text())['structural_validation']=='passed'
    assert json.loads((tmp_path/'visibility_views.json').read_text())==visibility_views(fixture())
    with pytest.raises(SystemExit,match='already attempted'):await m.main()
    assert calls==[1]
