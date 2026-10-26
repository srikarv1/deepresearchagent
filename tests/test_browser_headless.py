"""Offline: execute the vendored class with dependency stubs; never launch a driver."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
import yaml
from adr.agents.gpt_researcher import GPTResearcherAgent

ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'third_party/gpt-researcher/gpt_researcher/scraper'


def browser_class():
    # Execute the exact complete class, avoiding unrelated optional scraper imports.
    tree=ast.parse((VENDOR/'browser/browser.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BrowserScraper')
    scope={'os':os,'platform':'darwin','random':__import__('random'),'string':__import__('string')}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'vendored-browser','exec'),scope)
    return scope['BrowserScraper'],scope


@pytest.mark.parametrize('agent_name',['real_source_v2','adaptive_depth'])
def test_config_reaches_existing_chrome_path(monkeypatch,agent_name):
    base=yaml.safe_load((ROOT/f'configs/agents/gpt_researcher_{agent_name}.yaml').read_text())
    headless=yaml.safe_load((ROOT/f'configs/agents/gpt_researcher_{agent_name}_headless.yaml').read_text())
    expected={**base,'env':{**base['env'],'GPT_RESEARCHER_BROWSER_HEADLESS':'true'}}
    assert headless==expected
    for i in range(1,9):
        cfg=yaml.safe_load((ROOT/f'configs/real_source_v2_{i:03}.yaml').read_text())
        assert cfg['agent']['config']==f"configs/agents/gpt_researcher_real_source_v2{'_headless' if i>1 else ''}.yaml"
    cls,scope=browser_class()
    import sys
    scope['__package__']='gpt_researcher.scraper.browser'
    scope['__name__']='gpt_researcher.scraper.browser.browser'
    monkeypatch.setitem(sys.modules,'selenium.webdriver.chrome.service',NS(Service=lambda:NS(process=None)))
    monkeypatch.setitem(sys.modules,'gpt_researcher.scraper.browser.lifecycle',NS(OwnedBrowser=lambda service:NS(capture=lambda:None)))
    class Options:
        def __init__(self):self.arguments=[];self.prefs={}
        def add_argument(self,x):self.arguments.append(x)
        def add_experimental_option(self,k,v):self.prefs[k]=v
    web=NS(Chrome=Mock(return_value=object()),Firefox=Mock(),Safari=Mock())
    scope.update(ChromeOptions=Options,FirefoxOptions=Options,SafariOptions=Options,webdriver=web)
    monkeypatch.setattr(cls,'_import_selenium',lambda self:None)
    monkeypatch.delenv('GPT_RESEARCHER_BROWSER_HEADLESS',raising=False)
    visible=cls('https://example.org');assert visible.headless is False
    visible.setup_driver();a=web.Chrome.call_args.kwargs['options']
    # Real adapter environment propagation, without importing/running GPT Researcher.
    import adr.agents.gpt_researcher as adapter
    monkeypatch.setattr(adapter,'find_gpt_researcher',lambda _:NS(ok=False))
    for k in [*headless['env'],'TRAJECTORY_OUTPUT_DIR','DEEP_RESEARCH_DEPTH','DEEP_RESEARCH_BREADTH','DEEP_RESEARCH_CONCURRENCY','SCRAPER','RETRIEVER','MAX_SEARCH_RESULTS_PER_QUERY','MAX_SCRAPER_WORKERS']:
        monkeypatch.setenv(k,os.environ.get(k,''))
    GPTResearcherAgent(headless)._prepare_import()
    hidden=cls('https://example.org');assert hidden.headless is True
    hidden.setup_driver();b=web.Chrome.call_args.kwargs['options']
    assert b.arguments==[a.arguments[0],'--headless',*a.arguments[1:]]
    assert a.prefs==b.prefs and hidden.user_agent==visible.user_agent
    assert web.Chrome.call_count==2 and not web.Firefox.called and not web.Safari.called
    # Exercise the actual factory selection function, with unused classes stubbed.
    factory=ast.parse((VENDOR/'scraper.py').read_text())
    method=next(n for n in ast.walk(factory) if isinstance(n,ast.FunctionDef) and n.name=='get_scraper')
    ns={name:object for name in ['PyMuPDFScraper','ArxivScraper','BeautifulSoupScraper','WebBaseLoaderScraper','NoDriverScraper','TavilyExtract','FireCrawl']}
    ns.update(BrowserScraper=cls,urlparse=__import__('urllib.parse',fromlist=['urlparse']).urlparse)
    exec(compile(ast.Module(body=[method],type_ignores=[]),'vendored-factory','exec'),ns)
    assert ns['get_scraper'](NS(scraper='browser'),'https://example.org/article') is cls


@pytest.mark.parametrize('value,expected',[('false',False),('true',True),(' TRUE ',True)])
def test_boolean(monkeypatch,value,expected):
    cls,_=browser_class();monkeypatch.setattr(cls,'_import_selenium',lambda self:None)
    monkeypatch.setenv('GPT_RESEARCHER_BROWSER_HEADLESS',value)
    assert cls('https://example.org').headless is expected


def test_invalid_setting(monkeypatch):
    cls,_=browser_class();monkeypatch.setenv('GPT_RESEARCHER_BROWSER_HEADLESS','maybe')
    with pytest.raises(ValueError,match='must be true or false'):cls('https://example.org')
