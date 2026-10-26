"""Explicitly authorized, one custom-question live smoke. Never loads DRB queries."""
import argparse
import asyncio
from datetime import datetime,timezone
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

from adr.agents.base import AgentContext
from adr.agents.gpt_researcher import GPTResearcherAgent
from adr.core.types import Budget,Query,ResearchTask


async def run(config):
    load_dotenv(override=False)
    cfg = yaml.safe_load(Path(config).read_text())
    out = Path('runs/mlp-query-gate-smoke')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False)
    task = ResearchTask(query=Query(id='nonbenchmark-smoke',dataset='manual',
                        text='What do published studies report about shade trees reducing urban heat?'),
                        budget=Budget(max_steps=32,max_searches=64,max_reads=128,max_tokens=50000))
    agent = GPTResearcherAgent(cfg)
    trajectory = await agent.run(task,AgentContext(llm=None,search=None,extra={'run_dir':str(out),'config':{'concurrency':1}}))
    # Only gate logs are emitted here. Existing fork logging/reporting is unchanged.
    (out/'summary.json').write_text(json.dumps({'query_id':task.query.id,'report_created':bool(trajectory.report),
                                               'gate_enabled':True},indent=2)+'\n')
    print('Smoke artifacts:',out.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/agents/gpt_researcher_mlp_smoke.yaml')
    parser.add_argument('--allow-api',action='store_true')
    args = parser.parse_args()
    if not args.allow_api:
        parser.error('Live smoke requires --allow-api; use pytest tests/test_mlp_query_gate.py for offline smoke')
    asyncio.run(run(args.config))


if __name__=='__main__':
    main()
