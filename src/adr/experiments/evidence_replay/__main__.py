import argparse
import asyncio
import json
from pathlib import Path

from .core import Tokenizer, load_pool, make_plan, save_plan


def main():
    parser = argparse.ArgumentParser(description='Phase 1: isolated evidence replay; offline by default')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('replay')
    run.add_argument('--trajectory', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--selector', choices=['random', 'threshold', 'topk'], default='topk')
    run.add_argument('--selected-ids', type=Path, help='JSON list of explicit page IDs; overrides selector')
    run.add_argument('--budget', type=int, default=50000)
    run.add_argument('--seed', type=int, default=17)
    run.add_argument('--threshold', type=float, default=.35)
    run.add_argument('--encoding', default='o200k_base')
    run.add_argument('--model', default='gpt-5-mini')
    run.add_argument('--max-completion-tokens', type=int, default=8192)
    run.add_argument('--reasoning-effort', choices=['low', 'medium', 'high'], default=None)
    run.add_argument('--fork', type=Path, default=Path('third_party/gpt-researcher'))
    run.add_argument('--allow-api', action='store_true')
    ev = sub.add_parser('evaluate')
    ev.add_argument('--replay', type=Path, required=True)
    ev.add_argument('--output', type=Path, required=True)
    ev.add_argument('--bench', type=Path, default=Path('third_party/deep_research_bench'))
    ev.add_argument('--fact', action='store_true', help='Also run FACT, including external source scraping')
    ev.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    if args.command == 'evaluate':
        from .evaluation import evaluate
        result = evaluate(args.replay, args.output, args.bench, allow_api=args.allow_api, fact=args.fact)
        print(json.dumps(result, indent=2))
        return
    pool = load_pool(args.trajectory)
    selected = json.loads(args.selected_ids.read_text()) if args.selected_ids else None
    if selected is not None and (not isinstance(selected, list) or not all(isinstance(i, str) for i in selected)):
        parser.error('--selected-ids must contain a JSON list of strings')
    plan = make_plan(pool, Tokenizer(args.encoding), budget=args.budget, method=args.selector,
                     seed=args.seed, threshold=args.threshold, selected_ids=selected)
    plan['config'].update(model=args.model, max_completion_tokens=args.max_completion_tokens,
                          reasoning_effort=args.reasoning_effort,
                          synthesis_adapter='GPT Researcher generate_report; single nonstreaming request')
    save_plan(plan, args.output)
    if args.allow_api:
        from .synthesis import synthesize
        asyncio.run(synthesize(plan, args.output, allow_api=True, model=args.model,
                               max_completion_tokens=args.max_completion_tokens, fork_path=args.fork,
                               reasoning_effort=args.reasoning_effort))
    print(json.dumps({'output': str(args.output), 'selected': len(plan['selected_ids']),
                      'context_tokens': plan['context_tokens'], 'api_enabled': args.allow_api}))


if __name__ == '__main__':
    main()
