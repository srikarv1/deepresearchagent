"""DeepResearchGym port of the evidence-replay REINFORCE study, offline only.

Policy, features, proxy reward, samplers and evaluation are the DRB modules,
unchanged. What is added for DRG:

- pools come from a split file and saved Gym runs (``<run>/gpt_researcher/<id>.json``);
- every update samples K rollouts per training query (the DRB loop sampled one);
- the advantage is one of
    global1      R - EMA baseline, K=1        (the DRB pilot, for reference)
    global       R - EMA baseline, K rollouts (same episodes as center/standardize)
    center       R - mean of the query's K rollouts
    standardize  (R - mean) / (std + eps) over the query's K rollouts (GRPO);
- heuristic baselines, a proxy-greedy shortlist baseline and mask-change counts.

Test ids are refused unless ``allow_test`` is set; nothing here calls a model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .core import Tokenizer, select, sha
from .learning_signal import EVAL_SEEDS, eval_model, make_shortlist, order_seed, sample
from .reinforce import FEATURES, WEIGHTS, Config, ExactAllocator, Policy
from .reinforce_pilot import aggregate
from .reward import VERSION, RewardPool, total_reward

MODES = ('global1', 'global', 'center', 'standardize')
PROTOCOLS = ('shortlist', 'random_order', 'cosine')
ROLES = {'train': 'train', 'val': 'validation', 'test': 'test'}
EPS = 1e-6
EVAL_EPOCHS = 5  # evaluate the untrained policy, every 5th epoch and the last
MAX_EPOCHS = 500


@dataclass(frozen=True)
class LongConfig(Config):
    """Config without the pilot's 30-epoch cap; every other check is Config's own."""
    def __post_init__(self):
        if not 1 <= self.epochs <= MAX_EPOCHS:
            raise ValueError(f'Requires 1..{MAX_EPOCHS} epochs')
        Config(**(asdict(self) | {'epochs': 1}))


def make_policy(init_keep=.7):
    """Aadya's Policy with its initial p(KEEP) moved; .7 leaves it unchanged."""
    if not 0 < init_keep < 1:
        raise ValueError('init_keep must be in (0, 1)')
    model = Policy()
    torch.nn.init.constant_(model.net[-1].bias, math.log(init_keep / (1 - init_keep)))
    return model


def pool_size(path):
    return len(json.loads(Path(path).read_text(encoding='utf-8'))['evidence'])


def find_pool(qid, run_dirs):
    """A single run (``<run>/gpt_researcher/<id>.json``) or a rollout corpus
    (``<run>/<id>-s<seed>/gpt_researcher/<id>.json``, largest pool, lowest seed on ties)."""
    for run in run_dirs:
        path = Path(run) / 'gpt_researcher' / f'{qid}.json'
        if path.is_file():
            return path
        rollouts = sorted(Path(run).glob(f'{qid}-s*/gpt_researcher/{qid}.json'),
                          key=lambda p: int(p.parent.parent.name.rsplit('-s', 1)[1]))
        if rollouts:
            return max(rollouts, key=pool_size)  # max keeps the first, i.e. lowest seed, on ties
    raise FileNotFoundError(f'No saved pool for query {qid} under {[str(r) for r in run_dirs]}')


def load_drg(split_path, run_dirs, tokenizer, budget, *, allow_test=False):
    """Load every split query once. Returns pools, allocators, role and qid by trajectory id."""
    split = json.loads(Path(split_path).read_text(encoding='utf-8'))
    pools, allocators, roles, qids, sources = [], [], {}, {}, []
    for key, role in ROLES.items():
        if role == 'test' and not allow_test:
            continue
        for qid in split.get(key, []):
            path = find_pool(qid, run_dirs)
            rp = RewardPool.load(path)
            tid = rp.pool.trajectory_id
            if tid in roles:
                raise ValueError(f'Pool {tid} appears twice in the split')
            allocator = ExactAllocator(rp, tokenizer, budget)
            if allocator.tokens.min() > budget:
                raise ValueError(f'No page of query {qid} fits the budget')
            pools.append(rp); allocators.append(allocator)
            roles[tid] = role; qids[tid] = qid
            sources.append({'query_id': qid, 'trajectory_id': tid, 'role': role, 'pages': len(rp.pool.pages),
                            'provenance': rp.pool.provenance})
    if not any(r == 'train' for r in roles.values()):
        raise ValueError('Split has no training queries')
    return pools, allocators, roles, qids, sources


def advantages(rewards, groups, mode, baseline):
    """Per-episode advantages. ``groups`` names each episode's query."""
    r = np.asarray(rewards, dtype=float)
    if mode in ('global1', 'global'):
        return r - baseline
    if mode not in MODES:
        raise ValueError(f'Unknown advantage mode {mode}')
    groups = np.asarray(groups)
    out = np.empty_like(r)
    for g in np.unique(groups):
        ix = groups == g
        out[ix] = r[ix] - r[ix].mean()
        if mode == 'standardize':
            out[ix] /= r[ix].std() + EPS
    return out


def group_loss(episodes, adv, entropy):
    adv = torch.tensor(adv, dtype=torch.float32)
    log_probs = torch.stack([e['log_prob'] for e in episodes])
    return -(adv * log_probs).mean() - entropy * torch.stack([e['entropy'] for e in episodes]).mean()


def spread(rewards, groups):
    """Mean within-query std and std of query means."""
    by = {}
    for r, g in zip(rewards, groups):
        by.setdefault(g, []).append(r)
    within = float(np.mean([np.std(v) for v in by.values()]))
    between = float(np.std([np.mean(v) for v in by.values()]))
    zero = float(np.mean([np.ptp(v) == 0 for v in by.values()])) if all(len(v) > 1 for v in by.values()) else None
    return within, between, zero


def train_one(pools, allocators, shortlists, roles, protocol, mode, seed, config, group, output,
              init_keep=.7, eval_every=EVAL_EPOCHS):
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = make_policy(init_keep)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(seed); rng = random.Random(seed)
    k = 1 if mode == 'global1' else group
    training = [i for i, rp in enumerate(pools) if roles[rp.pool.trajectory_id] == 'train']
    baseline, history, status = 0., [], 'completed'
    with (output / 'initial.pt').open('xb') as f:
        torch.save(initial, f)
    for epoch in range(config.epochs):
        order = list(training); rng.shuffle(order); model.train()
        episodes, groups = [], []
        for i in order:
            tid = pools[i].pool.trajectory_id
            for j in range(k):
                episodes.append(sample(model, pools[i], allocators[i], protocol, action_generator=generator,
                                       ordering_seed=order_seed(seed, epoch * k + j, tid), shortlist=shortlists[i]))
                groups.append(tid)
        rewards = [e['reward'] for e in episodes]
        adv = advantages(rewards, groups, mode, baseline)
        loss = group_loss(episodes, adv, config.entropy)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        optimizer.zero_grad(); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        within, between, zero = spread(rewards, groups)
        history.append({'epoch': epoch + 1, 'episodes': len(episodes), 'reward': float(np.mean(rewards)),
                        'baseline_before': baseline, 'advantage_abs_mean': float(np.abs(adv).mean()),
                        'reward_within_query_std': within, 'reward_between_query_std': between,
                        'zero_spread_groups': zero, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                        'clipped': bool(norm > config.grad_clip),
                        **{d: float(np.mean([e['diagnostics'][d] for e in episodes])) for d in episodes[0]['diagnostics']},
                        'coverage': float(np.mean([e['score']['coverage'] for e in episodes])),
                        'redundancy': float(np.mean([e['score']['redundancy'] for e in episodes])),
                        'empty_fraction': float(np.mean([not e['score']['selected_count'] for e in episodes]))})
        baseline = config.baseline_decay * baseline + (1 - config.baseline_decay) * float(np.mean(rewards))
        if len(history) >= 3 and np.mean([h['empty_fraction'] for h in history[-3:]]) > .5:
            status = 'stopped_empty_collapse'
        if (epoch + 1) % eval_every == 0 or epoch + 1 == config.epochs or status != 'completed':
            with (output / f'epoch-{epoch + 1:03d}.pt').open('xb') as f:
                torch.save({'model': model.state_dict(), 'epoch': epoch + 1, 'protocol': protocol,
                            'mode': mode, 'group': k, 'seed': seed, 'status': status}, f)
        if status != 'completed':
            break
    with (output / 'history.json').open('x') as f:
        json.dump(history, f, indent=2, allow_nan=False)
    return initial, model, history, status


def greedy_shortlist(rp, allocator, shortlist):
    """Add the shortlist page with the largest proxy gain until no page improves it."""
    chosen, best = [], -1.
    ids = lambda ix: [rp.pool.pages[i].id for i in ix]
    while True:
        options = [(total_reward(rp.score(ids(chosen + [i]), allocator.tokenizer, allocator.budget), WEIGHTS),
                    rp.pool.pages[i].id, i)
                   for i in shortlist if i not in chosen and allocator.count(chosen + [i]) <= allocator.budget]
        if not options:
            break
        reward, _, i = max(options)
        if reward <= best:
            break
        chosen.append(i); best = reward
    return ids(chosen)


def baselines(pools, allocators, shortlists, tokenizer, budget, threshold=.35):
    rows = []
    for rp, allocator, shortlist in zip(pools, allocators, shortlists):
        picks = [('topk', None, select(rp.pool, 'topk', tokenizer, budget)[1]),
                 ('threshold', None, select(rp.pool, 'threshold', tokenizer, budget, threshold=threshold)[1]),
                 ('shortlist_all', None, [rp.pool.pages[i].id for i in shortlist]),
                 ('shortlist_greedy', None, greedy_shortlist(rp, allocator, shortlist))]
        picks += [('random', s, select(rp.pool, 'random', tokenizer, budget, seed=s)[1]) for s in EVAL_SEEDS]
        for method, seed, selected in picks:
            score = rp.score(selected, tokenizer, budget)
            rows.append(score | {'reward': total_reward(score, WEIGHTS), 'method': method, 'evaluation_seed': seed,
                                 'training_seed': None, 'constraint_violations': int(not score['feasible'])})
    return rows


def mask_changes(rows, qids):
    """Pages whose deterministic decision differs between the trained and untrained policy."""
    out = []
    by = {}
    for r in rows:
        if r['mode'] == 'deterministic':
            by.setdefault((r['protocol'], r['advantage'], r['training_seed'], r['trajectory_id']), {})[r['trained']] = set(r['selected_ids'])
    for (protocol, mode, seed, tid), masks in sorted(by.items()):
        if True in masks and False in masks:
            out.append({'protocol': protocol, 'advantage': mode, 'training_seed': seed, 'query_id': qids[tid],
                        'changed_pages': len(masks[True] ^ masks[False]),
                        'trained_kept': len(masks[True]), 'untrained_kept': len(masks[False])})
    return out


def diagnose(output, split_path, run_dirs, budget=50000, group=8, allow_test=False):
    """Phase 1: pool shape, decisions per rollout and reward spread of the untrained policy."""
    if output.exists():
        raise FileExistsError(output)
    tokenizer = Tokenizer()
    pools, allocators, roles, qids, sources = load_drg(split_path, run_dirs, tokenizer, budget, allow_test=allow_test)
    shortlists = [make_shortlist(p, a) for p, a in zip(pools, allocators)]
    torch.manual_seed(0); model = Policy(); model.eval()
    rows, spread_rows = [], []
    for protocol in PROTOCOLS:
        rewards, groups = [], []
        for rp, allocator, shortlist in zip(pools, allocators, shortlists):
            tid = rp.pool.trajectory_id
            eps = [sample(model, rp, allocator, protocol, action_generator=torch.Generator().manual_seed(1000 + s),
                          ordering_seed=order_seed(1000 + s, 0, tid), shortlist=shortlist, track_grad=False)
                   for s in range(group)]
            rewards += [e['reward'] for e in eps]; groups += [tid] * len(eps)
            rows.append({'protocol': protocol, 'query_id': qids[tid], 'role': roles[tid],
                         'pages': len(rp.pool.pages), 'pool_tokens': int(allocator.tokens.sum()),
                         'median_page_tokens': int(np.median(allocator.tokens)),
                         'clusters': len(set(rp.clusters)), 'hosts': len(set(rp.domains)),
                         'subquestions': rp.targets.shape[0] - 1, 'shortlist': len(shortlist),
                         'sampled_actions': float(np.mean([e['diagnostics']['sampled_actions'] for e in eps])),
                         'forced_actions': float(np.mean([e['diagnostics']['forced_actions'] for e in eps])),
                         'reward_mean': float(np.mean([e['reward'] for e in eps])),
                         'reward_std': float(np.std([e['reward'] for e in eps])),
                         **{c: float(np.mean([e['score'][c] for e in eps]))
                            for c in ('coverage', 'diversity', 'redundancy', 'token_cost')}})
        within, between, zero = spread(rewards, groups)
        spread_rows.append({'protocol': protocol, 'within_query_std': within, 'between_query_std': between,
                            'zero_spread_queries': zero})
    output.mkdir(parents=True)
    result = {'reward_version': VERSION, 'weights': asdict(WEIGHTS), 'budget': budget, 'group': group,
              'sources': sources, 'per_query': rows, 'spread': spread_rows, 'api_calls': 0}
    with (output / 'diagnostics.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    write_csv(output / 'diagnostics.csv', rows)
    return result


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def run(output, split_path, run_dirs, *, protocols=('shortlist', 'random_order'), modes=MODES,
        seeds=(0, 1, 2, 3, 4), group=8, config=Config(), init_keep=.7, eval_every=EVAL_EPOCHS):
    if output.exists():
        raise FileExistsError(output)
    if not set(protocols) <= set(PROTOCOLS) or not set(modes) <= set(MODES) or group < 2:
        raise ValueError('Unknown protocol/mode or group < 2')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); started = time.perf_counter()
    tokenizer = Tokenizer()
    pools, allocators, roles, qids, sources = load_drg(split_path, run_dirs, tokenizer, config.budget)
    shortlists = [make_shortlist(p, a) for p, a in zip(pools, allocators)]
    output.mkdir(parents=True)
    manifest = {'config': asdict(config), 'init_keep': init_keep, 'eval_every': eval_every, 'protocols': list(protocols), 'modes': list(modes),
                'training_seeds': list(seeds), 'evaluation_seeds': EVAL_SEEDS, 'group': group, 'eps': EPS,
                'weights': asdict(WEIGHTS), 'reward_version': VERSION, 'features': FEATURES,
                'split': str(split_path), 'split_sha256': sha(Path(split_path)), 'sources': sources,
                'matching': 'global/center/standardize share episodes per update; global1 is the DRB K=1 loop',
                'checkpoint': 'final epoch is the prespecified endpoint; curves are reported, not selected on',
                'code_hashes': {n: sha(Path(__file__).with_name(n)) for n in
                                ['drg.py', 'learning_signal.py', 'reinforce.py', 'reward.py', 'core.py']},
                'api_calls': 0, 'device': 'cpu', 'threads': 1}
    with (output / 'config.json').open('x') as f:
        json.dump(manifest, f, indent=2)
    base = baselines(pools, allocators, shortlists, tokenizer, config.budget)
    final, curves, statuses = [], [], []
    for protocol in protocols:
        for mode in modes:
            for seed in seeds:
                folder = output / protocol / mode / f'seed-{seed}'
                initial, model, history, status = train_one(pools, allocators, shortlists, roles, protocol, mode,
                                                            seed, config, group, folder, init_keep, eval_every)
                statuses.append({'protocol': protocol, 'mode': mode, 'seed': seed, 'epochs': len(history),
                                 'status': status})
                print(json.dumps(statuses[-1]), flush=True)
                tag = {'advantage': mode}
                cold = Policy(); cold.load_state_dict(initial)
                for deterministic in (False, True):
                    rows = eval_model(cold, pools, allocators, shortlists, protocol, seed, 0, False, deterministic)
                    final += [r | tag for r in rows]
                    if not deterministic:
                        curves += [r | tag | {'epoch': 0, 'training_seed': seed} for r in aggregate(rows, roles)]
                for checkpoint in sorted(folder.glob('epoch-*.pt')):
                    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                    model.load_state_dict(state['model'])
                    rows = eval_model(model, pools, allocators, shortlists, protocol, seed, state['epoch'], True)
                    curves += [r | tag | {'epoch': state['epoch'], 'training_seed': seed} for r in aggregate(rows, roles)]
                    if state['epoch'] == len(history):
                        final += [r | tag for r in rows]
                        final += [r | tag for r in eval_model(model, pools, allocators, shortlists, protocol, seed,
                                                              state['epoch'], True, True)]
    summary = [r | {'protocol': None, 'advantage': None} for r in aggregate(base, roles)]
    for protocol in protocols:
        for mode in modes:
            rows = [r for r in final if r['protocol'] == protocol and r['advantage'] == mode]
            summary += [r | {'protocol': protocol, 'advantage': mode} for r in aggregate(rows, roles)]

    def save(name, value):
        with (output / name).open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False)
    save('results.json', final); save('baselines.json', base); save('summary.json', summary)
    save('curves.json', curves); save('mask_changes.json', mask_changes(final, qids))
    save('status.json', {'runs': statuses, 'wall_seconds': time.perf_counter() - started, 'api_calls': 0})
    flat = [{k: v for k, v in r.items() if k != 'per_query'} for r in summary]
    write_csv(output / 'comparison.csv', flat)
    write_csv(output / 'per_query.csv', [dict(protocol=r['protocol'], advantage=r['advantage'], split=r['split'],
                                              method=r['method'], query_id=qids[q['trajectory_id']], **q)
                                         for r in summary for q in r['per_query']])
    return flat


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('diagnose', 'train'):
        s = sub.add_parser(name)
        s.add_argument('--output', type=Path, required=True)
        s.add_argument('--split', type=Path, default=Path('data/benchmarks/deep_research_gym/rl_small_split.json'))
        s.add_argument('--run-dir', type=Path, action='append', required=True,
                       help='Run directory holding gpt_researcher/<id>.json; repeat, first match wins')
        s.add_argument('--group', type=int, default=8, help='Rollouts per query per update')
    sub.choices['diagnose'].add_argument('--allow-test', action='store_true')
    t = sub.choices['train']
    t.add_argument('--protocols', nargs='+', default=['shortlist', 'random_order'], choices=PROTOCOLS)
    t.add_argument('--modes', nargs='+', default=list(MODES), choices=MODES)
    t.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    t.add_argument('--epochs', type=int, default=20, help=f'Updates per run, 1..{MAX_EPOCHS}')
    t.add_argument('--init-keep', type=float, default=.7, help='Initial p(KEEP); .7 is the DRB setting')
    t.add_argument('--eval-every', type=int, default=EVAL_EPOCHS, help='Checkpoint and evaluate every N updates')
    a = p.parse_args()

    def denied(*args, **kwargs):
        raise RuntimeError('Offline DRG study forbids network')
    socket.socket.connect = denied; socket.create_connection = denied
    if a.command == 'diagnose':
        result = diagnose(a.output, a.split, a.run_dir, group=a.group, allow_test=a.allow_test)
        print(json.dumps(result['spread'], indent=2))
    else:
        print(json.dumps(run(a.output, a.split, a.run_dir, protocols=a.protocols, modes=a.modes,
                             seeds=a.seeds, group=a.group, config=LongConfig(epochs=a.epochs),
                             init_keep=a.init_keep, eval_every=a.eval_every), indent=2))


if __name__ == '__main__':
    main()
