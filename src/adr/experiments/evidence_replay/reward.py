"""Offline proxy v1. No report, judge output, or future retrieval enters scoring."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from .core import Pool, Tokenizer, block, load_pool

VERSION = 'page-proxy-v2-cluster-support'


@dataclass(frozen=True)
class Weights:
    coverage: float = .70
    diversity: float = .10
    redundancy: float = .15
    token_cost: float = .05

    def __post_init__(self):
        values = list(asdict(self).values())
        if not all(math.isfinite(v) and v >= 0 for v in values) or not math.isclose(sum(values), 1):
            raise ValueError('Weights must be finite, nonnegative and sum to one')
        if self.coverage < .5 or max(values[1:]) > .25:
            raise ValueError('Coverage must dominate; auxiliary weights must be <= .25')


CANDIDATES = {
    'coverage_first': Weights(),
    'anti_redundancy': Weights(.60, .10, .20, .10),
    'source_balance': Weights(.60, .15, .15, .10),
}


def total_reward(components, weights=Weights()):
    if not components['feasible'] or not components['selected_count']:
        return -1.0  # constraint sentinel, outside all nonempty feasible weighted rewards
    return (weights.coverage * components['coverage'] + weights.diversity * components['diversity']
            - weights.redundancy * components['redundancy'] - weights.token_cost * components['token_cost'])


@dataclass
class RewardPool:
    pool: Pool
    vectors: np.ndarray
    targets: np.ndarray
    affinities: np.ndarray
    pair_cosines: np.ndarray
    domains: tuple[str, ...]
    duplicate_keys: tuple[str, ...]
    clusters: np.ndarray
    support_groups: int = 3
    duplicate_threshold: float = .90

    @classmethod
    def load(cls, path: Path):
        pool = load_pool(path)
        with np.load(pool.provenance['embeddings_path'], allow_pickle=False) as z:
            if 'subquestions' not in z or len(z['subquestions']) == 0:
                raise ValueError('Proxy requires saved subquestion embeddings')
            lookup = dict(zip(z['ids'].tolist(), np.asarray(z['vectors'], dtype=float)))
            return cls.from_arrays(pool, np.array([lookup[p.id] for p in pool.pages]),
                                   np.concatenate([z['query'], z['subquestions']]))

    @classmethod
    def from_arrays(cls, pool, vectors, targets, duplicate_threshold=.90, support_groups=3):
        vectors, targets = np.asarray(vectors, dtype=float), np.asarray(targets, dtype=float)
        if (vectors.ndim != 2 or targets.ndim != 2 or len(vectors) != len(pool.pages)
                or len(targets) < 2 or vectors.shape[1] != targets.shape[1]
                or not np.isfinite(vectors).all() or not np.isfinite(targets).all()
                or np.any(np.linalg.norm(vectors, axis=1) == 0)
                or np.any(np.linalg.norm(targets, axis=1) == 0)):
            raise ValueError('Missing, nonfinite, zero or misaligned embeddings')
        if not 0 <= duplicate_threshold < 1:
            raise ValueError('Duplicate threshold must be in [0,1)')
        if not isinstance(support_groups,int) or support_groups<1:
            raise ValueError('Support groups must be a positive integer')
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        targets = targets / np.linalg.norm(targets, axis=1, keepdims=True)
        affinities = np.clip(vectors @ targets.T, 0, 1)
        best = affinities.max(axis=0)
        if np.any(best <= 0):
            raise ValueError('A coverage target has no positive evidence similarity')
        affinities /= best
        domains = tuple((urlsplit(p.url).hostname or 'unknown').lower().removeprefix('www.') for p in pool.pages)
        keys = tuple(hashlib.sha256(' '.join(p.text.split()).casefold().encode()).hexdigest() for p in pool.pages)
        pair_cosines=np.clip(vectors @ vectors.T,0,1)
        parent=list(range(len(pool.pages)))
        def find(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]];i=parent[i]
            return i
        for i in range(len(parent)):
            for j in range(i):
                if pair_cosines[i,j]>=duplicate_threshold or keys[i]==keys[j]:
                    a,b=find(i),find(j)
                    if a!=b:parent[max(a,b)]=min(a,b)
        clusters=np.array([find(i) for i in range(len(parent))])
        return cls(pool, vectors, targets, affinities, pair_cosines, domains, keys,
                   clusters, support_groups, duplicate_threshold)

    def score(self, selected_ids, tokenizer: Tokenizer, budget=50000):
        if budget <= 0:
            raise ValueError('Budget must be positive')
        selected = set(selected_ids)
        if selected - {p.id for p in self.pool.pages}:
            raise ValueError('Unknown selected ID')
        ix = [i for i, p in enumerate(self.pool.pages) if p.id in selected]
        pages = sorted((self.pool.pages[i] for i in ix), key=lambda p: (p.round, p.id))
        context = '\n\n'.join(block(p) for p in pages)
        tokens = tokenizer.count(context)
        coverage = diversity = redundancy = max_coverage = 0.
        if ix:
            best = self.affinities[ix].max(axis=0)
            max_coverage = .5 * float(best[0]) + .5 * float(best[1:].mean())
            all_groups=np.array([self.affinities[self.clusters==c].max(axis=0) for c in np.unique(self.clusters)])
            selected_groups=np.array([self.affinities[[i for i in ix if self.clusters[i]==c]].max(axis=0)
                                      for c in np.unique(self.clusters[ix])])
            # Sum the strongest up-to-k independent groups per target. Missing
            # groups contribute zero; normalize by the best available pool support.
            denominator=np.sort(all_groups,axis=0)[-self.support_groups:].sum(axis=0)
            target_support=np.sort(selected_groups,axis=0)[-self.support_groups:].sum(axis=0)/denominator
            coverage=.5*float(target_support[0])+.5*float(target_support[1:].mean())
            # Source entropy uses provenance, not pairwise semantic distance. This
            # keeps the source-diversity bonus distinct from near-duplicate penalty.
            from collections import Counter
            counts = np.array(list(Counter(self.domains[i] for i in ix).values()), dtype=float)
            probabilities = counts / counts.sum()
            domain_count = len(set(self.domains))
            entropy = -float(np.sum(probabilities * np.log(probabilities)))
            diversity = coverage * entropy / math.log(domain_count) if domain_count > 1 else 0.
            if len(ix) > 1:
                sims = self.pair_cosines[np.ix_(ix, ix)].copy()
                for a, i in enumerate(ix):
                    for b, j in enumerate(ix):
                        if i != j and self.duplicate_keys[i] == self.duplicate_keys[j]:
                            sims[a, b] = 1.
                np.fill_diagonal(sims, 0)
                redundancy = float(np.clip((sims.max(axis=1) - self.duplicate_threshold) /
                                            (1 - self.duplicate_threshold), 0, 1).mean())
        identity = {'trajectory_sha256': self.pool.provenance.get('trajectory_sha256'),
                    'trajectory_id': self.pool.trajectory_id, 'selected_ids': sorted(selected),
                    'budget': budget, 'encoding': tokenizer.name}
        return {'proxy_version': VERSION, 'selection_id': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                **identity, 'selected_count': len(ix), 'context_tokens': tokens,
                'context_sha256': hashlib.sha256(context.encode()).hexdigest(),
                'coverage': float(np.clip(coverage, 0, 1)), 'diversity': float(np.clip(diversity, 0, 1)),
                'max_coverage_ablation':max_coverage,'support_groups':self.support_groups,
                'selected_clusters':len(np.unique(self.clusters[ix])) if ix else 0,
                'pool_clusters':len(np.unique(self.clusters)),
                'redundancy': redundancy, 'token_cost': min(1., tokens / budget),
                'unclipped_token_ratio': tokens / budget, 'feasible': tokens <= budget,
                'constraint': 'empty' if not ix else ('over_budget' if tokens > budget else None),
                'mean_query_cosine': float(np.mean(self.vectors[ix] @ self.targets[0])) if ix else None,
                'duplicate_threshold': self.duplicate_threshold}


def pair_quality(proxy_records, quality_records):
    """Join future outcomes by exact selection identity, never by page or rank.

    Each quality record supplies selection_id, context_sha256, quality_version,
    and metrics. This function scores nothing and makes no API calls.
    """
    lookup = {r['selection_id']: r for r in proxy_records}
    if len(lookup) != len(proxy_records):
        raise ValueError('Duplicate proxy identities; deduplicate repeated masks first')
    seen, pairs = set(), []
    for outcome in quality_records:
        key = outcome['selection_id']
        if key not in lookup or key in seen:
            raise ValueError('Unknown or duplicate quality identity')
        proxy = lookup[key]
        if outcome['context_sha256'] != proxy['context_sha256'] or not outcome.get('quality_version'):
            raise ValueError('Context mismatch or missing quality version')
        if not outcome.get('metrics') or not all(isinstance(v, (int,float)) and not isinstance(v, bool) and math.isfinite(v)
                                                  for v in outcome['metrics'].values()):
            raise ValueError('Quality metrics must be finite numbers')
        seen.add(key)
        pairs.append({'proxy': proxy, 'quality': outcome})
    return pairs


def score_saved_replay(replay: Path, tokenizer=None):
    """Attach a proxy identity to a saved mask without reading its report/outcome."""
    plan = json.loads((replay / 'plan.json').read_bytes())
    rp = RewardPool.load(Path(plan['provenance']['trajectory_path']))
    for field in ['trajectory_sha256', 'embeddings_sha256']:
        if rp.pool.provenance[field] != plan['provenance'][field]:
            raise ValueError('Saved replay source hash mismatch')
    tokenizer = tokenizer or Tokenizer(plan['config']['encoding'])
    if tokenizer.name != plan['config']['encoding']:
        raise ValueError('Saved replay tokenizer mismatch')
    score = rp.score(plan['selected_ids'], tokenizer, plan['config']['budget'])
    if (hashlib.sha256((replay/'context.txt').read_bytes()).hexdigest() != score['context_sha256']
            or plan['context_sha256'] != score['context_sha256']
            or plan['context_tokens'] != score['context_tokens']):
        raise ValueError('Saved replay context mismatch')
    return score | {'reward': total_reward(score),
                    'rewards': {name: total_reward(score,w) for name,w in CANDIDATES.items()}}
