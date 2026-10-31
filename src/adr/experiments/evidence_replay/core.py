"""Pure replay construction. No model clients, retrieval, or network operations."""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VERSION = 'evidence-replay-phase1-v1'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Tokenizer:
    """Use hash-verified local encoding data; never download or write a cache."""
    def __init__(self, encoding='o200k_base'):
        import tiktoken
        import tiktoken.load
        from unittest.mock import patch
        # tiktoken otherwise downloads missing encoding files implicitly.
        def offline(url, expected_hash=None):
            import importlib.util
            spec = importlib.util.find_spec('litellm')
            key = hashlib.sha1(url.encode()).hexdigest()
            source = (Path(spec.origin).parent / 'litellm_core_utils/tokenizers' / key
                      if spec and spec.origin else Path('/nonexistent'))
            if not expected_hash or not source.is_file():
                raise RuntimeError('Verified local tokenizer unavailable; replay never downloads tokenizers')
            data = source.read_bytes()
            # Some Windows installs check the bundled file out with CRLF line
            # endings; accept it only if the LF form matches the pinned hash.
            if hashlib.sha256(data).hexdigest() != expected_hash and b'\r\n' in data:
                data = data.replace(b'\r\n', b'\n')
            if hashlib.sha256(data).hexdigest() != expected_hash:
                raise ValueError('Local tokenizer hash mismatch')
            return data
        with patch.object(tiktoken.load, 'read_file_cached', offline):
            self.encoding = tiktoken.get_encoding(encoding)
        self.name = encoding

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))


@dataclass(frozen=True)
class Page:
    id: str
    text: str
    url: str
    subquery: str
    depth: int
    round: int
    similarity: float


@dataclass(frozen=True)
class Pool:
    query: str
    trajectory_id: str
    pages: tuple[Page, ...]
    provenance: dict


def load_pool(path: Path) -> Pool:
    path = path.resolve()
    raw = json.loads(path.read_text(encoding='utf-8'))
    evidence = raw.get('evidence')
    if not raw.get('report') or not isinstance(evidence, dict) or not evidence:
        raise ValueError('Require a completed trajectory with nonempty evidence')
    emb_path = path.with_name(path.stem + '_emb.npz')
    with np.load(emb_path, allow_pickle=False) as z:
        ids = z['ids'].tolist()
        vectors = np.array(z['vectors'], dtype=float)
        query = np.array(z['query'], dtype=float)
    if len(set(ids)) != len(ids) or set(ids) != set(evidence):
        raise ValueError('Evidence/embedding ID mismatch')
    if (vectors.ndim != 2 or vectors.shape[0] != len(ids) or
            query.shape != (1, vectors.shape[1]) or not np.isfinite(vectors).all() or
            not np.isfinite(query).all()):
        raise ValueError('Invalid embedding shapes or nonfinite embeddings')
    norms = np.linalg.norm(vectors, axis=1)
    qnorm = np.linalg.norm(query[0])
    if qnorm == 0 or np.any(norms == 0):
        raise ValueError('Zero embedding norm')
    scores = dict(zip(ids, (vectors @ query[0] / (norms * qnorm)).tolist()))
    pages = []
    for iid in sorted(evidence):
        e = evidence[iid]
        if e.get('item_id') != iid or e.get('compressed'):
            raise ValueError('Require matching immutable, uncompressed page IDs')
        if e.get('source_type') != 'page' or not e.get('content', '').strip() or not e.get('source_url'):
            raise ValueError('Require page text and source provenance')
        pages.append(Page(iid, e['content'], e['source_url'], e['source_subquery'],
                          int(e['tree_depth']), int(e['retrieval_round']), float(scores[iid])))
    return Pool(raw['query'], raw['query_id'], tuple(pages), {
        'trajectory_path': str(path), 'trajectory_sha256': sha(path),
        'embeddings_path': str(emb_path), 'embeddings_sha256': sha(emb_path),
        'embedding_dimensions': vectors.shape[1],
        'as_of_date': datetime.fromtimestamp(
            max((r['timestamp'] for r in raw.get('rounds', [])), default=0),
            tz=timezone.utc).date().isoformat(),
        'embedding_model': 'not certified by NPZ; consult source run configuration',
    })


def block(page: Page) -> str:
    # Only selected text, ID and source URL enter synthesis. Metadata is audit-only.
    return f'[Evidence {page.id}]\nSource: {page.url}\n{page.text}\n[/Evidence {page.id}]'


def build_context(pool: Pool, selected_ids, tokenizer: Tokenizer, budget: int) -> tuple[str, list[str], int]:
    if budget < 0:
        raise ValueError('Budget must be nonnegative')
    selected = set(selected_ids)
    known = {p.id for p in pool.pages}
    if selected - known:
        raise ValueError('Unknown selected evidence IDs')
    # Canonical order is independent of selector ranking or input-set iteration.
    pages = sorted((p for p in pool.pages if p.id in selected), key=lambda p: (p.round, p.id))
    context = '\n\n'.join(block(p) for p in pages)
    tokens = tokenizer.count(context)
    if tokens > budget:
        raise ValueError(f'Selected context exceeds budget: {tokens} > {budget}; no truncation applied')
    return context, [p.id for p in pages], tokens


def select(pool: Pool, method: str, tokenizer: Tokenizer, budget=50000, seed=17, threshold=.35):
    if not -1 <= threshold <= 1:
        raise ValueError('Cosine threshold must be in [-1, 1]')
    ranked = sorted(pool.pages, key=lambda p: (-p.similarity, p.id))
    if method == 'random':
        ranked = sorted(pool.pages, key=lambda p: p.id)
        random.Random(seed).shuffle(ranked)
    elif method == 'threshold':
        # Explicit distinction from top-k: threshold eligibility, then chronological
        # first-fit overflow, skipping each item that cannot fit (no truncation).
        ranked = sorted((p for p in pool.pages if p.similarity >= threshold),
                        key=lambda p: (p.round, p.id))
    elif method != 'topk':
        raise ValueError('Unknown selector')
    kept = []
    for p in ranked:
        try:
            build_context(pool, [*kept, p.id], tokenizer, budget)
        except ValueError as e:
            if 'exceeds budget' not in str(e):
                raise
        else:
            kept.append(p.id)
    return build_context(pool, kept, tokenizer, budget)


def make_plan(pool, tokenizer, *, budget=50000, method='topk', seed=17, threshold=.35, selected_ids=None):
    if selected_ids is None:
        context, ids, tokens = select(pool, method, tokenizer, budget, seed, threshold)
    else:
        context, ids, tokens = build_context(pool, selected_ids, tokenizer, budget)
        method = 'explicit_ids'
    import importlib.metadata
    return {
        'version': VERSION, 'query': pool.query, 'trajectory_id': pool.trajectory_id,
        'selected_ids': ids, 'mask': {p.id: p.id in set(ids) for p in pool.pages},
        'context': context, 'context_tokens': tokens,
        'context_sha256': hashlib.sha256(context.encode()).hexdigest(),
        'selected_provenance': [vars(p) | {'text': None} for p in pool.pages if p.id in set(ids)],
        'config': {'selector': method, 'seed': seed, 'threshold': threshold,
                   'budget': budget, 'encoding': tokenizer.name, 'granularity': 'page',
                   'as_of_date': pool.provenance.get('as_of_date', '1970-01-01'),
                   'budget_scope': 'formatted evidence context; excludes question and prompt overhead',
                   'overflow': 'skip nonfitting whole pages; never truncate',
                   'threshold_order': 'retrieval_round then ID', 'context_order': 'retrieval_round then ID'},
        'provenance': pool.provenance | {'core_sha256': sha(Path(__file__)),
            'tiktoken_version': importlib.metadata.version('tiktoken'),
            'numpy_version': np.__version__},
    }


def save_plan(plan: dict, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    # Binary UTF-8 serialization avoids platform-dependent newline translation.
    (output / 'context.txt').write_bytes(plan['context'].encode('utf-8'))
    metadata = {k: v for k, v in plan.items() if k != 'context'}
    (output / 'plan.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
