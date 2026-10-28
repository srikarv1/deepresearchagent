"""ReplayGym: RL for research-tree expansion with exact replay rewards.

Episode (one DRG question, one cached research tree):
  for each top-level branch b (in run order):
      decide RESEARCH b?            -> if no, the branch is skipped (depth 0)
      observe b's learnings; decide GO DEEPER (b's children)?     (depth 2)
      observe children;      decide GO DEEPER (b's grandchildren)? (depth 3)
  terminal: report = Write(question, context(researched nodes)) -> DRG judge
  reward   = 0.5*quality + 0.5*KPR - lam * research_tokens / 100K

Every decision is taken with only information available at that point in a
live run (the branch's own findings so far and what the other branches found).
Rewards are exact counterfactuals (see tree.py), memoized per distinct sub-tree.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from adr.replay.judge import judge_reports
from adr.replay.synth import apply_agent_env, synthesize_many
from adr.replay.tree import ResearchTree

ROOT_RUNS = Path("runs")
TAG = os.environ.get("REPLAY_TAG", "gpt5mini")  # keeps backbones' data separate
DATA = ROOT_RUNS / "replaygym" / TAG
N_FEAT = 10


# --------------------------------------------------------------------------- data
def find_trees(qids: list[str] | None = None) -> dict[str, Path]:
    """Latest node-cached trajectory per DRG query id for this backbone (run name tag)."""
    out: dict[str, tuple[float, Path]] = {}
    pattern = "*-replaygym-bedrock-*/gpt_researcher/*.json" if TAG == "bedrock" else "*-replay*[0-9]/gpt_researcher/*.json"
    for p in ROOT_RUNS.glob(pattern):
        if p.name.endswith("_emb.json") or not p.stem.isdigit():
            continue
        if qids is not None and p.stem not in qids:
            continue
        try:
            if not json.loads(p.read_text(encoding="utf-8")).get("nodes"):
                continue
        except Exception:
            continue
        mt = p.stat().st_mtime
        if p.stem not in out or mt > out[p.stem][0]:
            out[p.stem] = (mt, p)
    return {q: v[1] for q, v in out.items()}


def _local_embed(texts: list[str]) -> np.ndarray:
    """bge-small via scripts/embed_server.py if it is up (one shared model in RAM),
    else in-process."""
    import requests

    try:
        r = requests.post("http://127.0.0.1:8323/v1/embeddings", json={"input": texts}, timeout=300)
        r.raise_for_status()
        return np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
    except Exception:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("BAAI/bge-small-en-v1.5").encode(texts).astype(np.float32)


LOCAL_EMBED_MODEL = "bge-small-en-v1.5"
OPENAI_EMBED_MODEL = "text-embedding-3-small"


def embed_model_name() -> str:
    """Model behind the EpisodeState features. The live ExpansionPolicy embeds with
    the bge-small server (GR_REPLAYGYM_EMBED_URL), so only REPLAY_EMBED=local
    trains on the features it will see."""
    return LOCAL_EMBED_MODEL if os.environ.get("REPLAY_EMBED") == "local" else OPENAI_EMBED_MODEL


class Embedder:
    """Goal and learnings vectors per node, cached per query id and per model."""

    def __init__(self, cache_dir: Path = DATA / "emb"):
        self.model = embed_model_name()
        # One cache per model, so switching REPLAY_EMBED never reuses vectors
        # computed by the other model.
        self.cache_dir = cache_dir / self.model
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def for_tree(self, qid: str, tree: ResearchTree) -> dict[str, np.ndarray]:
        path = self.cache_dir / f"{qid}.npz"
        if path.exists():
            d = np.load(path)
            return {k: d[k] for k in d.files}
        texts = [tree.query] + [tree.nodes[n].research_goal or tree.nodes[n].query for n in tree.order]
        texts += [" ".join(tree.nodes[n].learnings) or "(none)" for n in tree.order]
        if self.model == LOCAL_EMBED_MODEL:
            m = _local_embed([t[:4000] for t in texts])
        else:
            from openai import OpenAI

            resp = OpenAI().embeddings.create(model=OPENAI_EMBED_MODEL, input=[t[:8000] for t in texts])
            m = np.array([e.embedding for e in resp.data], dtype=np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        k = len(tree.order)
        out = {"query": m[0], "goal": m[1 : 1 + k], "learn": m[1 + k :]}
        np.savez(path, **out)
        return out


# ------------------------------------------------------------------------- reward
class RewardService:
    """Exact replay reward with a persistent memo (one judge per distinct sub-tree)."""

    def __init__(self, lam: float = 2.0, cache_dir: Path = DATA / "memo", concurrency: int = int(os.environ.get("REPLAY_SYNTH_CONC", "2"))):
        self.lam = lam
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.concurrency = concurrency
        self._memo: dict[str, dict[str, dict[str, Any]]] = {}
        apply_agent_env()

    @staticmethod
    def key(nodes: frozenset[str]) -> str:
        return hashlib.sha1(",".join(sorted(nodes)).encode()).hexdigest()[:16]

    def _load(self, qid: str) -> dict[str, dict[str, Any]]:
        if qid not in self._memo:
            p = self.cache_dir / f"{qid}.json"
            self._memo[qid] = json.loads(p.read_text()) if p.exists() else {}
        return self._memo[qid]

    def _save(self, qid: str) -> None:
        (self.cache_dir / f"{qid}.json").write_text(json.dumps(self._memo[qid], indent=0))

    def scores(self, qid: str, tree: ResearchTree, subtrees: list[frozenset[str]]) -> list[dict[str, Any]]:
        memo = self._load(qid)
        todo = {self.key(s): s for s in subtrees if self.key(s) not in memo}
        if todo:
            rep_dir = self.cache_dir / "reports" / qid
            reports = {}
            for k in list(todo):  # reuse reports written before a restart
                f = rep_dir / f"{k}.md"
                if f.exists() and f.stat().st_size > 2000:
                    reports[k] = f.read_text(encoding="utf-8")
            contexts = {k: tree.context(s) for k, s in todo.items() if s and k not in reports}
            if contexts:
                reports.update(asyncio.run(synthesize_many(tree.query, contexts, self.concurrency)))
            for k, s in todo.items():
                if not s:  # nothing researched: an empty report scores 0
                    memo[k] = {"quality": 0.0, "kpr": 0.0, "tokens": 0, "n": 0}
            rep_dir.mkdir(parents=True, exist_ok=True)
            for k, r in reports.items():
                if r.strip():
                    (rep_dir / f"{k}.md").write_text(r, encoding="utf-8")
            pending = {k: r for k, r in reports.items() if r.strip()}
            work = self.cache_dir / "judge" / qid
            for attempt in range(4):
                if not pending:
                    break
                got = judge_reports(qid, tree.query, pending, work / f"b{len(memo)}_{attempt}")
                for k, sc in got.items():
                    if sc.get("quality") is not None and sc.get("kpr") is not None:
                        memo[k] = {"quality": sc["quality"], "kpr": sc["kpr"], "tokens": tree.tokens(todo[k]), "n": len(todo[k])}
                        pending.pop(k, None)
            self._save(qid)
        return [memo.get(self.key(s)) for s in subtrees]

    def report_text(self, qid: str, nodes: frozenset[str]) -> str:
        p = self.cache_dir / "reports" / qid / f"{self.key(nodes)}.md"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def reward(self, sc: dict[str, Any] | None) -> float | None:
        if sc is None:
            return None
        return 0.5 * sc["quality"] + 0.5 * sc["kpr"] - self.lam * sc["tokens"] / 1e5


# ---------------------------------------------------------------------- env logic
@dataclass
class Decision:
    feat: np.ndarray
    action: int


class EpisodeState:
    """Branch-by-branch expansion over one cached tree."""

    def __init__(self, qid: str, tree: ResearchTree, emb: dict[str, np.ndarray], n_full: int | None = None):
        self.qid, self.tree, self.emb = qid, tree, emb
        self.idx = {n: i for i, n in enumerate(tree.order)}
        self.full_tokens = max(1, tree.tokens(tree.all_nodes()))
        # Denominator of f[7]. The live policy divides by full_tree_size(breadth,
        # depth) of the agent config, not by the nodes this cached run happened to
        # produce, so pass that size for train/serve parity (default: cached nodes).
        if n_full is not None and n_full < len(tree.nodes):
            raise ValueError(
                f"n_full={n_full} < {len(tree.nodes)} cached nodes for {qid}: pass "
                "full_tree_size(breadth, depth) of the agent that produced the tree"
            )
        self.n_full = int(n_full) if n_full is not None else len(tree.nodes)
        # Mirrors the live agent's timing exactly:
        #  phase A: after the first round (all top-level branches researched), one
        #           "go deeper?" decision per top-level branch, all on the same state;
        #  phase B: depth-first per expanded branch: research its children, then one
        #           "go to full depth?" decision for that branch's children.
        self.researched: set[str] = set(tree.roots)
        self.queue: list[tuple[int, list[str]]] = [
            (2, list(tree.nodes[r].children)) for r in tree.roots if tree.nodes[r].children
        ]  # (level, node group) decisions still to make
        self._expand: list[str] = []  # roots whose children will be researched (phase B order)
        self.done = False
        if not self.queue:
            self._advance()

    def _advance(self) -> None:
        """Run deferred research until the next decision (or the end)."""
        while not self.queue:
            if not self._expand:
                self.done = True
                return
            r = self._expand.pop(0)
            kids = list(self.tree.nodes[r].children)
            self.researched.update(kids)
            grand = [g for k in kids for g in self.tree.nodes[k].children]
            if grand:
                self.queue.append((3, grand))

    def _learn_vecs(self, nodes) -> np.ndarray:
        ids = [self.idx[n] for n in nodes]
        return self.emb["learn"][ids] if ids else np.zeros((0, self.emb["learn"].shape[1]), np.float32)

    def features(self) -> np.ndarray:
        """Only what a live agent knows when deciding to expand `group`: the goals
        and findings of the parent nodes it would expand from (the children's own
        queries do not exist yet), and what every researched node has found."""
        level, group = self.queue[0]
        q = self.emb["query"]
        parents = sorted({self.tree.nodes[n].parent_node_id for n in group})
        pg = self.emb["goal"][[self.idx[p] for p in parents]]       # parents' research goals
        pv = self._learn_vecs(parents)                              # parents' findings
        rest = self._learn_vecs(self.researched - set(parents))     # everyone else's findings
        f = np.zeros(N_FEAT, np.float32)
        f[level - 1] = 1.0                                          # 1-2: which depth we'd open (f[0] unused)
        f[3] = float((pg @ q).mean())                               # branch goal relevance to question
        f[4] = float((pv @ rest.T).max()) if len(rest) else 0.0     # branch findings redundancy with others
        f[5] = float(np.clip(1.0 - f[4], 0.0, 1.0)) if len(rest) else 1.0  # novelty
        f[6] = float((pv @ q).mean())                               # findings relevance to question
        f[7] = len(self.researched) / max(1, self.n_full)           # sub-researches done (live-countable)
        f[8] = 0.0                                                  # reserved (not live-observable)
        f[9] = 1.0                                                  # bias
        return f

    def step(self, action: int) -> None:
        level, group = self.queue.pop(0)
        if level == 2:
            if action == 1:
                self._expand.append(self.tree.nodes[group[0]].parent_node_id)
        elif action == 1:  # level 3: research this branch's grandchildren
            self.researched.update(group)
        if not self.queue:
            self._advance()

    def subtree(self) -> frozenset[str]:
        return frozenset(self.researched)


class ReplayExpansionEnv(gym.Env):
    """Gymnasium wrapper (single episode at a time) around EpisodeState + RewardService."""

    metadata: dict[str, Any] = {}

    def __init__(self, trees: dict[str, ResearchTree], embs: dict[str, dict], reward: RewardService):
        self.trees, self.embs, self.reward_svc = trees, embs, reward
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (N_FEAT,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.state: EpisodeState | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        qid = (options or {}).get("qid") or self.np_random.choice(sorted(self.trees))
        self.state = EpisodeState(qid, self.trees[qid], self.embs[qid])
        return self.state.features(), {"qid": qid}

    def step(self, action):
        s = self.state
        s.step(int(action))
        if not s.done:
            return s.features(), 0.0, False, False, {}
        sc = self.reward_svc.scores(s.qid, s.tree, [s.subtree()])[0]
        r = self.reward_svc.reward(sc)
        return np.zeros(N_FEAT, np.float32), float(r if r is not None else 0.0), True, False, {"score": sc}


# ------------------------------------------------------------------------- policy
def rollout(state: EpisodeState, act: Callable[[np.ndarray], int]) -> list[Decision]:
    trace = []
    while not state.done:
        f = state.features()
        a = act(f)
        trace.append(Decision(f, a))
        state.step(a)
    return trace


def fixed_policy(kind: str) -> Callable[[EpisodeState], Callable[[np.ndarray], int]]:
    """Baselines: full tree, first level only, first two levels."""
    max_level = {"full": 3, "d1": 1, "d2": 2}[kind]

    def make(state: EpisodeState):
        return lambda f: int(np.argmax(f[:3]) + 1 <= max_level)

    return make


# ------------------------------------------------------------ pairwise reward
PAIR_PROMPT = """You are an expert evaluator of research reports.
Question: {question}

Report A:
<<<
{a}
>>>

Report B:
<<<
{b}
>>>

Which report better answers the question? Judge comprehensiveness, depth of analysis,
factual support and relevance. Ignore length and formatting. Answer "tie" only if they
are genuinely equivalent."""


class PairwiseJudge:
    """Head-to-head judgments within a GRPO group (both orders, memoized).

    Absolute rubric scores are nearly flat across sub-trees (and noisy), so the
    group reward is each sub-tree's win-rate against the other group members.
    """

    def __init__(self, memo_dir: Path = DATA / "pairs", model: str | None = None, concurrency: int = 6):
        self.memo_dir = memo_dir
        memo_dir.mkdir(parents=True, exist_ok=True)
        self.model = model or os.environ.get("REPLAY_JUDGE_MODEL", "gpt-5-mini")
        self.concurrency = concurrency

    def _memo(self, qid: str) -> tuple[Path, dict[str, float]]:
        p = self.memo_dir / f"{qid}.json"
        return p, (json.loads(p.read_text()) if p.exists() else {})

    async def _one(self, client, question: str, a: str, b: str) -> float | None:
        from typing import Literal

        from pydantic import BaseModel

        class Verdict(BaseModel):
            reason: str  # reason first, then decide
            winner: Literal["A", "B", "tie"]

        for attempt in range(4):
            try:
                r = await client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[{"role": "user", "content": PAIR_PROMPT.format(question=question, a=a[:30000], b=b[:30000])}],
                    response_format=Verdict, temperature=0,
                )
                w = (r.choices[0].message.parsed.winner or "").strip().lower().replace("report", "").strip()
                if w not in {"a", "b", "tie"}:
                    raise ValueError(f"unparseable verdict {w!r}")
                return 1.0 if w == "a" else 0.0 if w == "b" else 0.5
            except Exception:
                await asyncio.sleep(3 * (attempt + 1))
        return None

    def win_rates(self, qid: str, question: str, reports: dict[str, str]) -> dict[str, float]:
        """reports: {subtree_key: text}. Returns {key: mean win-rate vs the others}."""
        from openai import AsyncOpenAI

        keys = [k for k, t in reports.items() if t.strip()]
        path, memo = self._memo(qid)
        need = [(x, y) for i, x in enumerate(keys) for y in keys[i + 1:] if f"{x}|{y}" not in memo]

        async def run():
            client = AsyncOpenAI()
            sem = asyncio.Semaphore(self.concurrency)

            async def pair(x, y):
                async with sem:
                    ab = await self._one(client, question, reports[x], reports[y])
                    ba = await self._one(client, question, reports[y], reports[x])
                if ab is None or ba is None:
                    return x, y, None
                return x, y, (ab + (1.0 - ba)) / 2.0  # P(x beats y), position-debiased

            return await asyncio.gather(*(pair(x, y) for x, y in need))

        if need:
            for x, y, p in asyncio.run(run()):
                if p is not None:
                    memo[f"{x}|{y}"] = p
            path.write_text(json.dumps(memo))
        out = {}
        for x in keys:
            vals = []
            for y in keys:
                if x == y:
                    continue
                if f"{x}|{y}" in memo:
                    vals.append(memo[f"{x}|{y}"])
                elif f"{y}|{x}" in memo:
                    vals.append(1.0 - memo[f"{y}|{x}"])
            out[x] = sum(vals) / len(vals) if vals else 0.5
        return out


SPLITS = Path("data/benchmarks/deep_research_gym/splits.json")


def load_split(split: str) -> tuple[dict[str, ResearchTree], dict[str, dict]]:
    """Cached trees (this backbone) + their embeddings for a DRG split."""
    ids = json.loads(SPLITS.read_text())[split]
    trees = {q: ResearchTree.from_trajectory(p) for q, p in find_trees(ids).items()}
    emb = Embedder()
    return trees, {q: emb.for_tree(q, t) for q, t in trees.items()}
