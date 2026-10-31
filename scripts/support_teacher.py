#!/usr/bin/env python3
"""Support-preserving retention teacher for PILOT (local BC labels + same-state DPO pairs).

Outcome-selected BC copies whatever keep mask happened to occur in a run whose
final report scored well. This teacher instead labels each saved decision state
by what its evidence *supports*:

  1. label    One verifier call per unique page: which of the query's Gym key
              points does this page state or directly support? Results are
              cached in data/teacher/support_cache.jsonl keyed by (query, page
              content hash), so retained pages that reappear in later rounds or
              other seeds are labeled once. Pages behind the v2q BC states go
              first, so BC data is ready before the rest of the corpus is done.
  2. build    Per state, a support matrix S[page][key point]. The teacher mask is
              a token-weighted greedy set cover keeping up to ``--redundancy``
              supporting pages per key point the pool supports (pages that
              support nothing are dropped). Any other mask is scored from the
              same matrix with no further calls, which yields:
                - data/bc/gym_teacher/oai_{train,dev}.jsonl: the v2q states with only the KEEP line
                  replaced by the teacher mask (ALLOC and DECISION unchanged);
                - data/bc/gym_teacher/dpo_{train,dev}.jsonl: same-state pairs, chosen = teacher mask,
                  rejected = the historical / random / top-relevance mask when
                  the teacher keeps more key points at similar cost or the same
                  key points at clearly lower cost;
                - pilot.json: support retained and tokens per mask type.

Key points are the benchmark's per-query key points: privileged training-time
supervision, used only on training queries; the policy never sees them.

    python scripts/support_teacher.py label --concurrency 16 --scope v2q
    python scripts/support_teacher.py build
    python scripts/support_teacher.py onpolicy --base-url <tunnel>/v1 --model pilot-bc-teacher

``onpolicy`` builds DPO v2 pairs from the served BC policy's own samples: for
each labeled v2q state it samples ``--n`` completions, puts each sampled KEEP
into the state's own ALLOC / DECISION lines (so a pair differs only in KEEP),
scores every mask from the support matrix, and pairs the best against the
worst when they differ by at least one supported key point at similar cost or
by >= 20% tokens at equal support. Writes data/bc/gym_teacher/dpo_v2_{train,dev}.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

CORPUS = Path("runs/gym_corpus_v1/rollouts_gym")
STATES = Path("data/teacher/gym_all_states.jsonl")  # adr bc-pairs over every rollout
V2Q = Path("data/bc/gym_v2q/bc_pairs.jsonl")
KEY_POINTS = Path.home() / "deepresearchgym" / "key_point"
OUT = Path("data/teacher")
CACHE = OUT / "support_cache.jsonl"
COLAB = Path("data/bc/gym_teacher")  # BC + DPO files for the notebook
DEV_QUERIES = {"1004217", "969384"}  # same held-out queries as Gym v1 / v2q
PAGE_WORDS = 1000  # verifier sees at most this many words of a page (Azure TPM bound)
MODEL = "gpt-5-mini"

PROMPT = """You check whether a web page supports key points of a research question.

Question: {question}

Key points:
{points}

Page:
\"\"\"
{page}
\"\"\"

A key point counts only if the page states it or gives direct evidence for it; a page that
merely mentions the topic does not count. Return JSON: {{"supported": [key point numbers]}}"""


def pool_ids(prompt: str) -> list[str]:
    block = prompt.split("Evidence (", 1)[1].split("\n\n", 1)[0]
    return [line.strip().split(" | ", 1)[0] for line in block.splitlines()[1:]]


def rho_of(prompt: str) -> dict[str, float]:
    block = prompt.split("Evidence (", 1)[1].split("\n\n", 1)[0]
    out = {}
    for line in block.splitlines()[1:]:
        parts = line.strip().split(" | ")
        out[parts[0]] = float(parts[1].split()[0])
    return out


def page_key(qid: str, text: str) -> str:
    return f"{qid}:{hashlib.md5(text.encode()).hexdigest()}"


def tokens(text: str) -> int:
    return max(1, len(text) // 4)  # same estimate as the fork's estimate_tokens


def load_key_points(qid: str) -> tuple[str, list[str]]:
    data = json.loads((KEY_POINTS / f"{qid}_aggregated.json").read_text())
    return data["question"], [p["point_content"] for p in data["key_points"]]


class Corpus:
    """States (one per serialized orchestration round) with their pages."""

    def __init__(self) -> None:
        self.rows = [json.loads(line) for line in STATES.open()]
        self.v2q = {(r["run_id"], r["round_id"]) for r in map(json.loads, V2Q.open())}
        self._ev: dict[str, dict] = {}

    def evidence(self, run_id: str, qid: str) -> dict:
        if run_id not in self._ev:
            raw = json.loads((CORPUS / run_id / "gpt_researcher" / f"{qid}.json").read_text())
            self._ev[run_id] = raw["evidence"]
        return self._ev[run_id]

    def pages(self, row: dict) -> dict[str, str]:
        ev = self.evidence(row["run_id"], row["query_id"])
        return {i: str((ev.get(i) or {}).get("content") or "") for i in pool_ids(row["prompt"])}


# --------------------------------------------------------------------------
# label
# --------------------------------------------------------------------------


async def label(concurrency: int, limit: int | None, scope: str) -> None:
    import openai

    corpus = Corpus()
    done = set()
    if CACHE.exists():
        done = {json.loads(line)["key"] for line in CACHE.open()}

    # v2q states first, then everything else.
    ordered = sorted(corpus.rows, key=lambda r: (r["run_id"], r["round_id"]) not in corpus.v2q)
    if scope == "v2q":
        ordered = [r for r in ordered if (r["run_id"], r["round_id"]) in corpus.v2q]
    todo: dict[str, tuple[str, str]] = {}
    for row in ordered:
        for text in corpus.pages(row).values():
            k = page_key(row["query_id"], text)
            if k not in done and k not in todo:
                todo[k] = (row["query_id"], text)
    items = list(todo.items())[: limit or None]
    print(f"{len(done)} cached, {len(items)} to label")

    client = openai.AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    kp_cache: dict[str, tuple[str, list[str]]] = {}
    lock = asyncio.Lock()
    fh = CACHE.open("a")
    n_done = 0

    async def one(key: str, qid: str, text: str) -> None:
        nonlocal n_done
        if qid not in kp_cache:
            kp_cache[qid] = load_key_points(qid)
        question, points = kp_cache[qid]
        page = " ".join(text.split()[:PAGE_WORDS])
        msg = PROMPT.format(
            question=question,
            points="\n".join(f"{i + 1}. {p}" for i, p in enumerate(points)),
            page=page,
        )
        supported: list[int] | None = None
        async with sem:
            for attempt in range(12):
                try:
                    r = await client.chat.completions.create(
                        model=MODEL,
                        reasoning_effort="minimal",
                        response_format={"type": "json_object"},
                        messages=[{"role": "user", "content": msg}],
                        max_completion_tokens=400,
                    )
                    got = json.loads(r.choices[0].message.content or "{}").get("supported") or []
                    supported = sorted({int(x) for x in got if str(x).isdigit() and 1 <= int(x) <= len(points)})
                    break
                except Exception as exc:  # rate limits, transient errors, bad JSON
                    await asyncio.sleep(min(30, 2 ** attempt) + random.random() * 3)
                    err = str(exc)[:120]
            if supported is None:
                print("failed:", key, err)
                return
        async with lock:
            fh.write(json.dumps({"key": key, "supported": supported}) + "\n")
            fh.flush()
            n_done += 1
            if n_done % 500 == 0:
                print(f"labeled {n_done}/{len(items)}", flush=True)

    await asyncio.gather(*(one(k, q, t) for k, (q, t) in items))
    fh.close()
    print(f"done: labeled {n_done}/{len(items)}")


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def teacher_mask(S: dict[str, set[int]], cost: dict[str, int], redundancy: int, rho: dict[str, float]) -> set[str]:
    need = defaultdict(int)
    for kps in S.values():
        for k in kps:
            need[k] = min(redundancy, need[k] + 1)
    if not need:  # pool supports no key point: keep the three most query-relevant pages
        return set(sorted(S, key=lambda i: -rho.get(i, 0.0))[:3])
    chosen: set[str] = set()
    while any(v > 0 for v in need.values()):
        best, best_val = None, 0.0
        for i, kps in S.items():
            if i in chosen:
                continue
            gain = sum(1 for k in kps if need[k] > 0)
            if gain and gain / cost[i] > best_val:
                best, best_val = i, gain / cost[i]
        if best is None:
            break
        chosen.add(best)
        for k in S[best]:
            if need[k] > 0:
                need[k] -= 1
    return chosen


def support(mask: set[str], S: dict[str, set[int]]) -> int:
    return len(set().union(*(S[i] for i in mask))) if mask else 0


def with_keep(completion: str, keep: set[str], pool: list[str]) -> str:
    kept = sorted(i for i in keep if i in set(pool))
    keep_txt = "ALL" if pool and len(kept) == len(pool) else " ".join(kept)
    return re.sub(r"^KEEP:.*$", f"KEEP: {keep_txt}", completion, count=1, flags=re.M)


def parse_keep(completion: str, pool: list[str]) -> set[str]:
    body = re.search(r"^KEEP:\s*(.*)$", completion, re.M).group(1).strip()
    return set(pool) if body.upper() == "ALL" else set(body.split()) & set(pool)


def build(redundancy: int, seed: int) -> None:
    rng = random.Random(seed)
    corpus = Corpus()
    cache = {d["key"]: set(d["supported"]) for d in map(json.loads, CACHE.open())}

    bc = {"train": [], "dev": []}
    dpo = {"train": [], "dev": []}
    pilot = defaultdict(lambda: defaultdict(list))
    skipped = 0
    for row in corpus.rows:
        pages = corpus.pages(row)
        pool = pool_ids(row["prompt"])
        keys = {i: page_key(row["query_id"], t) for i, t in pages.items()}
        if any(k not in cache for k in keys.values()):
            skipped += 1
            continue
        S = {i: cache[keys[i]] for i in pool}
        cost = {i: tokens(pages[i]) for i in pool}
        rho = rho_of(row["prompt"])
        split = "dev" if row["query_id"] in DEV_QUERIES else "train"
        full = support(set(pool), S)

        T = teacher_mask(S, cost, redundancy, rho)
        H = parse_keep(row["completion"], pool)
        R = set(rng.sample(pool, min(len(T), len(pool))))
        budget = sum(cost[i] for i in T)
        K, used = set(), 0
        for i in sorted(pool, key=lambda i: -rho.get(i, 0.0)):
            if used + cost[i] <= budget:
                K.add(i)
                used += cost[i]
        masks = {"teacher": T, "historical": H, "random_same_count": R, "topk_rho_same_tokens": K}

        is_v2q = (row["run_id"], row["round_id"]) in corpus.v2q
        for name, m in masks.items():
            pilot[name]["support_retained"].append(support(m, S) / full if full else 1.0)
            pilot[name]["tokens"].append(sum(cost[i] for i in m))
            if is_v2q and name == "historical":
                pilot["v2q_label"]["support_retained"].append(support(m, S) / full if full else 1.0)
                pilot["v2q_label"]["tokens"].append(sum(cost[i] for i in m))

        system, user = row["messages"][0], row["messages"][1]
        if is_v2q:
            bc[split].append({"messages": [system, user, {"role": "assistant", "content": with_keep(row["completion"], T, pool)}]})

        chosen = with_keep(row["completion"], T, pool)
        sT, cT = support(T, S), sum(cost[i] for i in T)
        for name in ("historical", "random_same_count", "topk_rho_same_tokens"):
            m = masks[name]
            if m == T:
                continue
            sM, cM = support(m, S), sum(cost[i] for i in m)
            better_support = sT > sM and cT <= 1.25 * cM
            cheaper_same = sT >= sM and cT <= 0.8 * cM
            if better_support or cheaper_same:
                dpo[split].append({
                    "prompt": [system, user],
                    "chosen": [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": with_keep(row["completion"], m, pool)}],
                    "meta": {"run_id": row["run_id"], "round_id": row["round_id"], "vs": name,
                             "support": [sT, sM, full], "tokens": [cT, cM]},
                })

    # The Colab notebook (notebooks/pilot_teacher_bc_dpo_gym.ipynb) reads these names.
    COLAB.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        for name, rows in (("oai", bc[split]), ("dpo", dpo[split])):
            path = COLAB / f"{name}_{split}.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    summary = {
        name: {"n_states": len(v["tokens"]),
               "support_retained": round(st.mean(v["support_retained"]), 3),
               "tokens": round(st.mean(v["tokens"]))}
        for name, v in pilot.items()
    }
    summary["counts"] = {f"{n}_{s}": len(d[s]) for n, d in (("bc", bc), ("dpo", dpo)) for s in ("train", "dev")}
    summary["states_skipped_unlabeled"] = skipped
    summary["redundancy"] = redundancy
    (OUT / "pilot.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def state_tables(corpus: Corpus, cache: dict[str, set[int]], row: dict):
    """Support matrix, token costs and pool for one state, or None if any page is unlabeled."""
    pages = corpus.pages(row)
    pool = pool_ids(row["prompt"])
    keys = {i: page_key(row["query_id"], t) for i, t in pages.items()}
    if any(k not in cache for k in keys.values()):
        return None
    return {i: cache[keys[i]] for i in pool}, {i: tokens(pages[i]) for i in pool}, pool


async def onpolicy(base_url: str, model: str, n: int, temperature: float, concurrency: int, redundancy: int) -> None:
    import openai

    corpus = Corpus()
    cache = {d["key"]: set(d["supported"]) for d in map(json.loads, CACHE.open())}
    rows = [r for r in corpus.rows if (r["run_id"], r["round_id"]) in corpus.v2q]
    client = openai.AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    sem = asyncio.Semaphore(concurrency)

    async def sample(row: dict) -> list[str]:
        async with sem:
            for attempt in range(5):
                try:
                    r = await client.chat.completions.create(
                        model=model, messages=row["messages"][:2], n=n,
                        temperature=temperature, max_tokens=700,
                    )
                    return [c.message.content or "" for c in r.choices]
                except Exception:
                    await asyncio.sleep(2 ** attempt)
        return []

    samples = await asyncio.gather(*(sample(r) for r in rows))
    out = {"train": [], "dev": []}
    stats = defaultdict(int)
    gaps = []
    for row, outs in zip(rows, samples):
        tables = state_tables(corpus, cache, row)
        if tables is None or not outs:
            stats["skipped"] += 1
            continue
        S, cost, pool = tables
        full = support(set(pool), S)
        cands: dict[frozenset, tuple[int, int]] = {}
        for text in outs:
            m = re.search(r"^\s*KEEP\s*:\s*(.*)$", text, re.M | re.I)
            if not m:
                stats["unparsable_samples"] += 1
                continue
            body = m.group(1).strip()
            keep = set(pool) if body.upper() == "ALL" else set(body.split()) & set(pool)
            if not keep:
                stats["empty_keep_samples"] += 1
                continue
            cands[frozenset(keep)] = (support(keep, S), sum(cost[i] for i in keep))
        teacher = frozenset(teacher_mask(S, cost, redundancy, rho_of(row["prompt"])))
        cands.setdefault(teacher, (support(teacher, S), sum(cost[i] for i in teacher)))
        if len(cands) < 2:
            stats["no_diverse_samples"] += 1
            continue
        ranked = sorted(cands.items(), key=lambda kv: (-kv[1][0], kv[1][1]))
        (best, (sb, cb)), (worst, (sw, cw)) = ranked[0], ranked[-1]
        better_support = sb > sw and cb <= 1.25 * cw
        cheaper_same = sb >= sw and cb <= 0.8 * cw
        if not (better_support or cheaper_same):
            stats["no_clear_preference"] += 1
            continue
        split = "dev" if row["query_id"] in DEV_QUERIES else "train"
        out[split].append({
            "prompt": row["messages"][:2],
            "chosen": [{"role": "assistant", "content": with_keep(row["completion"], set(best), pool)}],
            "rejected": [{"role": "assistant", "content": with_keep(row["completion"], set(worst), pool)}],
            "meta": {"run_id": row["run_id"], "round_id": row["round_id"], "n_distinct": len(cands),
                     "chosen_is_teacher": best == teacher, "support": [sb, sw, full], "tokens": [cb, cw]},
        })
        gaps.append((sb - sw) / full if full else 0.0)
    for split, rows_out in out.items():
        (COLAB / f"dpo_v2_{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_out))
    summary = {
        "pairs_train": len(out["train"]), "pairs_dev": len(out["dev"]), **stats,
        "mean_support_gap_frac": round(st.mean(gaps), 3) if gaps else None,
        "chosen_is_teacher": sum(r["meta"]["chosen_is_teacher"] for s_ in out.values() for r in s_),
    }
    (OUT / "onpolicy.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("label")
    a.add_argument("--concurrency", type=int, default=48)
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--scope", choices=("v2q", "all"), default="v2q", help="v2q = only the v2q BC states (enough for BC + DPO)")
    b = sub.add_parser("build")
    b.add_argument("--redundancy", type=int, default=2)
    b.add_argument("--seed", type=int, default=0)
    o = sub.add_parser("onpolicy")
    o.add_argument("--base-url", required=True)
    o.add_argument("--model", default="pilot-bc-teacher")
    o.add_argument("--n", type=int, default=8)
    o.add_argument("--temperature", type=float, default=1.0)
    o.add_argument("--concurrency", type=int, default=6)
    o.add_argument("--redundancy", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "label":
        asyncio.run(label(args.concurrency, args.limit, args.scope))
    elif args.cmd == "onpolicy":
        asyncio.run(onpolicy(args.base_url, args.model, args.n, args.temperature, args.concurrency, args.redundancy))
    else:
        build(args.redundancy, args.seed)


if __name__ == "__main__":
    main()
