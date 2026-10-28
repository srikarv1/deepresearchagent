"""Exact GRPO (Dr. GRPO over the full plan space) for ReplayGym expansion policies.

    set -a; source .env; source scripts/replaygym_env.sh; set +a
    # 1) reward table: every reachable plan x replicates (agent writer + Gym judge)
    python scripts/replaygym_grpo.py table --split train --reps 2 --dry-run
    python scripts/replaygym_grpo.py table --split train --reps 2
    python scripts/replaygym_grpo.py table --split val --reps 2
    # 2) gate: noise, signal and cross-fitted headroom over the fixed-depth rules
    python scripts/replaygym_grpo.py stats --split train --lam 0.02
    # 3) train on train, select on val; evaluate; export the MLP for the fork
    python scripts/replaygym_grpo.py train --policy mlp --metric kpr --lam 0.02
    python scripts/replaygym_grpo.py eval --run <run_dir> --split test
    python scripts/replaygym_grpo.py export --run <run_dir>

--pack defaults to the live GR_FINAL_PACK; --n-full must be full_tree_size(breadth,
depth) of the agent config (28 for breadth 4, depth 3).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from adr.replay.grpo import LMBackend, MLPBackend, Objective, Question, TrainConfig, evaluate, train
from adr.replay.plans import enumerate_plans
from adr.replay.rl import DATA, LOCAL_EMBED_MODEL, embed_model_name, load_split
from adr.replay.table import (
    IDENTITY,
    METRICS,
    TABLE_VERSION,
    IncompleteTable,
    RewardTable,
    TableMismatch,
    fill_table,
    gate_stats,
    missing_work,
)
from adr.replay.tree import MAX_CONTEXT_WORDS, PACKS, WRITER_CONTEXT_RULE

DEFAULT_LR = {"mlp": 1e-2, "lm": 2e-5}
DEFAULT_BATCH = {"mlp": 0, "lm": 8}  # MLP: exact full-batch gradient; LM: 8 questions per step


def _qids(a) -> set[str]:
    return {q.strip() for q in (getattr(a, "qids", "") or "").split(",") if q.strip()}


def _resolve_pack(a) -> None:
    """Default to the live agent's GR_FINAL_PACK (its YAML env, then the shell)."""
    if a.pack is None:
        from adr.replay.synth import apply_agent_env

        apply_agent_env()
        a.pack = os.environ.get("GR_FINAL_PACK", "tail").strip().lower() or "tail"
    if a.pack not in PACKS:
        raise SystemExit(f"GR_FINAL_PACK={a.pack!r} cannot be replayed; ReplayGym supports {PACKS}")


def _trees(a, split: str):
    trees, embs = load_split(split)
    want = _qids(a)
    if want:
        trees = {q: t for q, t in trees.items() if q in want}
    if not trees:
        raise SystemExit(f"no cached trees for split {split!r} (runs need GR_NODE_CACHE=1)")
    return trees, embs


def load_questions(a, split: str, *, with_prompts: bool = False, min_reps: int = 1) -> list[Question]:
    trees, embs = _trees(a, split)
    prompt_fn = None
    if with_prompts:
        from adr.replay.lm_policy import decision_prompt

        prompt_fn = decision_prompt
    want = {"version": TABLE_VERSION, "context_rule": WRITER_CONTEXT_RULE, "pack": a.pack, "max_words": a.max_words}
    out: list[Question] = []
    skipped: Counter[str] = Counter()
    scorers: set[str] = set()
    for qid in sorted(trees):
        tab = RewardTable.load(a.table_dir, qid)
        if tab is None:
            skipped["no_table"] += 1
            continue
        tab.check(want, IDENTITY)
        scorers.add(json.dumps({"writer": tab.header["writer"], "judge": tab.header["judge_model"]}, sort_keys=True))
        space = enumerate_plans(qid, trees[qid], embs[qid], n_full=a.n_full, prompt_fn=prompt_fn)
        try:
            scores = tab.scores(space, min_reps)
        except IncompleteTable as exc:
            print(f"skip: {exc}")
            skipped["incomplete"] += 1
            continue
        out.append(Question(qid, space, scores))
    if len(scorers) > 1:
        raise TableMismatch(f"{split}: tables come from different writers or judges: {sorted(scorers)}")
    print(f"{split}: {len(out)} questions loaded, skipped {dict(skipped)}", flush=True)
    if not out:
        raise SystemExit(f"no complete reward tables for split {split!r}; run the table command first")
    return out


# ------------------------------------------------------------------------ commands
def cmd_table(a) -> None:
    from adr.replay.judge import judge_model
    from adr.replay.synth import apply_agent_env

    apply_agent_env()  # the live writer models, recorded in each table header
    _resolve_pack(a)
    trees, embs = _trees(a, a.split)
    jm = judge_model()

    def one(qid: str):
        space = enumerate_plans(qid, trees[qid], embs[qid], n_full=a.n_full)
        if a.dry_run:
            return qid, missing_work(trees[qid], space, root=a.table_dir, reps=a.reps, pack=a.pack, max_words=a.max_words)
        return qid, fill_table(
            trees[qid], space, root=a.table_dir, reps=a.reps, judge_model=jm, pack=a.pack,
            max_words=a.max_words, concurrency=a.concurrency,
        )

    total: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=a.q_workers) as ex:
        for qid, res in ex.map(one, sorted(trees)):
            print(json.dumps({"qid": qid, **res}), flush=True)
            total.update(res)
    if a.dry_run:
        # ~1.33 tokens per English word: writer input only; judges add their own.
        print(json.dumps({"questions": len(trees), **total, "writer_input_tokens_est": int(1.33 * total["context_words"])}))
    else:
        print(json.dumps({"questions": len(trees), **total}))


def cmd_stats(a) -> None:
    _resolve_pack(a)
    qs = load_questions(a, a.split, min_reps=2)
    for metric in a.metrics:
        s = gate_stats(qs, metric=metric, lam=a.lam, n_boot=a.boot, seed=a.seed)
        lo, _ = s["headroom_ci_points"]
        s["verdict"] = (
            "PASS: a per-question policy can beat the best fixed rule"
            if lo > 0 else "FAIL: no headroom beyond noise; fix the writer/reward before training"
        )
        print(json.dumps(s, indent=1), flush=True)


def _objective(a) -> Objective:
    return Objective(metric=a.metric, lam=a.lam, beta=a.beta, group=a.group, advantage=a.advantage)


def _backend(a, init: Path | None):
    if a.policy == "mlp":
        return MLPBackend(hidden=a.hidden, seed=a.seed, init=init)
    return LMBackend(model_name=a.model, lora_r=a.lora_r, init=init, chunk=a.chunk)


def cmd_train(a) -> None:
    if a.policy == "mlp" and embed_model_name() != LOCAL_EMBED_MODEL:
        raise SystemExit(
            "the live ExpansionPolicy embeds with bge-small: set REPLAY_EMBED=local so the MLP "
            "is trained on the features it will be served"
        )
    _resolve_pack(a)
    obj = _objective(a)
    a.lr = a.lr if a.lr is not None else DEFAULT_LR[a.policy]
    a.batch = a.batch if a.batch is not None else DEFAULT_BATCH[a.policy]
    cfg = TrainConfig(
        steps=a.steps, lr=a.lr, batch=a.batch, eval_every=a.eval_every, patience=a.patience, seed=a.seed,
    )
    with_prompts = a.policy == "lm"
    train_qs = load_questions(a, a.train_split, with_prompts=with_prompts)
    val_qs = load_questions(a, a.val_split, with_prompts=with_prompts)
    name = a.name or f"{a.policy}-{a.metric}-lam{a.lam:g}-beta{a.beta:g}-g{a.group}-s{a.seed}"
    run_dir = Path(a.out) if a.out else DATA / "grpo" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "func"}
    config.update({"embed_model": embed_model_name(), "replay_tag": DATA.name})
    (run_dir / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    backend = _backend(a, Path(a.init) if a.init else None)
    summary = train(backend, train_qs, val_qs, obj, cfg, run_dir)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=1))


def _run_args(run: Path, **overrides) -> SimpleNamespace:
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    cfg["table_dir"] = Path(cfg["table_dir"])
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def cmd_eval(a) -> None:
    run = Path(a.run)
    r = _run_args(run, qids=a.qids)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    backend = _backend(r, run / (MLPBackend.ckpt_name if r.policy == "mlp" else LMBackend.ckpt_name))
    qs = load_questions(r, a.split, with_prompts=r.policy == "lm")
    ev = evaluate(backend, qs, _objective(r), fixed_rule=summary["fixed_rule"], n_boot=a.boot, seed=r.seed)
    (run / f"eval_{a.split}.json").write_text(json.dumps(ev, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in ev.items() if k != "per_question"}, indent=1))


def cmd_export(a) -> None:
    run = Path(a.run)
    r = _run_args(run)
    if r.policy != "mlp":
        raise SystemExit("only the MLP is served by the fork's ExpansionPolicy")
    backend = MLPBackend(hidden=r.hidden, init=run / MLPBackend.ckpt_name)
    meta = {k: getattr(r, k) for k in ("embed_model", "n_full", "pack", "max_words", "metric", "lam", "beta", "replay_tag")}
    out = Path(a.out) if a.out else run / "replaygym_policy.json"
    out.write_text(json.dumps(backend.export({**meta, "run": str(run)})), encoding="utf-8")
    print(f"export GR_REPLAYGYM_POLICY={out.resolve()} GR_FINAL_PACK={r.pack} GR_BRANCH_INDEPENDENT=1")


# --------------------------------------------------------------------------- args
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def space_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--qids", default="", help="comma-separated subset of query ids")
        p.add_argument("--n-full", type=int, default=28, help="full_tree_size(breadth, depth) of the live agent")
        p.add_argument("--pack", choices=PACKS, default=None, help="writer packing; default: the live GR_FINAL_PACK")
        p.add_argument("--max-words", type=int, default=MAX_CONTEXT_WORDS)
        p.add_argument("--table-dir", type=Path, default=DATA / "table")
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("table", help="score every plan of a split")
    space_args(p)
    p.add_argument("--split", default="train")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("REPLAY_SYNTH_CONC", "2")))
    p.add_argument("--q-workers", type=int, default=int(os.environ.get("REPLAY_Q_WORKERS", "3")))
    p.add_argument("--dry-run", action="store_true", help="count the reports and context words still needed")
    p.set_defaults(func=cmd_table)

    p = sub.add_parser("stats", help="noise, signal and headroom gate")
    space_args(p)
    p.add_argument("--split", default="train")
    p.add_argument("--metrics", nargs="+", choices=sorted(METRICS), default=["kpr", "quality"])
    p.add_argument("--lam", type=float, default=0.02)
    p.add_argument("--boot", type=int, default=2000)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("train", help="exact or sampled GRPO")
    space_args(p)
    p.add_argument("--policy", choices=["mlp", "lm"], default="mlp")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--metric", choices=sorted(METRICS), default="kpr")
    p.add_argument("--lam", type=float, default=0.02)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--group", type=int, default=0, help="0 = exact expectation; G >= 2 = sampled GRPO")
    p.add_argument("--advantage", choices=["rloo", "std"], default="rloo")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=None, help="default 1e-2 (mlp), 2e-5 (lm)")
    p.add_argument("--batch", type=int, default=None, help="questions per step, 0 = all; default 0 (mlp), 8 (lm)")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--chunk", type=int, default=8, help="LM prompts per forward pass")
    p.add_argument("--init", default=None, help="MLP .pt or LoRA adapter dir to start from (and use as reference)")
    p.add_argument("--name", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="evaluate a trained run on a split")
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--qids", default="")
    p.add_argument("--boot", type=int, default=2000)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("export", help="MLP weights for GR_REPLAYGYM_POLICY")
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_export)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
