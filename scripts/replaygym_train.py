"""GRPO on ReplayGym: learn when researching deeper is worth it.

    set -a; source .env; set +a
    python scripts/replaygym_train.py train --iters 25 --batch 6 --group 8 --lam 2
    python scripts/replaygym_train.py eval  --split val

Stage A policy: small MLP over the 10 decision features (laptop).
Stage B (A100): same env and loss with a Qwen3 LoRA policy over the branch text.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

from adr.replay.rl import (
    DATA, N_FEAT, EpisodeState, PairwiseJudge, RewardService, fixed_policy, rollout,
)
from adr.replay.rl import load_split as load

OUT = DATA


class Policy(torch.nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(N_FEAT, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
        with torch.no_grad():  # start near "always go deeper" = today's agent
            self.net[2].bias.fill_(2.0)

    def p(self, f: np.ndarray) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.as_tensor(f, dtype=torch.float32)).squeeze(-1))


def train(a) -> None:
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    trees, embs = load("train")
    print(f"train trees available: {len(trees)} {sorted(trees)}", flush=True)
    svc = RewardService(lam=a.lam)
    pj = PairwiseJudge() if a.reward == "pairwise" else None
    pol = Policy()
    if a.resume and (OUT / "policy.pt").exists():
        pol.load_state_dict(torch.load(OUT / "policy.pt"))
        print("resumed policy from", OUT / "policy.pt", flush=True)
    opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
    log = []
    OUT.mkdir(parents=True, exist_ok=True)
    for it in range(a.iters):
        t0 = time.time()
        qids = random.sample(sorted(trees), min(a.batch, len(trees)))
        loss_terms, stats = [], []
        # 1) sample every group with the current policy
        groups = {}
        for qid in qids:
            eps = []
            for _ in range(a.group):
                st = EpisodeState(qid, trees[qid], embs[qid])
                with torch.no_grad():
                    trace = rollout(st, lambda f: int(torch.rand(()) < pol.p(f)))
                eps.append((st.subtree(), trace))
            groups[qid] = eps

        # 2) score all questions in parallel (independent memos / judge folders)
        def score_group(qid):
            eps = groups[qid]
            scs = svc.scores(qid, trees[qid], [s for s, _ in eps])
            if pj is not None:
                # reward = head-to-head win-rate within the group (x100) - cost
                keyed = {svc.key(s): svc.report_text(qid, s) for s, _ in eps}
                wr = pj.win_rates(qid, trees[qid].query, keyed)
                rs = [None if sc is None else 100.0 * (wr.get(svc.key(s), 0.5) if s else 0.0) - a.lam * sc["tokens"] / 1e5
                      for (s, _), sc in zip(eps, scs)]
            else:
                rs = [svc.reward(sc) for sc in scs]
            return qid, scs, rs

        from concurrent.futures import ThreadPoolExecutor

        import os

        with ThreadPoolExecutor(max_workers=int(os.environ.get("REPLAY_Q_WORKERS", "3"))) as ex:
            scored = list(ex.map(score_group, qids))

        # 3) GRPO update
        for qid, scs, rs in scored:
            eps = groups[qid]
            keep = [(tr, r, sc) for (_, tr), r, sc in zip(eps, rs, scs) if r is not None]
            if len(keep) < 2:
                continue
            r = np.array([k[1] for k in keep], np.float32)
            adv = (r - r.mean()) / (r.std() + 1e-6)          # GRPO group-normalized advantage
            for (tr, _, sc), A in zip(keep, adv):
                if not tr:
                    continue
                f = torch.as_tensor(np.stack([d.feat for d in tr]))
                act = torch.as_tensor([d.action for d in tr], dtype=torch.float32)
                p = pol.p(f).clamp(1e-4, 1 - 1e-4)
                logp = (act * p.log() + (1 - act) * (1 - p).log()).sum()
                ent = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
                loss_terms.append(-float(A) * logp - a.ent * ent)
                stats.append((sc["quality"], sc["kpr"], sc["tokens"] / trees[qid].tokens(trees[qid].all_nodes()), float(r.mean())))
        if loss_terms:
            opt.zero_grad(); torch.stack(loss_terms).mean().backward(); opt.step()
        s = np.array(stats) if stats else np.zeros((1, 4))
        row = {"iter": it, "quality": float(s[:, 0].mean()), "kpr": float(s[:, 1].mean()),
               "token_frac": float(s[:, 2].mean()), "reward": float(s[:, 3].mean()),
               "memo": sum(len(svc._load(q)) for q in trees), "sec": round(time.time() - t0)}
        log.append(row)
        print(json.dumps(row), flush=True)
        torch.save(pol.state_dict(), OUT / "policy.pt")
        (OUT / "train_log.json").write_text(json.dumps(log, indent=1))


def evaluate(a) -> None:
    """Held-out comparison on one split: absolute rubric scores, plus a
    head-to-head tournament between the methods' reports for each question
    (both orders, Qwen3-32B), which is far less noisy than absolute scores."""
    from concurrent.futures import ThreadPoolExecutor
    import os

    trees, embs = load(a.split)
    svc = RewardService(lam=a.lam)
    pj = PairwiseJudge()
    pol = Policy()
    pol.load_state_dict(torch.load(a.ckpt or OUT / "policy.pt"))
    methods = {
        "full tree": fixed_policy("full"),
        "depth 1": fixed_policy("d1"),
        "depth 2": fixed_policy("d2"),
        "learned (GRPO)": lambda st: (lambda f: int(pol.p(f) > 0.5)),
    }
    plans = {}  # qid -> {method: (subtree, expand-rate)}
    for qid, t in trees.items():
        plans[qid] = {}
        for name, make in methods.items():
            st = EpisodeState(qid, t, embs[qid])
            with torch.no_grad():
                tr = rollout(st, make(st))
            plans[qid][name] = (st.subtree(), float(np.mean([d.action for d in tr])) if tr else 0.0)

    def score(qid):
        t = trees[qid]
        subs = [plans[qid][m][0] for m in methods]
        scs = svc.scores(qid, t, subs)
        uniq = {svc.key(s): svc.report_text(qid, s) for s in set(subs) if s}
        if len(uniq) > 1:
            pj.win_rates(qid, t.query, uniq)
        memo = pj._memo(qid)[1]
        # a method's head-to-head score = mean P(win) against the other methods' plans
        out = {}
        for i, (m, s) in enumerate(zip(methods, subs)):
            k, vs = svc.key(s), []
            for j, s2 in enumerate(subs):
                if j == i:
                    continue
                k2 = svc.key(s2)
                vs.append(0.5 if k2 == k else memo[f"{k}|{k2}"] if f"{k}|{k2}" in memo else 1 - memo.get(f"{k2}|{k}", 0.5))
            out[m] = (scs[i], float(np.mean(vs)))
        return qid, out

    with ThreadPoolExecutor(max_workers=int(os.environ.get("REPLAY_Q_WORKERS", "3"))) as ex:
        res = dict(ex.map(score, sorted(trees)))

    rows = {}
    for m in methods:
        v, per = [], {}
        for qid, t in trees.items():
            sc, h2h = res[qid][m]
            if sc:
                v.append((sc["quality"], sc["kpr"], sc["tokens"] / t.tokens(t.all_nodes()), sc["tokens"], 100 * h2h, plans[qid][m][1]))
                per[qid] = {"quality": sc["quality"], "kpr": sc["kpr"], "h2h": 100 * h2h, "tokens": sc["tokens"]}
        v = np.array(v)
        rows[m] = {"n": len(v), "quality": v[:, 0].mean(), "kpr": v[:, 1].mean(), "token_frac": v[:, 2].mean(),
                   "tokens": v[:, 3].mean(), "h2h_winrate": v[:, 4].mean(), "expand_rate": v[:, 5].mean(), "per_question": per}
    same = sum(plans[q]["learned (GRPO)"][0] == plans[q]["depth 1"][0] for q in trees)
    print(f"\n{a.split} ({len(trees)} questions)  {'quality':>8} {'KPR':>6} {'tokens':>7} {'h2h win%':>9} {'expand':>7}")
    for k, r in rows.items():
        print(f"{k:16s} {r['quality']:8.1f} {r['kpr']:6.1f} {100 * r['token_frac']:6.0f}% {r['h2h_winrate']:9.1f} {r['expand_rate']:7.2f}")
    print(f"learned plan == depth-1 plan on {same}/{len(trees)} questions")
    rows["_learned_equals_d1"] = same
    (OUT / f"eval_{a.split}.json").write_text(json.dumps(rows, indent=1, default=float))


def export(a) -> None:
    """Trained MLP -> JSON the fork's GR_ORCHESTRATOR=replaygym policy loads."""
    pol = Policy()
    pol.load_state_dict(torch.load(OUT / "policy.pt"))
    lin1, lin2 = pol.net[0], pol.net[2]
    w = {"w1": lin1.weight.tolist(), "b1": lin1.bias.tolist(), "w2": lin2.weight.tolist(), "b2": lin2.bias.tolist()}
    path = OUT / "replaygym_policy.json"
    path.write_text(json.dumps(w))
    print(f"exported {path}  (set GR_REPLAYGYM_POLICY={path.resolve()})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "export"])
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--split", default="val")
    ap.add_argument("--reward", choices=["pointwise", "pairwise"], default="pairwise")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt", default=None, help="policy .pt for eval (default: latest)")
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "export": export}[a.mode](a)
