"""Stage B (A100): Qwen3 LoRA expansion policy on ReplayGym.

    # 1) DPO warm start on exact counterfactual pairs from the pairwise memo
    python scripts/replaygym_lm_train.py dpo  --model Qwen/Qwen3-1.7B --epochs 2
    # 2) GRPO in the replay env (pairwise tournament reward), from the DPO adapter
    python scripts/replaygym_lm_train.py grpo --model Qwen/Qwen3-1.7B --init runs/replaygym/bedrock/lm_dpo --iters 30
    # 3) Evaluate on val (same table as Stage A)
    python scripts/replaygym_lm_train.py eval --init runs/replaygym/bedrock/lm_grpo
Needs the cached trees + memo from Stage A (copy runs/ to the GPU box) and the
Bedrock env for new rewards: set -a; source .env; source scripts/replaygym_env.sh; set +a
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch

from adr.replay.lm_policy import LMPolicy, actions_for_subtree, decision_prompt
from adr.replay.rl import DATA, EpisodeState, PairwiseJudge, RewardService, fixed_policy, load_split as load, rollout


def _key_to_subtree(tree) -> dict[str, frozenset]:
    """All 256 realizable sub-trees of the branch-depth action space, by memo key."""
    from itertools import product

    out = {}
    roots = tree.roots
    for depths in product(range(1, 4), repeat=len(roots)):  # roots always researched
        s = set()
        for r, d in zip(roots, depths):
            s |= {n for n in tree.subtree(r) if tree.nodes[n].tree_depth - tree.nodes[r].tree_depth < d}
        s = frozenset(s)
        out[RewardService.key(s)] = s
    return out


def dpo(a) -> None:
    trees, embs = load("train")
    pairs_dir = DATA / "pairs"
    data = []
    for qid, t in trees.items():
        p = pairs_dir / f"{qid}.json"
        if not p.exists():
            continue
        k2s = _key_to_subtree(t)
        for kxy, pxy in json.loads(p.read_text()).items():
            x, y = kxy.split("|")
            if x not in k2s or y not in k2s or abs(pxy - 0.5) < a.margin:
                continue
            w, l = (x, y) if pxy > 0.5 else (y, x)
            data.append((qid, k2s[w], k2s[l]))
    print(f"DPO pairs: {len(data)} from {len(trees)} questions", flush=True)
    pol = LMPolicy(a.model, lora_r=a.lora_r)
    opt = torch.optim.AdamW([p for p in pol.parameters() if p.requires_grad], lr=a.lr)
    from adr.replay.lm_policy import dpo_loss

    for ep in range(a.epochs):
        random.shuffle(data)
        tot = 0.0
        for qid, w, l in data:
            win = actions_for_subtree(EpisodeState(qid, trees[qid], embs[qid]), w)
            lose = actions_for_subtree(EpisodeState(qid, trees[qid], embs[qid]), l)
            loss = dpo_loss(pol, win, lose, beta=a.beta)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        print(json.dumps({"epoch": ep, "dpo_loss": tot / max(1, len(data))}), flush=True)
    pol.model.save_pretrained(DATA / "lm_dpo")


def grpo(a) -> None:
    from peft import PeftModel

    trees, embs = load("train")
    svc, pj = RewardService(lam=a.lam), PairwiseJudge()
    pol = LMPolicy(a.model, lora_r=a.lora_r)
    if a.init:
        pol.model = PeftModel.from_pretrained(pol.model.base_model.model, a.init, is_trainable=True)
    opt = torch.optim.AdamW([p for p in pol.parameters() if p.requires_grad], lr=a.lr)
    for it in range(a.iters):
        qids = random.sample(sorted(trees), min(a.batch, len(trees)))
        terms, rew = [], []
        for qid in qids:
            eps = []
            for _ in range(a.group):
                st = EpisodeState(qid, trees[qid], embs[qid])
                trace = []
                while not st.done:
                    pr = decision_prompt(st)
                    act = int(random.random() < pol.p_yes(pr))
                    trace.append((pr, act)); st.step(act)
                eps.append((st.subtree(), trace))
            scs = svc.scores(qid, trees[qid], [s for s, _ in eps])
            wr = pj.win_rates(qid, trees[qid].query, {svc.key(s): svc.report_text(qid, s) for s, _ in eps})
            r = np.array([100.0 * (wr.get(svc.key(s), 0.5) if s else 0.0) - a.lam * (sc or {}).get("tokens", 0) / 1e5
                          for (s, _), sc in zip(eps, scs)], np.float32)
            adv = (r - r.mean()) / (r.std() + 1e-6)
            for (_, tr), A in zip(eps, adv):
                if tr:
                    p, act = zip(*tr)
                    terms.append(-float(A) * pol.logp(list(p), list(act)).sum())
            rew.append(float(r.mean()))
        opt.zero_grad(); torch.stack(terms).mean().backward(); opt.step()
        print(json.dumps({"iter": it, "reward": float(np.mean(rew))}), flush=True)
        pol.model.save_pretrained(DATA / "lm_grpo")


def evaluate(a) -> None:
    from peft import PeftModel

    trees, embs = load(a.split)
    svc, pj = RewardService(lam=a.lam), PairwiseJudge()
    pol = LMPolicy(a.model, lora_r=a.lora_r)
    pol.model = PeftModel.from_pretrained(pol.model.base_model.model, a.init)
    rows = {}
    for qid, t in trees.items():
        subs = {}
        for name, kind in (("full", "full"), ("d1", "d1"), ("d2", "d2")):
            st = EpisodeState(qid, t, embs[qid]); rollout(st, fixed_policy(kind)(st)); subs[name] = st.subtree()
        st = EpisodeState(qid, t, embs[qid])
        while not st.done:
            st.step(int(pol.p_yes(decision_prompt(st)) > 0.5))
        subs["lm"] = st.subtree()
        scs = dict(zip(subs, svc.scores(qid, t, list(subs.values()))))
        wr = pj.win_rates(qid, t.query, {svc.key(s): svc.report_text(qid, s) for s in subs.values()})
        full_tok = t.tokens(t.all_nodes())
        for name, s in subs.items():
            rows.setdefault(name, []).append((wr.get(svc.key(s), 0.5), (scs[name] or {}).get("kpr"), t.tokens(s) / full_tok))
    for name, v in rows.items():
        v = np.array([x for x in v if x[1] is not None], float)
        print(f"{name:6s} win-rate {v[:, 0].mean():.2f}  KPR {v[:, 1].mean():5.1f}  tokens {100 * v[:, 2].mean():4.0f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dpo", "grpo", "eval"])
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--init", default=None)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    {"dpo": dpo, "grpo": grpo, "eval": evaluate}[a.mode](a)
