"""GRPO for ReplayGym expansion policies, computed exactly over the plan space.

For a question q the policy induces a distribution over the reachable plans,
P_theta(p | q) = product over the plan's decisions of pi_theta(a | s). With a
reward table R_q(p) (``table.py``), GRPO with the group-mean baseline and no
std normalisation (Dr. GRPO) is an unbiased estimate of the gradient of

    J_q(theta) = sum_p P_theta(p | q) R_q(p),

a sum over at most 3**k plans. ``Objective(group=0)`` optimises it exactly,
minus beta * KL(P_theta || P_ref) to the frozen initial policy, also exact.
``group=G >= 2`` samples G plans per question instead (leave-one-out or the
original std-normalised advantages) and is kept for the estimator ablation.

Backends: the 10-feature MLP the fork's ExpansionPolicy loads, and the Qwen3
LoRA YES/NO head of ``lm_policy.py``. The LM gradient is exact too: one
no-grad pass gives the log-odds of every decision state, autograd gives
dLoss/dz on those, and a second pass in chunks backpropagates z . dLoss/dz.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from adr.replay.plans import FIXED_RULES, DecisionState
from adr.replay.rl import N_FEAT
from adr.replay.table import ScoredQuestion, bootstrap_ci

ADVANTAGES = ("rloo", "std")


# ----------------------------------------------------------------------- objective
@dataclass
class Objective:
    metric: str = "kpr"       # kpr | quality | mean
    lam: float = 0.02         # reward = metric in [0, 1] - lam * (research tokens / full tree)
    beta: float = 0.01        # exact KL to the initial policy
    group: int = 0            # 0 = exact expectation over every plan; G >= 2 = sampled GRPO
    advantage: str = "rloo"   # sampled only: rloo (unbiased) | std (original GRPO)

    def __post_init__(self) -> None:
        if self.group == 1 or self.group < 0:
            raise ValueError("group must be 0 (exact) or >= 2 (sampled)")
        if self.advantage not in ADVANTAGES:
            raise ValueError(f"advantage must be one of {ADVANTAGES}")
        if self.lam < 0 or self.beta < 0:
            raise ValueError("lam and beta must be >= 0")


@dataclass
class Question(ScoredQuestion):
    """A scored plan space plus the tensors the objective needs."""

    idx: torch.Tensor = field(init=False)
    act: torch.Tensor = field(init=False)
    mask: torch.Tensor = field(init=False)
    z_ref: torch.Tensor | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        idx, act, mask = self.space.path_arrays()
        self.idx, self.act, self.mask = torch.from_numpy(idx), torch.from_numpy(act), torch.from_numpy(mask)


def plan_logp(z: torch.Tensor, idx: torch.Tensor, act: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log P(plan) for every plan from the per-state log-odds z of expanding."""
    if z.numel() == 0:
        return torch.zeros(idx.shape[0], dtype=torch.float32)
    zz = z[idx]
    lp = act * F.logsigmoid(zz) + (1.0 - act) * F.logsigmoid(-zz)
    return (lp * mask).sum(dim=-1)


def question_loss(
    z: torch.Tensor, q: Question, obj: Objective, gen: torch.Generator | None = None
) -> tuple[torch.Tensor, dict[str, float]]:
    if q.z_ref is None:
        raise RuntimeError(f"{q.qid}: reference log-odds not set")
    logp = plan_logp(z, q.idx, q.act, q.mask)
    p = logp.exp()
    r = torch.as_tensor(q.reward(obj.metric, obj.lam), dtype=logp.dtype)
    kl = (p * (logp - plan_logp(q.z_ref.to(z.dtype), q.idx, q.act, q.mask))).sum()
    expected = (p * r).sum()
    if obj.group == 0:
        loss = -(expected - obj.beta * kl)
    else:
        draws = torch.multinomial(p.detach(), obj.group, replacement=True, generator=gen)
        rs = r[draws]
        if obj.advantage == "rloo":
            adv = rs - (rs.sum() - rs) / (obj.group - 1)
        else:
            adv = (rs - rs.mean()) / (rs.std(unbiased=False) + 1e-6)
        loss = -(adv.detach() * logp[draws]).mean() + obj.beta * kl
    with torch.no_grad():
        stats = {"expected_reward": float(expected), "kl": float(kl), "entropy": float(-(p * logp).sum())}
    return loss, stats


# ------------------------------------------------------------------------ backends
class Backend(Protocol):
    chunk: int | None   # None: one autograd pass; int: exact two-pass gradient in chunks

    def parameters(self) -> list[torch.nn.Parameter]: ...

    def log_odds(self, states: Sequence[DecisionState]) -> torch.Tensor: ...

    def save(self, run_dir: Path) -> None: ...


class MLPBackend:
    """Same layout as scripts/replaygym_train.py, so exports load in the fork."""

    chunk = None
    ckpt_name = "best.pt"

    def __init__(self, hidden: int = 32, seed: int = 0, init: Path | None = None):
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(N_FEAT, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
        if init is not None:
            self.load(init)
        else:
            with torch.no_grad():  # pi(expand) = 0.5 everywhere: the reference is the uniform policy
                self.net[2].weight.zero_()
                self.net[2].bias.zero_()

    def parameters(self) -> list[torch.nn.Parameter]:
        return list(self.net.parameters())

    def log_odds(self, states: Sequence[DecisionState]) -> torch.Tensor:
        if not states:
            return torch.zeros(0)
        f = torch.as_tensor(np.stack([s.features for s in states]), dtype=torch.float32)
        return self.net(f).squeeze(-1)

    def save(self, run_dir: Path) -> None:
        torch.save(self.net.state_dict(), Path(run_dir) / self.ckpt_name)

    def load(self, path: Path) -> None:
        state = torch.load(Path(path), map_location="cpu")
        # Also accepts scripts/replaygym_train.py checkpoints (keys "net.0.weight", ...).
        self.net.load_state_dict({k.removeprefix("net."): v for k, v in state.items()})

    def export(self, meta: dict[str, Any]) -> dict[str, Any]:
        """Weights in the fork's GR_REPLAYGYM_POLICY format (extra keys are ignored)."""
        lin1, lin2 = self.net[0], self.net[2]
        return {
            "w1": lin1.weight.tolist(),
            "b1": lin1.bias.tolist(),
            "w2": lin2.weight.tolist(),
            "b2": lin2.bias.tolist(),
            "meta": meta,
        }


class LMBackend:
    """Qwen3 LoRA YES/NO head of ``lm_policy.LMPolicy``; log-odds = logit(YES) - logit(NO)."""

    ckpt_name = "best"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B",
        lora_r: int = 16,
        init: Path | None = None,
        chunk: int = 8,
        max_length: int = 1536,
    ):
        from adr.replay.lm_policy import LMPolicy

        if chunk < 1:
            raise ValueError("chunk must be >= 1")
        self.pol = LMPolicy(model_name, lora_r=lora_r)
        if init is not None:
            self._load_adapter(Path(init), lora_r)
        # Logits are read at the trailing "Answer:", so truncate from the left.
        self.pol.tok.truncation_side = "left"
        # No LoRA dropout: both passes of the chunked gradient must evaluate the
        # same function (LoRA weights still train in eval mode).
        self.pol.model.eval()
        self.chunk = chunk
        self.max_length = max_length

    def _load_adapter(self, path: Path, lora_r: int) -> None:
        from peft import set_peft_model_state_dict
        from peft.utils import load_peft_weights

        cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        if int(cfg.get("r", -1)) != lora_r:
            raise ValueError(f"{path} is a rank-{cfg.get('r')} adapter; pass --lora-r {cfg.get('r')}")
        set_peft_model_state_dict(self.pol.model, load_peft_weights(str(path)))

    def parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.pol.model.parameters() if p.requires_grad]

    def _forward(self, prompts: list[str]) -> torch.Tensor:
        enc = self.pol.tok(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        ).to(self.pol.device)
        logits = self.pol.model(**enc).logits[:, -1, :]
        return (logits[:, self.pol.yes] - logits[:, self.pol.no]).float()

    def log_odds(self, states: Sequence[DecisionState]) -> torch.Tensor:
        prompts = [s.prompt for s in states]
        if not prompts:
            return torch.zeros(0)
        if any(not p for p in prompts):
            raise ValueError("LM backend needs plan spaces enumerated with prompts")
        if torch.is_grad_enabled():
            if len(prompts) > self.chunk:
                raise ValueError(f"{len(prompts)} prompts with grad > chunk={self.chunk}")
            return self._forward(prompts)
        return torch.cat([self._forward(prompts[i : i + self.chunk]) for i in range(0, len(prompts), self.chunk)])

    def save(self, run_dir: Path) -> None:
        self.pol.model.save_pretrained(str(Path(run_dir) / self.ckpt_name))


def accumulate_grad(
    backend: Backend, q: Question, obj: Objective, gen: torch.Generator | None, weight: float
) -> dict[str, float]:
    """Add weight * dLoss_q/dtheta to the parameters' .grad; return loss stats."""
    states = q.space.states
    if backend.chunk is None:
        loss, stats = question_loss(backend.log_odds(states).float().cpu(), q, obj, gen)
        (loss * weight).backward()
        return stats
    with torch.no_grad():
        z0 = backend.log_odds(states).float().cpu()
    z = z0.clone().requires_grad_(True)
    loss, stats = question_loss(z, q, obj, gen)
    (loss * weight).backward()
    grad_z = z.grad
    for i in range(0, len(states), backend.chunk):
        zc = backend.log_odds(states[i : i + backend.chunk])
        (zc * grad_z[i : i + backend.chunk].to(zc.device, zc.dtype)).sum().backward()
    return stats


# ---------------------------------------------------------------------- evaluation
def upper_hull_value(points: Sequence[tuple[float, float]], x: float) -> float:
    """Upper concave envelope of (cost, metric) points at cost x. Mixing fixed
    rules across questions reaches any point on it, so a learned policy must
    beat it, not just each rule."""
    best: dict[float, float] = {}
    for px, py in points:
        best[px] = max(py, best.get(px, -math.inf))
    hull: list[tuple[float, float]] = []
    for p in sorted(best.items()):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    if x <= hull[0][0]:
        return hull[0][1]
    if x >= hull[-1][0]:
        return hull[-1][1]
    for (x1, y1), (x2, y2) in zip(hull, hull[1:]):
        if x1 <= x <= x2:
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    return hull[-1][1]  # pragma: no cover


def _frontier_gap(rows: list[dict[str, Any]], sel: np.ndarray) -> float:
    pts = [
        (float(np.mean([rows[i][f"{r}_cost"] for i in sel])), float(np.mean([rows[i][f"{r}_metric"] for i in sel])))
        for r in FIXED_RULES
    ]
    cost = float(np.mean([rows[i]["cost"] for i in sel]))
    return float(np.mean([rows[i]["metric"] for i in sel])) - upper_hull_value(pts, cost)


@torch.no_grad()
def evaluate(
    backend: Backend,
    questions: list[Question],
    obj: Objective,
    *,
    fixed_rule: str | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Greedy (deployed) plan per question against the fixed-depth rules, on the
    table's mean scores. ``fixed_rule`` is the rule chosen on the training split.
    Points are on the 0-100 metric scale."""
    if not questions:
        raise ValueError("no questions to evaluate")
    rows: list[dict[str, Any]] = []
    for q in questions:
        m, c = q.metric(obj.metric), q.space.cost
        r = m - obj.lam * c
        if q.space.states:
            z = backend.log_odds(q.space.states).float().cpu()
            p = plan_logp(z, q.idx, q.act, q.mask).exp().numpy()
            g = q.space.greedy_plan(z.numpy())
            expand = [a for _, a in q.space.paths[g]]
        else:
            p, g, expand = np.ones(1), 0, []
        fixed = {rule: q.space.fixed_plan(rule) for rule in FIXED_RULES}
        row: dict[str, Any] = {
            "qid": q.qid,
            "metric": float(m[g]),
            "cost": float(c[g]),
            "reward": float(r[g]),
            "expected_reward": float((p * r).sum()),
            "expand_rate": float(np.mean(expand)) if expand else 0.0,
            "matches": [rule for rule, i in fixed.items() if i == g],
        }
        for rule, i in fixed.items():
            row.update({f"{rule}_metric": float(m[i]), f"{rule}_cost": float(c[i]), f"{rule}_reward": float(r[i])})
        if q.scores["kpr"].shape[1] >= 2:  # cross-fitted oracle: pick on half the replicates, score on the rest
            mr = q.metric_reps(obj.metric)
            a, b = mr[:, 0::2].mean(axis=1) - obj.lam * c, mr[:, 1::2].mean(axis=1) - obj.lam * c
            row["oracle_reward"] = float(0.5 * (b[int(np.argmax(a))] + a[int(np.argmax(b))]))
        rows.append(row)

    def mean(key: str) -> float:
        vals = [row[key] for row in rows if key in row]
        return float(np.mean(vals)) if vals else float("nan")

    out: dict[str, Any] = {
        "n_questions": len(rows),
        "objective": asdict(obj),
        "learned": {k: mean(k) for k in ("metric", "cost", "reward", "expected_reward", "expand_rate")},
        "fixed": {r: {k: mean(f"{r}_{k}") for k in ("metric", "cost", "reward")} for r in FIXED_RULES},
        "oracle_reward": mean("oracle_reward"),
        "share_equal_to_a_fixed_rule": float(np.mean([bool(row["matches"]) for row in rows])),
        "per_question": rows,
    }
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(rows), len(rows)) for _ in range(n_boot)]
    gap = _frontier_gap(rows, np.arange(len(rows)))
    gaps = [_frontier_gap(rows, s) for s in boots]
    out["frontier_gap_points"] = 100 * gap
    out["frontier_gap_ci_points"] = (100 * float(np.quantile(gaps, 0.025)), 100 * float(np.quantile(gaps, 0.975)))
    if fixed_rule is not None:
        diff = np.asarray([row["reward"] - row[f"{fixed_rule}_reward"] for row in rows])
        lo, hi = bootstrap_ci(diff, n_boot, seed)
        out["vs_fixed"] = {"rule": fixed_rule, "reward_points": 100 * float(diff.mean()), "ci_points": (100 * lo, 100 * hi)}
    return out


def best_fixed_rule(questions: list[Question], obj: Objective) -> str:
    means = {rule: float(np.mean([q.reward(obj.metric, obj.lam)[q.space.fixed_plan(rule)] for q in questions]))
             for rule in FIXED_RULES}
    return max(means, key=means.get)


# ------------------------------------------------------------------------ training
@dataclass
class TrainConfig:
    steps: int = 500
    lr: float = 1e-2
    batch: int = 0          # questions per step; 0 = all of them (exact full-batch gradient)
    eval_every: int = 10
    patience: int = 20      # evaluations without a val improvement before stopping; 0 = never stop
    grad_clip: float = 1.0
    seed: int = 0


def train(
    backend: Backend,
    train_qs: list[Question],
    val_qs: list[Question],
    obj: Objective,
    cfg: TrainConfig,
    run_dir: Path,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Maximise mean_q [J_q - beta * KL_q] on ``train_qs``; keep the checkpoint
    with the best greedy val reward (what deployment runs)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    learnable = [q for q in train_qs if q.space.states]  # no decision, no gradient
    if not learnable:
        raise ValueError("no training question has a decision to make")
    if not val_qs:
        raise ValueError("no validation questions")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    with torch.no_grad():  # the reference: the initial policy, frozen
        for q in learnable:
            q.z_ref = backend.log_odds(q.space.states).float().cpu()
    fixed_rule = best_fixed_rule(train_qs, obj)
    params = backend.parameters()
    opt = torch.optim.Adam(params, lr=cfg.lr)
    best: dict[str, Any] | None = None
    stale = 0
    train_stats: list[dict[str, float]] = []
    log_path = run_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            ev = evaluate(backend, val_qs, obj, fixed_rule=fixed_rule, n_boot=200, seed=cfg.seed)
            row = {
                "step": step,
                "train": {k: float(np.mean([s[k] for s in train_stats])) for k in train_stats[0]} if train_stats else {},
                "val": {**ev["learned"], "frontier_gap_points": ev["frontier_gap_points"]},
            }
            train_stats = []
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            log(json.dumps(row))
            if best is None or ev["learned"]["reward"] > best["learned"]["reward"] + 1e-12:
                best, stale = ev, 0
                best["step"] = step
                backend.save(run_dir)
            else:
                stale += 1
                if cfg.patience and stale >= cfg.patience:
                    log(f"early stop at step {step}: no val improvement in {stale} evaluations")
                    break
        if step == cfg.steps:
            break
        if cfg.batch <= 0 or cfg.batch >= len(learnable):
            batch = learnable
        else:
            batch = [learnable[i] for i in rng.choice(len(learnable), cfg.batch, replace=False)]
        opt.zero_grad(set_to_none=True)
        train_stats += [accumulate_grad(backend, q, obj, gen, 1.0 / len(batch)) for q in batch]
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()

    assert best is not None
    summary = {"best_step": best["step"], "fixed_rule": fixed_rule, "val": {k: v for k, v in best.items() if k != "per_question"}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
