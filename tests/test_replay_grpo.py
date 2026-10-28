import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adr.replay import tree as tree_mod  # noqa: E402
from adr.replay.grpo import (  # noqa: E402
    MLPBackend,
    Objective,
    Question,
    TrainConfig,
    accumulate_grad,
    plan_logp,
    question_loss,
    train,
    upper_hull_value,
)
from adr.replay.plans import FIXED_RULES, enumerate_plans  # noqa: E402
from adr.replay.rl import EpisodeState, rollout  # noqa: E402
from adr.replay.table import (  # noqa: E402
    IncompleteTable,
    RewardTable,
    ScoredQuestion,
    TableMismatch,
    fill_table,
    gate_stats,
)
from adr.replay.tree import ResearchTree  # noqa: E402


# ----------------------------------------------------------------------- fixtures
def _write_tree(tmp_path, n_roots=2, kids=2, grand=2, name="t", contexts=None, learnings=None):
    contexts, learnings = contexts or {}, learnings or {}
    nodes = []

    def node(nid, parent, depth):
        nodes.append({
            "node_id": nid, "parent_node_id": parent, "tree_depth": depth, "research_goal": f"goal {nid}",
            "learnings": learnings.get(nid, [f"L{nid}"]), "citations": {}, "context": contexts.get(nid, nid),
            "cost": {"tokens_input": 100},
        })

    for r in range(n_roots):
        rid = f"r{r}"
        node(rid, None, 1)
        for c in range(kids):
            cid = f"{rid}c{c}"
            node(cid, rid, 2)
            for g in range(grand):
                node(f"{cid}g{g}", cid, 3)
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"query": f"question {name}", "nodes": nodes}))
    return ResearchTree.from_trajectory(p)


def _emb(tree, seed=0, dim=8):
    rng = np.random.default_rng(seed)
    k = len(tree.order)
    m = rng.normal(size=(1 + 2 * k, dim)).astype("float32")
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return {"query": m[0], "goal": m[1:1 + k], "learn": m[1 + k:]}


def _question(space, metric_values, reps=2, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = np.asarray(metric_values, np.float64)[:, None]
    scores = {m: np.clip(base + noise * rng.normal(size=(len(base), reps)), 0.0, 1.0) for m in ("kpr", "quality")}
    return Question(space.qid, space, scores)


# ---------------------------------------------------------------------- plan space
@pytest.mark.parametrize("n_roots,plans,states", [(2, 9, 6), (4, 81, 44)])
def test_enumerates_every_reachable_plan_once(tmp_path, n_roots, plans, states):
    t = _write_tree(tmp_path, n_roots=n_roots)
    space = enumerate_plans("q", t, _emb(t), n_full=len(t.nodes))
    assert len(space.plans) == plans == len(set(space.plans))
    assert len(space.states) == states
    assert all(t.is_valid(p) for p in space.plans)
    for rule, depth in zip(FIXED_RULES, (1, 2, 3)):
        assert space.plans[space.fixed_plan(rule)] == frozenset(t.up_to_depth(depth))


def test_paths_replay_to_their_plans_and_states(tmp_path):
    t = _write_tree(tmp_path)
    e = _emb(t)
    space = enumerate_plans("q", t, e)
    for plan, path in zip(space.plans, space.paths):
        st = EpisodeState("q", t, e)
        for s, a in path:
            np.testing.assert_allclose(st.features(), space.states[s].features)
            st.step(a)
        assert st.done and st.subtree() == plan


def test_missing_children_shrink_the_plan_space(tmp_path):
    t = _write_tree(tmp_path, n_roots=2, kids=2, grand=0)
    # r1 loses its children: only r0 has a decision, and it has no grandchildren
    t.nodes["r1"].children = []
    space = enumerate_plans("q", t, _emb(t))
    assert len(space.plans) == 2 and len(space.states) == 1


def test_plan_probabilities_form_a_distribution(tmp_path):
    t = _write_tree(tmp_path, n_roots=4)
    space = enumerate_plans("q", t, _emb(t))
    q = _question(space, np.zeros(len(space.plans)))
    z = torch.randn(len(space.states), generator=torch.Generator().manual_seed(0)) * 2
    assert float(plan_logp(z, q.idx, q.act, q.mask).exp().sum()) == pytest.approx(1.0, abs=1e-5)


def test_greedy_plan_is_the_live_threshold_rollout(tmp_path):
    t = _write_tree(tmp_path, n_roots=4)
    e = _emb(t, seed=3)
    space = enumerate_plans("q", t, e)
    torch.manual_seed(1)
    net = torch.nn.Sequential(torch.nn.Linear(10, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    with torch.no_grad():
        z = net(torch.as_tensor(np.stack([s.features for s in space.states]))).squeeze(-1).numpy()
        st = EpisodeState("q", t, e)
        rollout(st, lambda f: int(float(net(torch.as_tensor(f)).item()) >= 0.0))
    assert space.plans[space.greedy_plan(z)] == st.subtree()


def test_episode_state_uses_the_live_tree_size(tmp_path):
    t = _write_tree(tmp_path)
    e = _emb(t)
    assert EpisodeState("q", t, e, n_full=28).features()[7] == pytest.approx(2 / 28)
    with pytest.raises(ValueError):
        EpisodeState("q", t, e, n_full=len(t.nodes) - 1)


def test_embedder_cache_is_separate_per_model(tmp_path, monkeypatch):
    from adr.replay.rl import LOCAL_EMBED_MODEL, OPENAI_EMBED_MODEL, Embedder

    monkeypatch.setenv("REPLAY_EMBED", "local")
    assert Embedder(tmp_path).cache_dir == tmp_path / LOCAL_EMBED_MODEL
    monkeypatch.delenv("REPLAY_EMBED")
    assert Embedder(tmp_path).cache_dir == tmp_path / OPENAI_EMBED_MODEL


# ----------------------------------------------------------------------- objective
def _space_and_question(tmp_path, seed=0, n_roots=2):
    t = _write_tree(tmp_path, n_roots=n_roots, name=f"t{seed}")
    space = enumerate_plans(f"q{seed}", t, _emb(t, seed=seed))
    rng = np.random.default_rng(seed)
    return space, _question(space, rng.uniform(0.3, 0.7, len(space.plans)))


def test_sampled_rloo_grpo_is_unbiased_for_the_exact_gradient(tmp_path):
    _, q = _space_and_question(tmp_path, seed=2)
    z0 = torch.randn(len(q.space.states), generator=torch.Generator().manual_seed(0))
    q.z_ref = z0.clone()

    def grad(obj, gen=None):
        z = z0.clone().requires_grad_(True)
        loss, _ = question_loss(z, q, obj, gen)
        loss.backward()
        return z.grad.numpy().copy()

    exact = grad(Objective(lam=0.0, beta=0.0))
    gen = torch.Generator().manual_seed(7)
    draws = np.stack([grad(Objective(lam=0.0, beta=0.0, group=8), gen) for _ in range(3000)])
    se = draws.std(axis=0) / np.sqrt(len(draws))
    assert np.all(np.abs(draws.mean(axis=0) - exact) <= 4 * se + 1e-4)


def test_kl_vanishes_at_the_reference(tmp_path):
    _, q = _space_and_question(tmp_path)
    z = torch.randn(len(q.space.states), generator=torch.Generator().manual_seed(0)).requires_grad_(True)
    q.z_ref = z.detach().clone()
    _, stats = question_loss(z, q, Objective(beta=1.0))
    assert stats["kl"] == pytest.approx(0.0, abs=1e-6)


def test_two_pass_chunked_gradient_equals_one_pass(tmp_path):
    _, q = _space_and_question(tmp_path, n_roots=4)
    one = MLPBackend(hidden=8, seed=0)
    with torch.no_grad():  # non-trivial output layer so every parameter gets a gradient
        one.net[2].weight.normal_()
    two = copy.deepcopy(one)
    two.chunk = 5  # forces the no-grad pass + chunked surrogate backward
    obj = Objective(lam=0.01, beta=0.1)
    for b in (one, two):
        with torch.no_grad():
            q.z_ref = b.log_odds(q.space.states).float() * 0.5
        accumulate_grad(b, q, obj, None, 1.0)
    for p1, p2 in zip(one.parameters(), two.parameters()):
        torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-5, atol=1e-7)


def test_upper_hull_interpolates_mixtures_of_fixed_rules():
    pts = [(0.2, 0.40), (0.5, 0.42), (1.0, 0.50)]  # (0.5, 0.42) lies under the d1-d3 chord
    assert upper_hull_value(pts, 0.5) == pytest.approx(0.40 + 0.10 * 0.3 / 0.8)
    assert upper_hull_value(pts, 0.1) == pytest.approx(0.40)
    assert upper_hull_value([(0.2, 0.4), (0.2, 0.5)], 0.2) == pytest.approx(0.5)


# ------------------------------------------------------------------------ learning
def _threshold_task(tmp_path, n, offset):
    """Expanding pays +v when the parent's findings point at the question
    (features[6] > 0) and -v otherwise; v is larger for children than for
    grandchildren, so 'expand iff features[6] > 0' is the optimal policy."""
    qs = []
    for i in range(n):
        t = _write_tree(tmp_path, n_roots=2, name=f"thr{offset + i}")
        space = enumerate_plans(f"q{offset + i}", t, _emb(t, seed=offset + i), n_full=len(t.nodes))
        v = np.asarray([(0.05 if s.level == 2 else 0.02) * (1 if s.features[6] > 0 else -1) for s in space.states])
        value = np.asarray([0.5 + sum(a * v[s] for s, a in path) for path in space.paths])
        qs.append(_question(space, value))
    return qs


def test_exact_grpo_learns_the_optimal_threshold_policy(tmp_path):
    train_qs, val_qs = _threshold_task(tmp_path, 24, 0), _threshold_task(tmp_path, 12, 100)
    obj = Objective(metric="kpr", lam=0.0, beta=0.001)
    cfg = TrainConfig(steps=300, lr=0.05, eval_every=25, patience=0, seed=0)
    summary = train(MLPBackend(hidden=16, seed=0), train_qs, val_qs, obj, cfg, tmp_path / "run")
    val = summary["val"]
    best_fixed = max(v["reward"] for v in val["fixed"].values())
    assert val["learned"]["reward"] > best_fixed + 0.02
    assert val["learned"]["reward"] >= val["oracle_reward"] - 0.01
    assert val["frontier_gap_points"] > 2.0
    assert (tmp_path / "run" / "best.pt").exists() and (tmp_path / "run" / "train_log.jsonl").exists()


def test_export_matches_the_fork_forward(tmp_path):
    b = MLPBackend(hidden=6, seed=3)
    with torch.no_grad():
        b.net[2].weight.normal_()
        b.net[2].bias.fill_(0.3)
    w = b.export({"embed_model": "bge-small-en-v1.5"})
    f = np.random.default_rng(0).normal(size=10).astype(np.float32)
    # gpt_researcher/orchestration/replaygym.py ExpansionPolicy.p_deeper
    h = np.tanh(np.asarray(w["w1"]) @ f + np.asarray(w["b1"]))
    p = 1.0 / (1.0 + np.exp(-float((np.asarray(w["w2"]) @ h + np.asarray(w["b2"])).reshape(-1)[0])))
    with torch.no_grad():
        assert p == pytest.approx(float(torch.sigmoid(b.net(torch.as_tensor(f)))), abs=1e-6)


# ----------------------------------------------------------------- writer context
def _fake_fork():
    """Copies of the fork's count_words / trim_context_to_word_limit / pack_context_head."""

    def count_words(text):
        if isinstance(text, list):
            text = " ".join(str(item) for item in text)
        return len(str(text).split())

    def trim_context_to_word_limit(context_list, max_words=25000):
        total, out = 0, []
        for item in reversed(context_list):
            text = " ".join(str(part) for part in item) if isinstance(item, list) else str(item)
            words = count_words(item)
            if total + words <= max_words:
                out.insert(0, item)
                total += words
            elif not out:
                out.insert(0, " ".join(text.split()[:max_words]))
                break
            else:
                break
        return out

    def pack_context_head(context_list, max_words=25000):
        packed, total = [], 0
        for item in context_list:
            words = count_words(item)
            if total + words <= max_words:
                packed.append(item)
                total += words
        return packed

    return SimpleNamespace(trim_context_to_word_limit=trim_context_to_word_limit, pack_context_head=pack_context_head)


def test_writer_context_reproduces_the_live_nested_trim(tmp_path, monkeypatch):
    monkeypatch.setattr(tree_mod, "_fork_skill", _fake_fork)
    t = _write_tree(
        tmp_path, n_roots=2, kids=2, grand=0,
        contexts={"r0": "a a", "r1": "b b", "r0c0": "c c c c", "r0c1": "d d d d"},
        learnings={"r0": ["L0"], "r1": ["L1"], "r0c0": [], "r0c1": [], "r1c0": [], "r1c1": []},
    )
    plan = {"r0", "r1", "r0c0", "r0c1"}
    # Live: the top call returns trim(["a a", "b b", "c c c c", "d d d d"], 9) =
    # ["c c c c", "d d d d"]; run() prepends the learnings and packs.
    assert t.writer_context(plan, "tail", max_words=9).split("\n") == ["L1", "c c c c", "d d d d"]
    assert t.writer_context(plan, "head", max_words=9).split("\n") == ["L0", "L1", "c c c c"]
    with pytest.raises(ValueError):
        t.writer_context({"r0c0"}, "tail")


# -------------------------------------------------------------------------- table
def test_table_fill_retries_resumes_and_refuses_to_mix(tmp_path, monkeypatch):
    monkeypatch.setattr(tree_mod, "_fork_skill", _fake_fork)
    t = _write_tree(tmp_path, n_roots=2, kids=2, grand=0)
    space = enumerate_plans("q", t, _emb(t))
    calls = {"synth": 0, "judge": 0}

    def synth(query, contexts, conc):
        calls["synth"] += len(contexts)
        return {tag: f"report {tag} " + "x " * 1500 for tag in contexts}

    def judge(qid, query, reports, work):
        calls["judge"] += 1
        # the first judge call drops one report, as a rate-limited judge would
        drop = sorted(reports)[0] if calls["judge"] == 1 else None
        return {tag: (None if tag == drop else {"quality": 60.0, "kpr": 40.0}) for tag in reports}

    kw = dict(root=tmp_path / "table", judge_model="judge-x", pack="tail", max_words=9, synthesize=synth, judge=judge)
    stats = fill_table(t, space, reps=2, **kw)
    assert stats["missing"] == 0 and stats["synthesized"] == 2 * len(space.plans)
    scores = RewardTable.load(tmp_path / "table", "q").scores(space, min_reps=2)
    assert scores["kpr"].shape == (len(space.plans), 2) and np.allclose(scores["kpr"], 0.4)
    before = dict(calls)
    assert fill_table(t, space, reps=2, **kw)["missing"] == 0 and calls == before
    with pytest.raises(TableMismatch):
        fill_table(t, space, reps=2, **{**kw, "pack": "head"})
    with pytest.raises(IncompleteTable):
        RewardTable.load(tmp_path / "table", "q").scores(space, min_reps=3)


def test_gate_crossfit_does_not_count_judge_luck_as_headroom(tmp_path):
    qs = []
    for i in range(40):  # every plan is truly worth 0.5; scores are pure noise
        t = _write_tree(tmp_path, n_roots=2, name=f"g{i}")
        space = enumerate_plans(f"g{i}", t, _emb(t, seed=i))
        rng = np.random.default_rng(i)
        noisy = np.clip(0.5 + 0.05 * rng.normal(size=(len(space.plans), 2)), 0, 1)
        qs.append(ScoredQuestion(f"g{i}", space, {"kpr": noisy, "quality": noisy}))
    s = gate_stats(qs, metric="kpr", lam=0.0, n_boot=500)
    assert s["noise_sd_points"] == pytest.approx(5.0, rel=0.2)
    assert s["snr"] < 0.5
    # picking the best plan on the same noisy scores looks like 2+ points of headroom ...
    assert s["oracle_insample_reward"] - s["fixed_reward"][s["best_fixed"]] > 0.02
    # ... but cross-fitted, the gate does not pass on pure noise
    assert s["headroom_ci_points"][0] < 0.0


# --------------------------------------------------------------------- LM backend
def _tiny_lm(tmp_path):
    """A random 2-layer Qwen2 with a byte-level BPE where " YES" and " NO" are
    single tokens, as in the real Qwen vocabulary. Nothing is downloaded."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {"[PAD]": 0}
    for ch in sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet()):
        vocab.setdefault(ch, len(vocab))
    merges = [("Ġ", "Y"), ("ĠY", "E"), ("ĠYE", "S"), ("Ġ", "N"), ("ĠN", "O")]
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    tok = tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocab, merges=merges))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = tokenizers.decoders.ByteLevel()
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="[PAD]", model_input_names=["input_ids", "attention_mask"]
    )
    cfg = transformers.Qwen2Config(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    path = tmp_path / "tiny-lm"
    transformers.Qwen2ForCausalLM(cfg).save_pretrained(path)
    fast.save_pretrained(path)
    return path


def test_lm_backend_two_pass_gradient_and_adapter_roundtrip(tmp_path):
    from adr.replay.grpo import LMBackend
    from adr.replay.lm_policy import decision_prompt

    model = _tiny_lm(tmp_path)
    t = _write_tree(tmp_path, n_roots=2)
    space = enumerate_plans("q", t, _emb(t), prompt_fn=decision_prompt)
    q = _question(space, np.random.default_rng(0).uniform(0.3, 0.7, len(space.plans)))
    lm = LMBackend(str(model), lora_r=4, chunk=2)
    with torch.no_grad():  # LoRA B starts at zero; make every adapter weight matter
        for p in lm.parameters():
            p.normal_(std=0.1)
        q.z_ref = lm.log_odds(space.states).float().cpu() * 0.5
    obj = Objective(lam=0.01, beta=0.1)

    accumulate_grad(lm, q, obj, None, 1.0)
    two_pass = [p.grad.clone() for p in lm.parameters()]
    for p in lm.parameters():
        p.grad = None
    # Reference: one graph over the same chunks (same padding, so the same function).
    z = torch.cat([lm._forward([s.prompt for s in space.states[i:i + 2]]) for i in range(0, len(space.states), 2)])
    loss, _ = question_loss(z.float().cpu(), q, obj)
    loss.backward()
    for g2, p in zip(two_pass, lm.parameters()):
        torch.testing.assert_close(g2, p.grad, rtol=1e-4, atol=1e-6)

    lm.save(tmp_path)
    again = LMBackend(str(model), lora_r=4, init=tmp_path / LMBackend.ckpt_name, chunk=2)
    with torch.no_grad():
        torch.testing.assert_close(again.log_odds(space.states), lm.log_odds(space.states))
    with pytest.raises(ValueError):
        LMBackend(str(model), lora_r=8, init=tmp_path / LMBackend.ckpt_name)
