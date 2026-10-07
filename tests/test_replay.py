import json

import pytest

from adr.replay.shapley import all_coalitions, banzhaf, shapley
from adr.replay.tree import ResearchTree


def test_shapley_additive_game_recovers_item_values():
    players = ["a", "b", "c"]
    val = {"a": 3.0, "b": -1.0, "c": 0.5}
    v = {s: sum(val[p] for p in s) for s in all_coalitions(players)}
    assert shapley(players, v) == pytest.approx(val)
    assert banzhaf(players, v) == pytest.approx(val)


def test_shapley_efficiency_with_interaction():
    players = ["a", "b"]
    v = {frozenset(): 0.0, frozenset({"a"}): 1.0, frozenset({"b"}): 1.0, frozenset({"a", "b"}): 1.0}  # redundant
    phi = shapley(players, v)
    assert sum(phi.values()) == pytest.approx(1.0)
    assert phi["a"] == pytest.approx(0.5)


def _tree(tmp_path):
    nodes = [
        {"node_id": "r1", "parent_node_id": None, "tree_depth": 1, "learnings": ["L1"], "citations": {"L1": "u1"}, "context": "C1", "cost": {"tokens_input": 10, "tokens_output": 2}},
        {"node_id": "c1", "parent_node_id": "r1", "tree_depth": 2, "learnings": ["L2"], "citations": {}, "context": "C2", "cost": {"tokens_input": 5}},
        {"node_id": "r2", "parent_node_id": None, "tree_depth": 1, "learnings": ["L3", "L1"], "citations": {}, "context": "C3", "cost": {"tokens_input": 7}},
    ]
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"query": "q", "nodes": nodes}))
    return ResearchTree.from_trajectory(p)


def test_tree_structure_context_and_cost(tmp_path):
    t = _tree(tmp_path)
    assert t.roots == ["r1", "r2"]
    assert t.subtree("r1") == ["r1", "c1"]
    assert t.is_valid({"r1", "c1"}) and not t.is_valid({"c1"})
    assert t.up_to_depth(1) == {"r1", "r2"}
    assert t.tokens(t.all_nodes()) == 24
    ctx = t.context({"r1", "c1"}).split("\n")
    assert ctx == ["L1 [Source: u1]", "L2", "C1", "C2"]
    assert "L1 [Source: u1]" in t.context({"r1", "r2"}) and t.context({"r1", "r2"}).count("L1") == 1


def test_tree_requires_node_cache(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"query": "q", "nodes": []}))
    with pytest.raises(ValueError):
        ResearchTree.from_trajectory(p)


# --- ReplayGym environment ------------------------------------------------------

def _full_tree(tmp_path):  # noqa: D401
    """2 roots, each with 2 children, each with 2 grandchildren (b2 d3)."""
    nodes, k = [], 0
    for r in range(2):
        rid = f"r{r}"
        nodes.append({"node_id": rid, "research_goal": f"goal {rid}", "parent_node_id": None, "tree_depth": 1, "learnings": [f"L{rid}"], "citations": {}, "context": rid, "cost": {"tokens_input": 100}})
        for c in range(2):
            cid = f"{rid}c{c}"
            nodes.append({"node_id": cid, "research_goal": f"goal {cid}", "parent_node_id": rid, "tree_depth": 2, "learnings": [f"L{cid}"], "citations": {}, "context": cid, "cost": {"tokens_input": 10}})
            for g in range(2):
                gid = f"{cid}g{g}"
                nodes.append({"node_id": gid, "research_goal": f"goal {gid}", "parent_node_id": cid, "tree_depth": 3, "learnings": [f"L{gid}"], "citations": {}, "context": gid, "cost": {"tokens_input": 1}})
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"query": "q", "nodes": nodes}))
    return ResearchTree.from_trajectory(p)


def _emb(tree):
    import numpy as np
    rng = np.random.default_rng(0)
    k = len(tree.order)
    m = rng.normal(size=(1 + 2 * k, 8)).astype("float32")
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return {"query": m[0], "goal": m[1:1 + k], "learn": m[1 + k:]}


def test_episode_fixed_policies_give_expected_subtrees(tmp_path):
    from adr.replay.rl import EpisodeState, fixed_policy, rollout

    t = _full_tree(tmp_path)
    e = _emb(t)
    for kind, depth in (("d1", 1), ("d2", 2), ("full", 3)):
        st = EpisodeState("q", t, e)
        trace = rollout(st, fixed_policy(kind)(st))
        assert st.subtree() == frozenset(t.up_to_depth(depth))
        assert all(-1.01 <= x <= 1.01 for d in trace for x in d.feat)
    st = EpisodeState("q", t, e)
    rollout(st, lambda f: 0)                      # never go deeper = first level only
    assert st.subtree() == frozenset(t.up_to_depth(1))


def test_episode_decisions_are_per_branch(tmp_path):
    from adr.replay.rl import EpisodeState, rollout

    t = _full_tree(tmp_path)
    st = EpisodeState("q", t, _emb(t))
    # live order: the first round decides for every branch at once
    # (r0 deeper=1, r1 deeper=0), then r0's children decide full depth (1)
    acts = iter([1, 0, 1])
    trace = rollout(st, lambda f: next(acts))
    assert len(trace) == 3
    assert st.subtree() == frozenset(t.subtree("r0")) | {"r1"}


def test_gym_env_runs_with_stub_reward(tmp_path):
    from adr.replay.rl import ReplayExpansionEnv

    t = _full_tree(tmp_path)

    class Stub:
        def scores(self, qid, tree, subs):
            return [{"quality": 50.0, "kpr": 50.0, "tokens": tree.tokens(s), "n": len(s)} for s in subs]

        def reward(self, sc):
            return 0.5 * sc["quality"] + 0.5 * sc["kpr"] - 2.0 * sc["tokens"] / 1e5

    env = ReplayExpansionEnv({"q": t}, {"q": _emb(t)}, Stub())
    obs, info = env.reset(options={"qid": "q"})
    assert obs.shape == (10,) and info["qid"] == "q"
    done, total = False, 0.0
    while not done:
        obs, r, done, _, _ = env.step(1)
        total += r
    assert total == pytest.approx(50.0 - 2.0 * t.tokens(t.all_nodes()) / 1e5)


def test_actions_for_subtree_reproduce_the_subtree(tmp_path):
    from adr.replay.lm_policy import actions_for_subtree, decision_prompt
    from adr.replay.rl import EpisodeState

    t = _full_tree(tmp_path)
    e = _emb(t)
    target = frozenset(t.subtree("r0")) | {"r1"}
    seq = actions_for_subtree(EpisodeState("q", t, e), target)
    st = EpisodeState("q", t, e)
    for _, a in seq:
        st.step(a)
    assert st.subtree() == target
    assert "Answer YES or NO" in decision_prompt(EpisodeState("q", t, e))


def test_first_round_decisions_share_one_state(tmp_path):
    """Live, all top-level 'go deeper?' decisions are made at the same checkpoint,
    before any child research, so they must see identical budget state."""
    from adr.replay.rl import EpisodeState

    t = _full_tree(tmp_path)
    st = EpisodeState("q", t, _emb(t))
    f_r0 = st.features(); st.step(1)
    f_r1 = st.features()
    assert f_r0[7] == f_r1[7]              # nothing researched between them
    assert st.queue[0][0] == 2             # still a first-round decision
    st.step(1)
    assert st.queue and st.queue[0][0] == 3  # r0's children researched, now full-depth decision
    assert set(t.nodes["r0"].children) <= st.researched
    assert not (set(t.nodes["r1"].children) & st.researched)  # depth-first: r1 not yet


def test_live_policy_features_match_training_env(tmp_path):
    """The fork's live ExpansionPolicy must see exactly the features the policy
    was trained on in ReplayGym, at the same decision points."""
    import sys

    import numpy as np

    sys.path.insert(0, "third_party/gpt-researcher")
    from gpt_researcher.orchestration.replaygym import ExpansionPolicy, full_tree_size

    from adr.replay.rl import EpisodeState

    t = _full_tree(tmp_path)
    e = _emb(t)
    idx = {n: i for i, n in enumerate(t.order)}
    text2vec = {t.query: e["query"]}
    for n in t.order:
        text2vec[t.nodes[n].research_goal or t.nodes[n].query] = e["goal"][idx[n]]
        text2vec[" ".join(t.nodes[n].learnings) or "(none)"] = e["learn"][idx[n]]
    w = {"w1": np.zeros((4, 10)).tolist(), "b1": [0.0] * 4, "w2": [[0.0] * 4], "b2": [0.0]}
    live = ExpansionPolicy(w, embed=lambda texts: np.stack([text2vec[x] for x in texts]))
    n_full = len(t.nodes)

    def goals(ns):
        return [t.nodes[n].research_goal or t.nodes[n].query for n in ns]

    env = EpisodeState("q", t, e)
    live.set_query(t.query)
    live.observe(t.roots, goals(t.roots), [t.nodes[n].learnings for n in t.roots])
    for r in t.roots:  # first-round checkpoint: one decision per branch
        np.testing.assert_allclose(live.features([r], 2, n_full), env.features(), atol=1e-6)
        env.step(1)
    r0 = t.roots[0]  # depth-first: r0's children get researched, then decide full depth
    kids = list(t.nodes[r0].children)
    live.observe(kids, goals(kids), [t.nodes[n].learnings for n in kids])
    np.testing.assert_allclose(live.features(kids, 3, n_full), env.features(), atol=1e-6)
    assert full_tree_size(2, 3) == 14 and full_tree_size(4, 3) == 28
