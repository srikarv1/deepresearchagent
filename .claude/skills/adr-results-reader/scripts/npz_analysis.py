"""Common npz analysis operations for adr run results.

Usage:
    python scripts/npz_analysis.py <npz_path> <traj_json_path> [--item <item_id>]

Examples:
    python scripts/npz_analysis.py runs/20260915-025610-gpt-researcher-bench/gpt_researcher/51_emb.npz \
                                   runs/20260915-025610-gpt-researcher-bench/gpt_researcher/51.json

    python scripts/npz_analysis.py gpt_researcher/51_emb.npz gpt_researcher/51.json --item 072cbf49e212
"""

import argparse
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def load(npz_path: str, traj_path: str):
    npz = np.load(npz_path, allow_pickle=True)
    traj = json.loads(open(traj_path).read())
    ids = list(npz["ids"])
    vecs = npz["vectors"]
    q = npz["query"][0]
    return npz, traj, ids, vecs, q


def rank_evidence(ids, vecs, q, traj, top_n=10):
    """Rank all evidence by cosine similarity to the root query."""
    sims = vecs @ q
    ranked = np.argsort(-sims)
    print(f"{'rank':<6} {'item_id':<14} {'cos':>6} {'retained':>9} {'url'}")
    print("-" * 90)
    for rank, i in enumerate(ranked[:top_n], 1):
        ev = traj["evidence"][ids[i]]
        print(
            f"{rank:<6} {ids[i]:<14} {sims[i]:>6.3f} {ev['was_retained']!s:>9} {ev['source_url'][:55]}"
        )
    print(f"\n... {len(ids)} items total")


def retained_vs_pruned(ids, vecs, q, traj):
    """Compare cosine distributions of retained vs pruned evidence."""
    sims = vecs @ q
    ret_mask = np.array([traj["evidence"][ids[i]]["was_retained"] for i in range(len(ids))])
    ret_sims = sims[ret_mask]
    pru_sims = sims[~ret_mask]
    print(
        f"retained ({ret_sims.shape[0]:>3}): mean={ret_sims.mean():.3f}  min={ret_sims.min():.3f}  max={ret_sims.max():.3f}"
    )
    print(
        f"pruned   ({pru_sims.shape[0]:>3}): mean={pru_sims.mean():.3f}  min={pru_sims.min():.3f}  max={pru_sims.max():.3f}"
    )


def frontier_alignment(npz):
    """Show which frontier node aligns with which subquestion."""
    node_sq = npz["nodes"] @ npz["subquestions"].T
    n_ids = list(npz["nodes_ids"])
    print(f"{'node_id':<14} {'best_sq':>7} {'cos':>6}")
    print("-" * 30)
    for i, nid in enumerate(n_ids):
        best = int(np.argmax(node_sq[i]))
        print(f"{nid:<14} {best + 1:>7} {node_sq[i, best]:>6.3f}")


def evidence_coverage(ids, vecs, npz, traj, threshold=0.5):
    """Count evidence items within cosine threshold of each frontier node."""
    ev_node = vecs @ npz["nodes"].T
    n_ids = list(npz["nodes_ids"])
    print(f"{'node_id':<14} {'within_' + str(threshold):>10} {'retained':>9}")
    print("-" * 36)
    for i, nid in enumerate(n_ids):
        mask = ev_node[:, i] > threshold
        n_close = int(mask.sum())
        n_ret = sum(1 for j in np.where(mask)[0] if traj["evidence"][ids[j]]["was_retained"])
        print(f"{nid:<14} {n_close:>10} {n_ret:>9}")


def lookup_item(item_id, ids, vecs, q, npz, traj):
    """Full lookup of a single evidence item across JSON and npz."""
    ev = traj["evidence"].get(item_id)
    if ev is None:
        print(f"{item_id} not found in evidence dict")
        return

    print(f"item_id:          {ev['item_id']}")
    print(f"source_url:       {ev['source_url']}")
    print(f"source_subquery:  {ev['source_subquery'][:120]}")
    print(f"tree_depth:       {ev['tree_depth']}")
    print(f"retrieval_round:  {ev['retrieval_round']}")
    print(f"was_retained:     {ev['was_retained']}")
    print(f"word_count:       {ev['word_count']}")
    print(f"content[:200]:    {ev['content'][:200]}")

    # Round history
    print("\nround history:")
    for r in traj["rounds"]:
        in_new = item_id in r["new_item_ids"]
        in_kept = item_id in r["decision"]["kept_item_ids"]
        in_pruned = item_id in r["decision"]["pruned_item_ids"]
        if in_new or in_kept or in_pruned:
            print(f"  round {r['round_id']}: new={in_new} kept={in_kept} pruned={in_pruned}")

    # Vector analysis
    if item_id not in ids:
        print(f"\n{item_id} not in npz ids array")
        return

    idx = ids.index(item_id)
    vec = vecs[idx]
    cos_q = float(vec @ q)
    all_sims = vecs @ q
    rank = int((all_sims > cos_q).sum()) + 1

    print(f"\ncosine to query:  {cos_q:.4f} (rank {rank}/{len(ids)})")

    # Closest frontier node
    n_sims = vec @ npz["nodes"].T
    best_n = int(np.argmax(n_sims))
    best_nid = list(npz["nodes_ids"])[best_n]
    print(f"closest node:     {best_nid} (cos={n_sims[best_n]:.4f})")

    # Most similar retained items
    item_sims = vecs @ vec
    ranked = np.argsort(-item_sims)
    print("\nmost similar retained items:")
    count = 0
    for j in ranked:
        if ids[j] != item_id and traj["evidence"][ids[j]]["was_retained"]:
            print(
                f"  {ids[j]}  cos={item_sims[j]:.3f}  url={traj['evidence'][ids[j]]['source_url'][:70]}"
            )
            count += 1
            if count >= 3:
                break


def main():
    parser = argparse.ArgumentParser(description="Analyze adr npz embedding files")
    parser.add_argument("npz_path", help="Path to <id>_emb.npz")
    parser.add_argument("traj_path", help="Path to <id>.json (fork trajectory)")
    parser.add_argument("--item", help="Look up a specific evidence item by ID")
    parser.add_argument(
        "--top", type=int, default=10, help="Number of top items to show in ranking"
    )
    args = parser.parse_args()

    npz, traj, ids, vecs, q = load(args.npz_path, args.traj_path)

    if args.item:
        lookup_item(args.item, ids, vecs, q, npz, traj)
        return

    print("=" * 60)
    print("EVIDENCE RANKING (by cosine to root query)")
    print("=" * 60)
    rank_evidence(ids, vecs, q, traj, top_n=args.top)

    print(f"\n{'=' * 60}")
    print("RETAINED vs PRUNED")
    print("=" * 60)
    retained_vs_pruned(ids, vecs, q, traj)

    print(f"\n{'=' * 60}")
    print("FRONTIER NODE ALIGNMENT (to subquestions)")
    print("=" * 60)
    frontier_alignment(npz)

    print(f"\n{'=' * 60}")
    print("EVIDENCE COVERAGE PER FRONTIER NODE")
    print("=" * 60)
    evidence_coverage(ids, vecs, npz, traj)


if __name__ == "__main__":
    main()
