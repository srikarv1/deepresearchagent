"""Merge the per-question metrics of one Table 2 variant that ran in several
run directories (e.g. a run restarted after a search outage) into one row.

    python scripts/table2_row.py table2-stopextbal          # run_name prefix
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

TEST = ["471082", "745702", "879779", "921842", "923549"]
CRIT = ["Clarity", "Breadth", "Depth", "Support", "Balance", "Insightfulness"]


def collect(prefix: str) -> dict:
    per = {}
    for d in sorted(glob.glob(f"runs/*-{prefix}-gpt5mini")) + sorted(glob.glob(f"runs/*-{prefix}-rest-gpt5mini")):
        d = Path(d)
        q_files = glob.glob(str(d / "metrics/deep_research_gym/quality_gpt-5-mini.json"))
        r_files = glob.glob(str(d / "metrics/deep_research_gym/relevance_gpt-5-mini.json"))
        if not q_files or not (d / "metrics/local.json").exists():
            continue
        q, r = json.loads(Path(q_files[0]).read_text()), json.loads(Path(r_files[0]).read_text())
        loc = {p["id"]: p for p in json.loads((d / "metrics/local.json").read_text())["per_query"]}
        for qid in q:
            if not (d / f"reports/{qid}.md").exists():
                continue
            s = {c: q[qid]["scores"][c][0] for c in CRIT}
            labels = [v[0] for v in r[qid]["labels"].values()]
            traj = json.loads((d / f"gpt_researcher/{qid}.json").read_text())
            per[qid] = {**s, "KPR": 100 * sum(x == "Supported" for x in labels) / len(labels),
                        "Overall": 10 * float(np.mean(list(s.values()))),
                        "tokens": loc[qid]["tokens"], "latency": loc[qid]["wall_s"], "calls": loc[qid]["n_llm_calls"],
                        "evidence_items": len(traj["evidence"]), "cost": traj.get("total_cost", 0.0), "run": d.name}
    return per


def main() -> None:
    prefix = sys.argv[1]
    per = collect(prefix)
    missing = [q for q in TEST if q not in per]
    m = {k: float(np.mean([per[q][k] for q in per])) for k in ["Clarity", "Breadth", "Depth", "Support", "KPR", "Overall", "tokens", "latency", "calls", "cost"]}
    row = {"n_questions": len(per), "missing": missing, **{k: round(v, 3) for k, v in m.items()},
           "q_per_1k_tok": round(m["Overall"] / (m["tokens"] / 1000), 3), "per_question": per}
    out = Path(f"runs/replaygym/bedrock/{prefix}_row.json")
    out.write_text(json.dumps(row, indent=1))
    print(json.dumps({k: v for k, v in row.items() if k != "per_question"}))
    for qid, v in sorted(per.items()):
        print(qid, "Overall", round(v["Overall"], 1), "KPR", round(v["KPR"], 1), "tok", v["tokens"], "evidence", v["evidence_items"], v["run"][:15])


if __name__ == "__main__":
    main()
