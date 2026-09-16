# Third-party evaluation repos

The official judges are **not** vendored. Everything in this directory except this file is gitignored, so nothing here is ever committed.

```bash
bash scripts/bootstrap_third_party.sh
adr doctor      # confirms what resolved
```

The script symlinks a checkout you already have rather than cloning a second copy. Override the lookup with `ADR_DRB_DIR`, or point at a different remote with `DRB_REPO_URL`.

| Used for                              | Accepted directory names                       | Marker file                                 |
| ------------------------------------- | ---------------------------------------------- | ------------------------------------------- |
| RACE + FACT                           | `deep_research_bench`                          | `deepresearch_bench_race.py`                |
| `gpt_researcher` agent (not a judge)  | `gpt-researcher`, `gpt_researcher`             | `gpt_researcher/utils/trajectory_logger.py` |

BrowseComp-Plus judging is **in-harness** (`adr evaluate <run> --official browsecomp_plus`): accuracy uses the official grader prompt, recall uses `evidence_docs` already on each query. The upstream `texttron/BrowseComp-Plus` checkout is optional.

Upstream: [Ayanami0730/deep_research_bench](https://github.com/Ayanami0730/deep_research_bench). The `gpt_researcher` row is the agent under test, not a judge: [WilliamOdinson/gpt-researcher](https://github.com/WilliamOdinson/gpt-researcher). Override with `GR_REPO_URL` or `ADR_GR_DIR`.

The harness writes reports in the official DRB format, invokes the judge scripts as subprocesses, and parses their result files back into scores.
