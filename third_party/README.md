# Third-party evaluation repos

The official judges are **not** vendored. Everything in this directory except this file is gitignored, so nothing here is ever committed.

```bash
bash scripts/bootstrap_third_party.sh
adr doctor      # confirms what resolved
```

The script symlinks a checkout you already have rather than cloning a second copy. Override the lookup with `ADR_DRB_DIR` / `ADR_GYM_DIR`, or point at a different remote with `DRB_REPO_URL` / `GYM_REPO_URL`.

| Used for                              | Accepted directory names                       | Marker file                                 |
| ------------------------------------- | ---------------------------------------------- | ------------------------------------------- |
| RACE + FACT                           | `deep_research_bench`                          | `deepresearch_bench_race.py`                |
| quality / KPR / citation + key points | `deepresearchgym`, `deepresearch_benchmarking` | `eval_quality_async.py`                     |
| `gpt_researcher` agent (not a judge)  | `gpt-researcher`, `gpt_researcher`             | `gpt_researcher/utils/trajectory_logger.py` |

Two directory names are accepted for the Gym judges because forks rename the repo; the marker file is what actually decides whether a candidate is valid.

Upstreams: [Ayanami0730/deep_research_bench](https://github.com/Ayanami0730/deep_research_bench) and [cxcscmu/deepresearch_benchmarking](https://github.com/cxcscmu/deepresearch_benchmarking). The `gpt_researcher` row is the agent under test, not a judge: [WilliamOdinson/gpt-researcher](https://github.com/WilliamOdinson/gpt-researcher). Override with `GR_REPO_URL` or `ADR_GR_DIR`.

The harness writes reports in each repo's official format, invokes their scripts as subprocesses, and parses their result files back into scores.
