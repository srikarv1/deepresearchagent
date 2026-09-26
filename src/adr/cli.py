from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from adr.datasets.loader import load_queries
from adr.eval.compare import compare_summaries
from adr.eval.importers import (
    resolve_question,
    trajectories_from_drb_jsonl,
    trajectories_from_gym_folder,
    trajectory_from_pair,
    write_trajectories,
)
from adr.eval.repos import (
    find_bcp_dense_index,
    find_bcp_index,
    find_deep_research_bench,
    find_deep_research_gym,
    find_gpt_researcher,
    find_key_points,
)
from adr.runner.config import load_config
from adr.runner.experiment import evaluate_run_dir, run_experiment

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _benches(value: str | None) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def _print_summary(summary: dict[str, Any]) -> None:
    table = Table(title="Local cost metrics", show_header=True)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key in sorted(k for k in summary if k.startswith("mean_") or k.startswith("n_")):
        table.add_row(key, str(summary[key]))
    console.print(table)

    scores = summary.get("scores") or {}
    if scores:
        judged = Table(title="Official judge scores", show_header=True)
        judged.add_column("metric")
        judged.add_column("value", justify="right")
        for key in sorted(scores):
            judged.add_row(key, f"{scores[key]:.4f}")
        console.print(judged)

    for bench, block in (summary.get("official") or {}).items():
        if isinstance(block, dict) and block.get("reason"):
            console.print(f"[yellow]{bench}:[/yellow] {block['reason']}")


@app.command("queries")
def queries_cmd(
    dataset: str = typer.Option("deep_research_gym", "--dataset", "-d"),
    language: str | None = typer.Option(None, "--language"),
    limit: int | None = typer.Option(None, "--limit"),
    ids: str | None = typer.Option(None, "--ids", help="Comma-separated query ids"),
    split: str | None = typer.Option(
        None, "--split", help="Pinned hold-out: train | val | test (BrowseComp-Plus)"
    ),
    show_answer: bool = typer.Option(
        False, "--show-answer", help="Show metadata answer (BrowseComp-Plus)"
    ),
) -> None:
    """List and inspect benchmark queries."""
    query_ids = [x.strip() for x in ids.split(",") if x.strip()] if ids else None
    rows = load_queries(
        dataset, language=language, limit=limit, query_ids=query_ids, split=split
    )
    table = Table(title=f"{dataset} ({len(rows)} queries)", show_lines=True)
    table.add_column("id", no_wrap=True)
    table.add_column("lang", no_wrap=True)
    table.add_column("text")
    if show_answer:
        table.add_column("answer")
    for row in rows:
        cols = [row.id, row.language, row.text]
        if show_answer:
            cols.append(str(row.metadata.get("answer", "")))
        table.add_row(*cols)
    console.print(table)


@app.command("run")
def run_cmd(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset: str | None = typer.Option(None, "--dataset", "-d"),
    agent: str | None = typer.Option(None, "--agent", "-a"),
    llm_provider: str | None = typer.Option(None, "--llm"),
    search_backend: str | None = typer.Option(None, "--search"),
    language: str | None = typer.Option(None, "--language"),
    limit: int | None = typer.Option(None, "--limit"),
    split: str | None = typer.Option(
        None, "--split", help="Pinned hold-out: train | val | test (BrowseComp-Plus)"
    ),
    run_name: str | None = typer.Option(None, "--run-name"),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Exact run directory name (default: <timestamp>-<run_name>)"
    ),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Parent of the run directory"),
    query_ids: str | None = typer.Option(None, "--query-ids", help="Comma-separated query ids"),
    official: str | None = typer.Option(
        None, "--official", help="Comma-separated: deep_research_bench,deep_research_gym,browsecomp_plus"
    ),
) -> None:
    overrides: dict = {}
    if dataset:
        overrides.setdefault("dataset", {})["name"] = dataset
    if language:
        overrides.setdefault("dataset", {})["language"] = language
    if limit is not None:
        overrides.setdefault("dataset", {})["limit"] = limit
    if split:
        overrides.setdefault("dataset", {})["split"] = split
    if query_ids:
        overrides.setdefault("dataset", {})["query_ids"] = _benches(query_ids)
    if run_id:
        overrides["run_id"] = run_id
    if output_dir is not None:
        overrides["output_dir"] = str(output_dir)
    if agent:
        overrides.setdefault("agent", {})["name"] = agent
    if llm_provider:
        overrides.setdefault("llm", {})["provider"] = llm_provider
    if search_backend:
        overrides.setdefault("search", {})["backend"] = search_backend
    if run_name:
        overrides["run_name"] = run_name
    if official:
        overrides.setdefault("eval", {})["official_benches"] = _benches(official)
    cfg = load_config(config, overrides)
    manifest = run_experiment(cfg)
    console.print(f"[green]Run written to[/green] {manifest.run_dir}")
    summary_path = manifest.run_dir / "metrics" / "summary.json"
    if summary_path.exists():
        _print_summary(json.loads(summary_path.read_text(encoding="utf-8")))


@app.command("rollouts")
def rollouts_cmd(
    config: Path = typer.Option(
        Path("configs/rollouts_drb_random.yaml"), "--config", "-c",
        help="adr run config for one rollout (agent, dataset, budget)",
    ),
    out: Path = typer.Option(Path("runs/rollouts"), "--out", help="Corpus directory"),
    n_seeds: int = typer.Option(10, "--n-seeds", help="Rollouts per query"),
    seed_base: int = typer.Option(0, "--seed-base", help="First GR_ORCH_SEED"),
    parallel: int = typer.Option(1, "--parallel", "-j", help="Concurrent rollout subprocesses"),
    policy: str = typer.Option("random", "--policy", help="GR_ORCHESTRATOR value"),
    query_ids: str | None = typer.Option(None, "--query-ids", help="Comma-separated query ids"),
    split: str | None = typer.Option(None, "--split", help="train | val | test (pinned splits)"),
    limit: int | None = typer.Option(None, "--limit", help="Max queries"),
    evaluate: str | None = typer.Option(
        None, "--evaluate", help="Judge each finished rollout, e.g. deep_research_bench (serialized)"
    ),
    env: list[str] = typer.Option([], "--env", help="Extra KEY=VALUE for the subprocess (repeatable)"),
    timeout_s: float | None = typer.Option(None, "--timeout-s", help="Per-rollout timeout"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan without running"),
) -> None:
    """Generate randomized orchestration rollouts (PILOT §4.5), resumable."""
    from adr.rollouts.driver import run_rollouts

    extra_env: dict[str, str] = {}
    for item in env:
        if "=" not in item:
            raise typer.BadParameter(f"--env expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        extra_env[k.strip()] = v

    counts: dict[str, int] = {}

    def _on(res) -> None:  # noqa: ANN001
        counts[res.status] = counts.get(res.status, 0) + 1
        tag = {"done": "green", "skipped": "dim", "failed": "red", "dry-run": "cyan"}.get(res.status, "white")
        extra = f" eval={'ok' if res.evaluated else 'no'}" if evaluate and res.status != "dry-run" else ""
        console.print(f"[{tag}]{res.status:8}[/{tag}] {res.run_id}" + (f"  {res.wall_s}s" if res.wall_s else "") + extra)
        if res.status == "dry-run":
            console.print("  " + " ".join(res.extra.get("cmd", [])), style="dim")

    results = run_rollouts(
        config=config, out=out, n_seeds=n_seeds, seed_base=seed_base, parallel=parallel,
        policy=policy, query_ids=_benches(query_ids) or None, split=split, limit=limit,
        evaluate=evaluate, extra_env=extra_env, dry_run=dry_run, timeout_s=timeout_s, on_result=_on,
    )
    console.print(f"[bold]{len(results)} rollouts[/bold] -> {out}  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


@app.command("bc-pairs")
def bc_pairs_cmd(
    rollouts: list[Path] = typer.Argument(..., help="Rollout corpus dir(s) or individual run dirs"),
    out: Path = typer.Option(Path("data/bc/bc_pairs.jsonl"), "--out"),
    top_frac: float = typer.Option(0.1, "--top-frac", help="Fraction of rollouts (by return) to clone"),
    min_per_query: int = typer.Option(1, "--min-per-query"),
    max_per_query: int | None = typer.Option(None, "--max-per-query"),
    require_quality: bool = typer.Option(True, "--require-quality/--allow-unscored",
                                         help="Drop rollouts without a RACE score"),
    policy: str | None = typer.Option("random", "--policy", help="Only clone rollouts from this policy ('' = any)"),
    reward_config: Path | None = typer.Option(None, "--reward-config", help="YAML/JSON with RewardConfig fields"),
    goal_file: Path | None = typer.Option(None, "--goal-file", help="JSON {run_id|query_id: G} for alpha_g"),
    snippet_chars: int | None = typer.Option(None, "--snippet-chars"),
    no_final_terminate: bool = typer.Option(False, "--no-final-terminate",
                                            help="Do not relabel the last round as TERMINATE"),
) -> None:
    """Score rollouts with R(tau) and write the top-return rounds as BC pairs."""
    import yaml

    from adr.rollouts.bc_pairs import build_bc_pairs
    from adr.rollouts.fork import ForkModuleMissing, load_fork_module
    from adr.rollouts.reward import RewardConfig

    try:
        load_fork_module("serialize")
    except ForkModuleMissing as exc:
        raise typer.BadParameter(str(exc))

    cfg = RewardConfig()
    if reward_config:
        cfg = RewardConfig.from_dict(yaml.safe_load(reward_config.read_text(encoding="utf-8")) or {})

    stats = build_bc_pairs(
        [Path(p) for p in rollouts], out, reward_cfg=cfg, top_frac=top_frac,
        min_per_query=min_per_query, max_per_query=max_per_query, require_quality=require_quality,
        require_policy=policy or None, final_round_terminate=not no_final_terminate,
        snippet_chars=snippet_chars, goal_file=goal_file,
    )
    table = Table(title=f"bc_pairs -> {out}")
    table.add_column("stat")
    table.add_column("value", justify="right")
    for k, v in stats.as_dict().items():
        table.add_row(k, json.dumps(v) if isinstance(v, dict) else str(v))
    console.print(table)


@app.command("score")
def score_cmd(
    report: Path | None = typer.Option(None, "--report", "-r", help="File containing one report"),
    question: str | None = typer.Option(None, "--question", "-q", help="The query text"),
    query_id: str | None = typer.Option(None, "--query-id", help="Benchmark query id"),
    reports_dir: Path | None = typer.Option(
        None, "--reports-dir", help="Folder of <id>.q / <id>.a files"
    ),
    drb_jsonl: Path | None = typer.Option(
        None, "--drb-jsonl", help="DRB raw file of {id,prompt,article} rows"
    ),
    dataset: str = typer.Option("deep_research_gym", "--dataset", "-d"),
    official: str | None = typer.Option(
        None, "--official", help="Benches to judge with; defaults to --dataset"
    ),
    judge_model: str | None = typer.Option(None, "--judge-model"),
    run_dir: Path | None = typer.Option(None, "--run-dir", help="Where to write artifacts"),
    local_only: bool = typer.Option(
        False, "--local-only", help="Skip the judges, cost metrics only"
    ),
) -> None:
    """Score reports the harness did not produce, including a single hand-written one."""
    if reports_dir:
        trajectories = trajectories_from_gym_folder(reports_dir)
        if not trajectories:
            raise typer.BadParameter(f"No <id>.q / <id>.a pairs found in {reports_dir}")
    elif drb_jsonl:
        trajectories = trajectories_from_drb_jsonl(drb_jsonl)
        dataset = "deep_research_bench"
    elif report:
        if not report.exists():
            raise typer.BadParameter(f"Report not found: {report}")
        resolved_id, resolved_question = resolve_question(
            dataset=dataset, query_id=query_id, question=question
        )
        trajectories = [
            trajectory_from_pair(
                question=resolved_question,
                report=report.read_text(encoding="utf-8"),
                query_id=resolved_id,
                dataset=dataset,
            )
        ]
        console.print(
            f"[dim]query id[/dim] {resolved_id}  [dim]question[/dim] {resolved_question[:90]}"
        )
    else:
        raise typer.BadParameter("Pass one of --report, --reports-dir, or --drb-jsonl")

    target = run_dir or Path("runs") / f"score-{dataset}-{len(trajectories)}q"
    write_trajectories(target, trajectories)
    console.print(f"[dim]artifacts[/dim] {target}")

    benches = [] if local_only else (_benches(official) or [dataset])
    overrides: dict = {"agent": {"name": "imported"}}
    if judge_model:
        overrides["eval"] = {"deep_research_gym": {"judge_model": judge_model}}

    summary = evaluate_run_dir(target, official_benches=benches, config=overrides)
    _print_summary(summary)


@app.command("evaluate")
def evaluate_cmd(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    official: str = typer.Option(
        "", "--official", help="Comma-separated benches, or empty for local metrics only (deep_research_bench,deep_research_gym,browsecomp_plus)"
    ),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    cfg = load_config(config) if config else {}
    summary = evaluate_run_dir(run_dir, official_benches=_benches(official), config=cfg)
    _print_summary(summary)


@app.command("compare")
def compare_cmd(
    left: Path = typer.Argument(..., exists=True, help="Baseline run"),
    right: Path = typer.Argument(..., exists=True, help="New run"),
) -> None:
    left_summary = left / "metrics" / "summary.json" if left.is_dir() else left
    right_summary = right / "metrics" / "summary.json" if right.is_dir() else right
    out = compare_summaries(left_summary, right_summary)

    sections = (
        ("Quality (higher is better)", "quality_deltas"),
        ("Cost (lower is better)", "cost_deltas"),
        ("Structure (context and report shape)", "structure_deltas"),
    )
    for title, key in sections:
        rows = out[key]
        if not rows:
            continue
        table = Table(title=title)
        table.add_column("metric")
        table.add_column("baseline", justify="right")
        table.add_column("new", justify="right")
        table.add_column("delta", justify="right")
        table.add_column("%", justify="right")
        for metric in sorted(rows):
            pct = out["percent_change"].get(metric)
            table.add_row(
                metric,
                f"{out['left_values'][metric]:.4f}",
                f"{out['right_values'][metric]:.4f}",
                f"{rows[metric]:+.4f}",
                "-" if pct is None else f"{pct:+.2f}%",
            )
        console.print(table)


def _drb_judge_backend() -> tuple[str, str]:
    """Effective DRB judge backend and where it came from: the LLM_BACKEND
    variable, else judge.backend of configs/eval/deep_research_bench.yaml,
    else the upstream default."""
    import yaml

    from adr.runner.config import ROOT

    env_backend = os.environ.get("LLM_BACKEND", "").strip().lower()
    if env_backend:
        return env_backend, "LLM_BACKEND"
    path = ROOT / "configs" / "eval" / "deep_research_bench.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        backend = str(((data.get("judge") or {}).get("backend")) or "").strip().lower()
    except (OSError, yaml.YAMLError):
        backend = ""
    if backend:
        return backend, "configs/eval/deep_research_bench.yaml"
    return "openrouter", "upstream default"


@app.command("doctor")
def doctor_cmd() -> None:
    """Report whether the official judge repos and API keys are usable."""
    table = Table(title="Evaluation prerequisites")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")

    drb = find_deep_research_bench()
    table.add_row(
        "DeepResearch Bench repo",
        "[green]found[/green]" if drb.ok else "[red]missing[/red]",
        str(drb.path or drb.reason),
    )
    gym = find_deep_research_gym()
    table.add_row(
        "DeepResearchGym repo",
        "[green]found[/green]" if gym.ok else "[red]missing[/red]",
        str(gym.path or gym.reason),
    )
    if gym.ok:
        key_points = find_key_points(gym.path)
        table.add_row(
            "Gym key points",
            "[green]found[/green]" if key_points else "[yellow]missing[/yellow]",
            str(key_points or "needed for key-point recall only"),
        )

    gr = find_gpt_researcher()
    table.add_row(
        "gpt-researcher fork",
        "[green]found[/green]" if gr.ok else "[yellow]missing[/yellow]",
        str(gr.path or gr.reason),
    )
    if gr.ok:
        try:
            import importlib.util

            spec = importlib.util.find_spec("gpt_researcher")
            importable = spec is not None and spec.origin and str(gr.path) in str(spec.origin)
        except Exception:
            importable = False
        table.add_row(
            "gpt_researcher importable from fork",
            "[green]yes[/green]" if importable else "[yellow]no[/yellow]",
            "" if importable else f"pip install -e {gr.path}",
        )

    bcp = find_bcp_index()
    table.add_row(
        "BrowseComp-Plus BM25 index",
        "[green]found[/green]" if bcp.ok else "[yellow]missing[/yellow]",
        str(bcp.path or bcp.reason),
    )
    dense = find_bcp_dense_index()
    table.add_row(
        "BrowseComp-Plus dense index (Qwen3-Embedding-0.6B)",
        "[green]found[/green]" if dense.ok else "[yellow]missing[/yellow]",
        str(dense.path or dense.reason),
    )
    try:
        import importlib.util

        has_pyserini = importlib.util.find_spec("pyserini") is not None
        has_numpy = importlib.util.find_spec("numpy") is not None
    except Exception:
        has_pyserini = False
        has_numpy = False
    import shutil

    java = shutil.which("java")
    table.add_row(
        "pyserini + java (BrowseComp-Plus retriever)",
        "[green]yes[/green]" if (has_pyserini and java) else "[yellow]no[/yellow]",
        f"java={java or 'missing'}; " + ("" if has_pyserini else r"pip install -e '.\[bcp]'"),
    )
    table.add_row(
        "numpy (dense ranking)",
        "[green]yes[/green]" if has_numpy else "[yellow]no[/yellow]",
        "" if has_numpy else r"pip install -e '.\[bcp]'",
    )
    ollama_host = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
    table.add_row(
        "Ollama (dense query encoder)",
        "[green]up[/green]" if _ollama_reachable(ollama_host) else "[yellow]down[/yellow]",
        ollama_host,
    )

    drb_backend, drb_source = _drb_judge_backend()
    for name, used_for in (
        ("OPENAI_API_KEY", "all Gym judges" + (f"; DRB judge (backend openai, {drb_source})" if drb_backend == "openai" else "")),
        ("OPENROUTER_API_KEY", f"DRB judge (backend openrouter, {drb_source})" if drb_backend == "openrouter" else "DRB judge only with backend openrouter"),
        ("JINA_API_KEY", "DRB FACT scraping"),
        ("DEEPRESEARCHGYM_API_KEY", "Gym search backend"),
        ("TAVILY_API_KEY", "live web search"),
    ):
        present = bool(os.environ.get(name))
        table.add_row(name, "[green]set[/green]" if present else "[yellow]unset[/yellow]", used_for)

    console.print(table)


@app.command("serve-retriever")
def serve_retriever_cmd(
    index: str | None = typer.Option(
        None,
        "--index",
        help="Lucene index dir (default: ADR_BCP_INDEX or third_party/bcp_indexes/bm25)",
    ),
    retriever: str | None = typer.Option(
        None,
        "--retriever",
        help="bm25 (default) or dense (Qwen3-Embedding shards + Ollama query encoder)",
    ),
    dense_index: str | None = typer.Option(
        None,
        "--dense-index",
        help="Tevatron pickle/npz dir (default: ADR_BCP_DENSE or third_party/bcp_indexes/qwen3-embedding-8b)",
    ),
    embed_model: str | None = typer.Option(
        None,
        "--embed-model",
        help="Ollama embedding tag (default: ADR_BCP_EMBED_MODEL or qwen3-embedding:8b)",
    ),
    embed_base_url: str | None = typer.Option(
        None,
        "--embed-base-url",
        help="Ollama base URL (default: OLLAMA_HOST or http://127.0.0.1:11434)",
    ),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8321, "--port"),
    k: int = typer.Option(5, "--k", help="Default hits per query when the client sends no k"),
    max_chars: int = typer.Option(
        16000, "--max-chars", help="Truncate raw_content per doc (0 = full document)"
    ),
) -> None:
    """Serve the BrowseComp-Plus corpus for gpt-researcher's RETRIEVER=custom."""
    from adr.tools.bcp_server import serve

    chosen = (retriever or os.environ.get("ADR_BCP_RETRIEVER") or "bm25").lower()
    serve(
        index_path=index,
        host=host,
        port=port,
        default_k=k,
        max_chars=max_chars,
        retriever=chosen,
        dense_path=dense_index,
        embed_model=embed_model,
        embed_base_url=embed_base_url,
    )


def _ollama_reachable(base_url: str) -> bool:
    from urllib.error import URLError
    from urllib.request import urlopen

    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urlopen(url, timeout=1.5) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except (URLError, TimeoutError, OSError):
        return False


@app.command("bootstrap")
def bootstrap_cmd() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_third_party.sh"
    raise typer.Exit(code=_run_script(script))


def _run_script(script: Path) -> int:
    import subprocess

    if not script.exists():
        console.print(f"[red]Missing {script}[/red]")
        return 1
    return subprocess.call(["bash", str(script)])


if __name__ == "__main__":
    app()
