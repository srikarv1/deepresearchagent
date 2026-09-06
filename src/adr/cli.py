from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

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
    language: Optional[str] = typer.Option(None, "--language"),
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    rows = load_queries(dataset, language=language, limit=limit)
    table = Table(title=f"{dataset} ({len(rows)} queries)")
    table.add_column("id")
    table.add_column("lang")
    table.add_column("text")
    for row in rows:
        text = row.text if len(row.text) < 80 else row.text[:77] + "..."
        table.add_row(row.id, row.language, text)
    console.print(table)


@app.command("run")
def run_cmd(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a"),
    llm_provider: Optional[str] = typer.Option(None, "--llm"),
    search_backend: Optional[str] = typer.Option(None, "--search"),
    language: Optional[str] = typer.Option(None, "--language"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    run_name: Optional[str] = typer.Option(None, "--run-name"),
    official: Optional[str] = typer.Option(None, "--official", help="Comma-separated: deep_research_bench,deep_research_gym"),
) -> None:
    overrides: dict = {}
    if dataset:
        overrides.setdefault("dataset", {})["name"] = dataset
    if language:
        overrides.setdefault("dataset", {})["language"] = language
    if limit is not None:
        overrides.setdefault("dataset", {})["limit"] = limit
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


@app.command("score")
def score_cmd(
    report: Optional[Path] = typer.Option(None, "--report", "-r", help="File containing one report"),
    question: Optional[str] = typer.Option(None, "--question", "-q", help="The query text"),
    query_id: Optional[str] = typer.Option(None, "--query-id", help="Benchmark query id"),
    reports_dir: Optional[Path] = typer.Option(None, "--reports-dir", help="Folder of <id>.q / <id>.a files"),
    drb_jsonl: Optional[Path] = typer.Option(None, "--drb-jsonl", help="DRB raw file of {id,prompt,article} rows"),
    dataset: str = typer.Option("deep_research_gym", "--dataset", "-d"),
    official: Optional[str] = typer.Option(None, "--official", help="Benches to judge with; defaults to --dataset"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model"),
    run_dir: Optional[Path] = typer.Option(None, "--run-dir", help="Where to write artifacts"),
    local_only: bool = typer.Option(False, "--local-only", help="Skip the judges, cost metrics only"),
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
        console.print(f"[dim]query id[/dim] {resolved_id}  [dim]question[/dim] {resolved_question[:90]}")
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
    official: str = typer.Option("", "--official", help="Comma-separated benches, or empty for local metrics only"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
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


@app.command("doctor")
def doctor_cmd() -> None:
    """Report whether the official judge repos and API keys are usable."""
    import os

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

    for name, used_for in (
        ("OPENAI_API_KEY", "DRB judge (LLM_BACKEND=openai) + all Gym judges"),
        ("OPENROUTER_API_KEY", "DRB judge (LLM_BACKEND=openrouter, default)"),
        ("JINA_API_KEY", "DRB FACT scraping"),
        ("DEEPRESEARCHGYM_API_KEY", "Gym search backend"),
        ("TAVILY_API_KEY", "live web search"),
    ):
        present = bool(os.environ.get(name))
        table.add_row(name, "[green]set[/green]" if present else "[yellow]unset[/yellow]", used_for)

    console.print(table)


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
