from pathlib import Path
import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    "summarize_bcp_table",
    Path(__file__).resolve().parents[1] / "scripts" / "summarize_bcp_table.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_mod)
collect_rows = _mod.collect_rows
to_markdown = _mod.to_markdown
zip_trajectories = _mod.zip_trajectories


def test_skeleton_has_ten_rows_for_both_librarians(tmp_path: Path):
    rows = collect_rows(tmp_path)
    assert [(r["retriever"], r["policy"]) for r in rows] == [
        ("bm25", "none"),
        ("bm25", "topk"),
        ("bm25", "extractive"),
        ("bm25", "llmlingua"),
        ("bm25", "prompted"),
        ("dense", "none"),
        ("dense", "topk"),
        ("dense", "extractive"),
        ("dense", "llmlingua"),
        ("dense", "prompted"),
    ]
    md = to_markdown(rows)
    assert "depth=3, breadth=2" in md
    assert md.count("| bm25 |") == 5
    assert md.count("| dense |") == 5


def test_zip_trajectories(tmp_path: Path):
    run = tmp_path / "20260101-bcp-overnight-bm25-none"
    (run / "trajectories").mkdir(parents=True)
    (run / "trajectories" / "1.json").write_text("{}\n")
    dest = tmp_path / "out.zip"
    assert zip_trajectories(tmp_path, dest) == 1
    assert dest.exists() and dest.stat().st_size > 0
