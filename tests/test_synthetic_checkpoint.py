import json

import pytest

from adr.llm.mock import MockLLM
from adr.synthetic import generate


def distinct_response():
    data = json.loads(generate.smoke_response())
    data["original_question"] = "Evaluate satellite orbital debris monitoring, international liability, and collision avoidance governance."
    return json.dumps(data)


@pytest.fixture
def run_setup(tmp_path, monkeypatch):
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({"questions": [{"id": "ref", "text": "Historical ceramic glazing preservation techniques"}]}))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"questions": 2, "states_per_question": 2,
        "candidates_per_state": 2, "topics": ["topic one", "topic two"],
        "heldout_path": str(heldout), "output_dir": str(tmp_path / "output"),
        "llm": {"provider": "mock", "model": "test"}}))

    def run(client, resume=None):
        monkeypatch.setattr(generate, "build_llm", lambda _: client)
        argv = ["generate", "--config", str(cfg)]
        if resume:
            argv += ["--resume", str(resume)]
        monkeypatch.setattr("sys.argv", argv)
        generate.main()

    return run, cfg, heldout, tmp_path / "output" / "v1"


def fail_second(run_setup):
    run, _, _, root = run_setup
    client = MockLLM(model="test", replies=[generate.smoke_response(), "bad", "bad", "bad"])
    with pytest.raises(RuntimeError, match="Question 2 failed after 3 attempts"):
        run(client)
    assert len(client.calls) == 4
    output, = root.iterdir()
    return output


def test_failure_preserves_question_and_resume_checks_duplicates(run_setup):
    run, _, _, _ = run_setup
    output = fail_second(run_setup)
    original = (output / "examples.jsonl").read_text()
    assert len(original.splitlines()) == 4
    assert {json.loads(line)["question_id"] for line in original.splitlines()} == {"v1-q001"}
    before = json.loads((output / "manifest.json").read_text())
    assert before["summary"]["questions"] == 1
    assert len(before["attempts"]) == 4
    # First resumed answer repeats q001: must be rejected using reloaded rows.
    client = MockLLM(model="test", replies=[generate.smoke_response(), distinct_response()])
    run(client, output)
    assert len(client.calls) == 2
    assert all("topic two" in str(call) and "topic one" not in str(call) for call in client.calls)
    final = (output / "examples.jsonl").read_text()
    assert final.startswith(original)
    assert len(final.splitlines()) == 8
    rows = [json.loads(line) for line in final.splitlines()]
    assert [r["question_id"] for r in rows] == ["v1-q001"] * 4 + ["v1-q002"] * 4
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest.keys() == before.keys()
    assert manifest["generated_at"] == before["generated_at"]
    assert manifest["summary"]["valid"] and manifest["summary"]["questions"] == 2
    assert len(manifest["attempts"]) == 6
    # A completed run must not even construct a teacher client.
    run(None, output)
    assert (output / "examples.jsonl").read_text() == final


def test_checkpoint_is_written_before_next_question(run_setup, monkeypatch):
    run, _, _, root = run_setup
    client = MockLLM(model="test", replies=[generate.smoke_response(), distinct_response()])
    complete = client.complete

    async def inspect_then_complete(*args, **kwargs):
        if client.calls:
            output, = root.iterdir()
            checkpoint = json.loads((output / "checkpoint.json").read_text())
            assert len(checkpoint["examples"]) == 4
            assert checkpoint["manifest"]["summary"]["questions"] == 1
            assert len((output / "examples.jsonl").read_text().splitlines()) == 4
        return await complete(*args, **kwargs)

    monkeypatch.setattr(client, "complete", inspect_then_complete)
    run(client)


@pytest.mark.parametrize("change", ["config", "heldout", "prompt", "rows", "ids"])
def test_resume_rejects_incompatible_or_corrupt_checkpoint(run_setup, change):
    run, cfg, heldout, _ = run_setup
    output = fail_second(run_setup)
    if change == "config":
        data = json.loads(cfg.read_text())
        data["topics"].reverse()
        cfg.write_text(json.dumps(data))
    elif change == "heldout":
        heldout.write_text(heldout.read_text() + "\n")
    else:
        path = output / "checkpoint.json"
        data = json.loads(path.read_text())
        if change == "prompt":
            data["manifest"]["prompt_version"] = "wrong"
        elif change == "rows":
            data["examples"].pop()
        else:
            for row in data["examples"]:
                row["question_id"] = "v1-q002"
                row["trajectory_id"] = "v1-q002-trajectory"
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        run(None, output)


def test_resume_recovers_canonical_files_from_atomic_checkpoint(run_setup, monkeypatch):
    run, _, _, root = run_setup
    original_write = generate.atomic_write

    def interrupted_write(path, text):
        if path.name == "examples.jsonl" and text:
            raise OSError("simulated interrupted canonical write")
        original_write(path, text)

    monkeypatch.setattr(generate, "atomic_write", interrupted_write)
    client = MockLLM(model="test", replies=[generate.smoke_response()])
    with pytest.raises(OSError, match="simulated interrupted"):
        run(client)
    assert len(client.calls) == 1  # Persistence errors must not retry generation.
    output, = root.iterdir()
    assert len(json.loads((output / "checkpoint.json").read_text())["examples"]) == 4
    monkeypatch.setattr(generate, "atomic_write", original_write)
    resumed = MockLLM(model="test", replies=[distinct_response()])
    run(resumed, output)
    assert len(resumed.calls) == 1
    assert len((output / "examples.jsonl").read_text().splitlines()) == 8


def test_atomic_replace_failure_keeps_previous_checkpoint(tmp_path, monkeypatch):
    from pathlib import Path
    path = tmp_path / "checkpoint.json"
    generate.atomic_write(path, '{"old": true}')

    def fail_replace(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError):
        generate.atomic_write(path, '{"new": true}')
    assert json.loads(path.read_text()) == {"old": True}
