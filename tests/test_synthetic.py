import copy
import json
import zipfile

import pytest
from pydantic import ValidationError

from adr.llm.mock import MockLLM
from adr.synthetic.generate import GenerationConfig, compile_examples, generate_dataset, parse_teacher, safe_config, smoke_response
from adr.synthetic.prompt import teacher_messages
from adr.synthetic.schema import FEATURE_NAMES, BudgetState, Candidate, FeatureVector, ResearchSnapshot, SyntheticExample, TeacherTrajectory, derive_features, serialize_context
from adr.synthetic.validate import import_heldout_zip, load_heldout, validate_examples

HELDOUT = [{"id": "heldout-1", "text": "Analyze historical ceramic glazing techniques and preservation chemistry."}]


def config():
    return GenerationConfig(questions=1, states_per_question=2, candidates_per_state=2)


def examples():
    return compile_examples(parse_teacher(smoke_response()), question_id="q1", config=config(),
                            timestamp="2026-09-22T00:00:00Z", provider="mock", model="fake")


def test_schema_and_derived_context():
    rows = examples()
    record = SyntheticExample.model_validate(rows[0])
    assert record.context_before == serialize_context(record.state)
    reversed_keys = dict(reversed(record.state.model_dump().items()))
    assert serialize_context(ResearchSnapshot.model_validate(reversed_keys)) == record.context_before
    assert "label" not in record.context_before and "reason" not in record.context_before
    assert record.numerical_features.names == list(FEATURE_NAMES)
    assert len(record.numerical_features.values) == 15
    rows[0]["context_before"] += "future result"
    with pytest.raises(ValidationError):
        SyntheticExample.model_validate(rows[0])


def test_feature_semantics():
    state = ResearchSnapshot.model_validate({"evidence": [
        {"id": "a", "url": "u", "title": "water", "text": "water " * 500},
        {"id": "b", "url": "v", "text": "soil", "retained": False}],
        "branches": [{"id": "b1", "goal": "need"}, {"id": "b2", "goal": "done", "status": "done"}],
        "budget": {"used_steps": 8, "used_tokens": 10000, "used_searches": 2, "used_reads": 3}})
    features = dict(zip(FEATURE_NAMES, derive_features("water soil", state).values))
    assert features == dict(n_evidence=2, n_retained=1, n_pruned=1, n_open_branches=1, n_branches=2,
                           mean_score=0, mean_chars=2000, query_term_coverage=0.5, unique_urls=1,
                           step_frac=0.25, token_frac=0.2, remaining_steps=24, remaining_searches=62,
                           remaining_reads=125, remaining_tokens=40000)
    with pytest.raises(ValidationError):
        FeatureVector(names=list(reversed(FEATURE_NAMES)), values=[0] * 15)
    with pytest.raises(ValidationError):
        FeatureVector(values=[0] * 14)
    rows = examples(); rows[0]["numerical_features"]["values"][0] = 9
    with pytest.raises(ValidationError):
        SyntheticExample.model_validate(rows[0])


@pytest.mark.parametrize("field", ["observation", "stats_after", "cost", "latency", "decision_type", "resulting_evidence"])
def test_leakage_fields_rejected(field):
    data = json.loads(smoke_response())
    data["states"][0]["state"][field] = "forbidden"
    with pytest.raises(ValidationError):
        TeacherTrajectory.model_validate(data)


def test_invalid_labels_and_empty_ids():
    data = json.loads(smoke_response())["states"][0]["candidates"][0]
    for changes in [{"label": "SKIP"}, {"label": "OTHER"}, {"candidate_id": "  "}, {"candidate_subquery": ""}]:
        with pytest.raises(ValidationError):
            Candidate.model_validate({**data, **changes})


def test_bounds_and_successive_states():
    for changes in [{"used_tokens": 50001}, {"used_steps": -1}, {"max_reads": 0}, {"used_reads": 1.5}]:
        with pytest.raises(ValidationError):
            BudgetState.model_validate(changes)
    data = json.loads(smoke_response())
    data["states"][1]["state"]["budget"]["max_tokens"] = 100000
    with pytest.raises(ValidationError):
        TeacherTrajectory.model_validate(data)
    data = json.loads(smoke_response())
    data["states"][0]["state"]["executed_subqueries"] = ["previous search"]
    with pytest.raises(ValidationError):
        TeacherTrajectory.model_validate(data)
    data = json.loads(smoke_response())
    data["states"][1]["state"]["evidence"] *= 2
    with pytest.raises(ValidationError):
        TeacherTrajectory.model_validate(data)


def test_summary_and_counterfactuals():
    rows = examples()
    summary = validate_examples(rows, HELDOUT, expected_states=2, expected_candidates=2)
    assert summary["valid"] and summary["questions"] == 1 and summary["trajectories"] == 1
    assert summary["labels"] == {"EXECUTE": 2, "SKIP": 2}
    assert summary["counterfactual_pairs"] == 1
    assert not validate_examples(rows[:-2], HELDOUT)["valid"]
    assert not validate_examples(rows, HELDOUT, expected_states=5)["valid"]
    rows[2]["provenance"]["counterfactual_pair_id"] = None
    assert not validate_examples(rows, HELDOUT)["valid"]


def test_grouping_and_duplicates():
    rows = examples()
    assert not validate_examples(rows + [copy.deepcopy(rows[0])], HELDOUT)["valid"]
    rows = examples(); rows[2]["question_id"] = "different"
    assert not validate_examples(rows, HELDOUT)["valid"]
    rows = examples(); rows[2]["provenance"]["state_index"] = 7
    assert not validate_examples(rows, HELDOUT)["valid"]


def test_heldout_exact_near_and_prompt_isolation():
    rows = examples()
    question = rows[0]["original_question"]
    for text in [question, question.upper(), question + " Discuss."]:
        assert not validate_examples(rows, [{"id": "reference", "text": text}])["valid"]
    prompt = json.dumps(teacher_messages(2, 2, "library planning"))
    assert HELDOUT[0]["text"] not in prompt


def test_zip_reads_only_queries_and_unions_subsets(tmp_path):
    source = tmp_path / "reference.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("run-a/queries/1.json", json.dumps({"id": "1", "text": "Question one"}))
        archive.writestr("run-b/queries/2.json", json.dumps({"id": "2", "text": "Question two"}))
        archive.writestr("run-a/reports/1.json", "THIS IS NOT VALID JSON")
    dest = tmp_path / "heldout.json"
    import_heldout_zip(source, dest)
    assert len(load_heldout(dest)) == 2
    assert "THIS IS NOT VALID JSON" not in dest.read_text()


@pytest.mark.asyncio
async def test_generation_retry_without_network():
    client = MockLLM(replies=["not JSON", smoke_response()])
    rows, attempts = await generate_dataset(config(), client=client, heldout=HELDOUT, timestamp="now")
    assert len(rows) == 4 and len(client.calls) == 2 and len(attempts) == 2
    assert validate_examples(rows, HELDOUT)["valid"]
    assert HELDOUT[0]["text"] not in str(client.calls)


@pytest.mark.asyncio
async def test_generation_rejects_heldout_without_echoing():
    client = MockLLM(replies=[smoke_response()])
    protected = [{"id": "heldout-secret-id-123", "text": json.loads(smoke_response())["original_question"]}]
    with pytest.raises(RuntimeError, match="failed after"):
        await generate_dataset(config(), client=client, heldout=protected, timestamp="now")
    assert len(client.calls) == 3
    assert protected[0]["id"] not in str(client.calls)
    assert protected[0]["text"] not in str(client.calls)


def test_config_secret_redaction():
    cfg = GenerationConfig(llm={"model": "fake", "api_key": "secret", "base_url": "https://user:secret@host"})
    assert "secret" not in json.dumps(safe_config(cfg))


def test_api_requires_explicit_opt_in(tmp_path, monkeypatch):
    from adr.synthetic import generate
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({"questions": HELDOUT}))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(json.dumps({"heldout_path": str(heldout), "llm": {"provider": "openai", "model": "fake"}}))
    def forbidden(*args, **kwargs):
        pytest.fail("API client must not be constructed without opt-in")
    monkeypatch.setattr(generate, "build_llm", forbidden)
    monkeypatch.setattr("sys.argv", ["generate", "--config", str(cfg), "--questions", "1"])
    with pytest.raises(SystemExit) as exc:
        generate.main()
    assert exc.value.code == 2


def test_state_evidence_must_not_be_rewritten():
    data = json.loads(smoke_response())
    data["states"][0]["state"]["evidence"] = copy.deepcopy(data["states"][1]["state"]["evidence"])
    data["states"][1]["state"]["evidence"][0]["text"] = "Future information inserted into earlier evidence"
    with pytest.raises(ValidationError, match="rewritten"):
        TeacherTrajectory.model_validate(data)


@pytest.mark.parametrize("change", ["remove", "id", "url", "title", "text"])
def test_prior_evidence_fields_are_immutable_even_after_pruning(change):
    data = json.loads(smoke_response())
    data["states"][0]["state"]["evidence"] = copy.deepcopy(data["states"][1]["state"]["evidence"])
    data["states"][0]["state"]["evidence"][0]["retained"] = False
    if change == "remove":
        data["states"][1]["state"]["evidence"] = []
    else:
        data["states"][1]["state"]["evidence"][0][change] = "changed"
    with pytest.raises(ValidationError, match="prior evidence removed or rewritten"):
        TeacherTrajectory.model_validate(data)


def test_retention_transitions_and_new_evidence_preserve_history():
    data = json.loads(smoke_response())
    data["states"][0]["state"]["evidence"] = copy.deepcopy(data["states"][1]["state"]["evidence"])
    data["states"][1]["state"]["evidence"][0]["retained"] = False
    data["states"][1]["state"]["evidence"].append({
        "id": "e2", "url": "https://example.org/new", "text": "New conflicting finding."})
    trajectory = TeacherTrajectory.model_validate(data)
    assert trajectory.states[1].state.evidence[0].retained is False
    assert trajectory.states[0].state.evidence[0].text == trajectory.states[1].state.evidence[0].text
    data["states"][0]["state"]["evidence"][0]["retained"] = False
    data["states"][1]["state"]["evidence"][0]["retained"] = True
    TeacherTrajectory.model_validate(data)  # Existing schema also permits re-retention.


@pytest.mark.asyncio
async def test_evidence_failure_retry_is_specific_and_preserves_original_prompt():
    data = json.loads(smoke_response())
    data["states"][0]["state"]["evidence"] = [{
        "id": "private-evidence-id", "url": "https://example.org/private",
        "text": "PRIVATE_RESPONSE_SENTINEL"}]
    client = MockLLM(replies=[json.dumps(data), smoke_response()])
    rows, attempts = await generate_dataset(config(), client=client, heldout=HELDOUT, timestamp="now")
    assert len(rows) == 4 and len(attempts) == 2
    assert client.calls[1][:-1] == client.calls[0]
    feedback = client.calls[1][-1].content
    assert "prior evidence removed or rewritten; change retention instead" in feedback
    assert "id, url, title, and text exactly" in feedback
    assert "retained=false" in feedback
    assert "PRIVATE_RESPONSE_SENTINEL" not in str(client.calls)
    assert "private-evidence-id" not in str(client.calls)
    assert HELDOUT[0]["text"] not in str(client.calls)


def test_retry_feedback_does_not_forward_unknown_error_details():
    from adr.synthetic.generate import retry_feedback
    for exc in [ValueError("held-out overlap: PRIVATE_REFERENCE"),
                RuntimeError("credential: PRIVATE_CREDENTIAL")]:
        assert "PRIVATE" not in retry_feedback(exc)
    with pytest.raises(ValidationError) as caught:
        TeacherTrajectory.model_validate({"original_question": "PRIVATE_INPUT", "states": []})
    assert "PRIVATE_INPUT" not in retry_feedback(caught.value)


def test_prompt_explains_cumulative_evidence_without_changing_v2_constraints():
    prompt = teacher_messages(5, 4, "topic")[1]["content"]
    assert "exactly 5" in prompt and "4 plausible candidate subqueries" in prompt
    for instruction in ["EVERY later state's", "ONLY retained may change",
                        "never rewrite its original url, title, or text/body",
                        "NEVER removal or rewriting", "TWO distinct linked counterfactual",
                        "independently plausible research actions", "imperfect research progression",
                        "all 15 numerical features"]:
        assert instruction in prompt


def test_counterfactual_requires_information_change():
    data = json.loads(smoke_response())
    prior = copy.deepcopy(data["states"][0]["state"])
    prior["budget"]["used_steps"] = 1
    data["states"][1]["state"] = prior
    rows = compile_examples(TeacherTrajectory.model_validate(data), question_id="q1", config=config(),
                            timestamp="now", provider="mock", model="fake")
    assert not validate_examples(rows, HELDOUT)["valid"]


@pytest.mark.parametrize("defect", ["one", "three", "reversed", "same_label", "same_state"])
@pytest.mark.asyncio
async def test_invalid_pair_retry_preserves_task_and_repairs_pair(defect):
    data = json.loads(smoke_response())
    early, late = [s["candidates"] for s in data["states"]]
    if defect == "one":
        late[0]["counterfactual_pair_id"] = None
    elif defect == "three":
        early[1]["counterfactual_pair_id"] = "pair1"
    elif defect == "reversed":
        early[0].update(label="SKIP", decision_category="premature")
        late[0].update(label="EXECUTE", decision_category="necessary_dependency")
    elif defect == "same_label":
        late[0].update(label="EXECUTE", decision_category="necessary_dependency")
    else:
        late[0]["counterfactual_pair_id"] = None
        early[1]["counterfactual_pair_id"] = "pair1"
    early[0]["reason"] = "PRIVATE_REJECTED_RESPONSE"
    proposed = compile_examples(TeacherTrajectory.model_validate(data), question_id="q1",
        config=config(), timestamp="now", provider="mock", model="fake")
    summary = validate_examples(proposed, HELDOUT, expected_states=2, expected_candidates=2)
    assert "invalid counterfactual pair: q1-trajectory/pair1" in summary["errors"]

    client = MockLLM(replies=[json.dumps(data), smoke_response()])
    rows, attempts = await generate_dataset(config(), client=client, heldout=HELDOUT, timestamp="now")
    assert len(rows) == 4 and len(attempts) == 2 and len(client.calls) == 2
    assert client.calls[1][:-1] == client.calls[0]
    feedback = client.calls[1][-1].content
    for instruction in ["invalid counterfactual pair", "exactly twice across the entire trajectory",
                        "EXECUTE in an earlier state", "SKIP in a later state",
                        "different chronological states", "Use null for unrelated candidates",
                        "any additional appearances", "audit every non-null pair ID",
                        "labels, and state indices"]:
        assert instruction in feedback
    assert "PRIVATE_REJECTED_RESPONSE" not in str(client.calls)
    assert HELDOUT[0]["text"] not in str(client.calls)
    assert validate_examples(rows, HELDOUT)["valid"]


def test_pair_feedback_is_allowlisted_without_error_payloads():
    from adr.synthetic.generate import retry_feedback
    prefix = "generated dataset failed local validation: "
    message = (prefix + "held-out overlap: PRIVATE_REFERENCE; "
               "invalid counterfactual pair: PRIVATE_TRAJECTORY/PRIVATE_PAIR; "
               "PRIVATE_CREDENTIAL")
    feedback = retry_feedback(ValueError(message))
    assert "exactly twice" in feedback
    assert "PRIVATE" not in feedback
    assert "exactly twice" not in retry_feedback(RuntimeError(message))
    assert "exactly twice" not in retry_feedback(ValueError("unrelated PRIVATE_ERROR"))
    assert "exactly twice" not in retry_feedback(ValueError(prefix + "held-out overlap: PRIVATE_REFERENCE"))


def test_nonfinite_features_and_empty_registry(tmp_path):
    with pytest.raises(ValidationError):
        FeatureVector(values=[float("nan")] * 15)
    path = tmp_path / "heldout.json"
    path.write_text('{"questions": []}')
    with pytest.raises(ValueError, match="nonempty"):
        load_heldout(path)


@pytest.mark.parametrize("credential", [None, "", "   ", "fake-test-key"])
def test_openai_generation_credential_guard(tmp_path, monkeypatch, capsys, credential):
    from adr.synthetic import generate

    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({"questions": HELDOUT}))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(json.dumps({
        "heldout_path": str(heldout), "output_dir": str(tmp_path / "output"),
        "states_per_question": 2, "candidates_per_state": 2,
        "llm": {"provider": "openai", "model": "fake"},
    }))
    if credential is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", credential)
    calls = []

    def fake_client(settings):
        calls.append(settings)
        return MockLLM(replies=[smoke_response()])

    monkeypatch.setattr(generate, "build_llm", fake_client)
    monkeypatch.setattr("sys.argv", ["generate", "--config", str(cfg), "--questions", "1", "--allow-api"])
    if not credential or not credential.strip():
        with pytest.raises(SystemExit) as exc:
            generate.main()
        assert exc.value.code == 2
        assert "OPENAI_API_KEY must be set and non-empty" in capsys.readouterr().err
        assert calls == []
        assert not (tmp_path / "output").exists()
    else:
        generate.main()
        assert len(calls) == 1
        assert credential not in capsys.readouterr().out
