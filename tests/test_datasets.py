from adr.datasets.loader import load_queries


def test_load_drb_english_slice():
    rows = load_queries("deep_research_bench", language="en", limit=3)
    assert len(rows) == 3
    assert all(r.language == "en" for r in rows)
    assert all(r.text for r in rows)
    assert rows[0].id == "51"


def test_load_gym_by_id():
    rows = load_queries("deep_research_gym", query_ids=["923549"])
    assert len(rows) == 1
    assert "chip shortage" in rows[0].text


def test_load_browsecomp_plus_slice():
    rows = load_queries("browsecomp_plus", limit=5)
    assert len(rows) == 5
    assert all(r.language == "en" for r in rows)
    assert all(r.text for r in rows)
    assert all(r.dataset == "browsecomp_plus" for r in rows)
    # Original BrowseComp ids, sparse: 830 of 1266 survived corpus verification.
    assert [r.id for r in rows] == ["1", "3", "5", "6", "7"]


def test_load_browsecomp_plus_decryption():
    rows = load_queries("browsecomp_plus", query_ids=["1"])
    assert len(rows) == 1
    row = rows[0]
    # Same query text as BrowseComp id 1; decrypted, not base64 ciphertext.
    assert "African author" in row.text
    assert row.metadata["answer"] == "1988-96"
    assert row.metadata["raw"]["query"] != row.text

    evidence = row.metadata["evidence_docs"]
    gold = row.metadata["gold_docs"]
    # Matches topics-qrels/qrel_evidence.txt and qrel_golds.txt upstream.
    assert sorted(d["docid"] for d in evidence) == ["2882", "62014", "68543", "74874"]
    assert sorted(d["docid"] for d in gold) == ["62014", "68543"]
    assert {d["docid"] for d in gold} <= {d["docid"] for d in evidence}
    assert all(d["url"].startswith("http") for d in evidence)


def test_load_browsecomp_plus_full():
    rows = load_queries("browsecomp_plus")
    assert len(rows) == 830
    assert len({r.id for r in rows}) == 830
    assert all(r.metadata["evidence_docs"] for r in rows)
