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


def test_load_browsecomp_slice():
    rows = load_queries("browsecomp", limit=5)
    assert len(rows) == 5
    assert all(r.language == "en" for r in rows)
    assert all(r.text for r in rows)
    assert all(r.dataset == "browsecomp" for r in rows)
    assert rows[0].id == "1"


def test_load_browsecomp_decryption():
    rows = load_queries("browsecomp", query_ids=["1"])
    assert len(rows) == 1
    assert rows[0].topic == "Art"
    # Verify text is decrypted (readable English), not base64 ciphertext
    assert "African author" in rows[0].text
    assert rows[0].metadata["answer"] == "1988-96"
    # Raw row should still contain encrypted fields
    assert "problem" in rows[0].metadata["raw"]
    assert rows[0].metadata["raw"]["problem"] != rows[0].text


def test_load_browsecomp_full():
    rows = load_queries("browsecomp")
    assert len(rows) == 1266
