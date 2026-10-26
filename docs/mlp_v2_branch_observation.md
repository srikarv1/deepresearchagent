# Frozen branch observation contract

Version: `mlp-observation-v2-branches-cited-top3-v1`.
Implementation: `adr.mlp.branch_observation_v2`. The earlier findings-only 2K
prototype is preserved; it is not this contract. No live gate is enabled.

Both future synthetic and live consumers must call `construct` (or `model_inputs`).
Supply detached pre-retrieval source data: complete hierarchy, published results
in stable publication order, and available completed/executed queries in execution
order. A processed but unpublished result is not eligible. Freeze the source for
the entire batch. The constructor cannot infer publication timestamps from text;
the caller must enforce that boundary. The development replay verifies the boundary
against recorded processing/publication events.

Ranking document is newline-joined goal, retained findings, then follow-ups.
TF-IDF uses sklearn lowercase word tokens (default `(?u)\b\w\w+\b`), English
stopwords, (1,2) ngrams, sublinear TF, smooth IDF and L2 norm. Fit on eligible
branch documents only, transform the candidate, rank cosine scores descending;
publication order breaks ties. Empty vocabularies and unseen candidate terms give
zero scores, not errors. Select at most three. Citations do not affect ranking.

Exact embedded JSON (UTF-8, ensure_ascii=False, sorted keys, compact separators):

```json
{"branches":[{"cited_insights":{"published insight":"https://source.example/"},"findings":["Complete retained finding"],"followups":["Complete follow-up"],"goal":"Full research goal","relation":"parent","tree_depth":1}],"current_need":"Full current need","recent_queries":["Prior executed query"]}
```

All strings are preserved. History includes only the last six available executed
queries. Parent means direct parent; ancestor excludes direct parent; other means
neither. Internal IDs and scores occur in returned audit metadata only, never in
embedded text. No raw-page field, pending registry, or other result fields are copied.

Exact cl100k_base token count, using the verified offline tokenizer cache. Over
4096 tokens returns the complete unshortened text with `overflow=True` and
`required_bypass_decision='EXECUTE'`. Future live consumers MUST audit version,
selected IDs, scores, tokens and overflow, bypass embeddings and the MLP, and
execute. No sentences, cards or history are dropped to satisfy this bound.
This module deliberately does not activate a live policy.

`model_inputs` delegates the numerical vector to the existing `derive_features`
on the caller's COMPLETE pre-decision ResearchSnapshot, independent of selection.
All 15 names, order, formulas, overrun saturation and remaining-budget flooring
are unchanged. Use the existing unenforced-live snapshot type for budget overruns.
The replay validates semantic observations, not a fabricated reconstruction of
canonical live numerical evidence: raw-page trace records and the numerical
ledger are different representations. Unit tests establish complete-state feature
independence and exact equality with the existing feature implementation.

Embedding contract (no embeddings produced here): original question, this context,
candidate query, each frozen text-embedding-3-small/1536 dimensions; concatenate
with the existing 15 train-normalized numerical values for 4623 dimensions.

Read-only verification command:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m adr.development.replay_observation_v2 \
 runs/representation-study/20260926-223740-development-representation-001 \
 runs/representation-study-002/20260927-003826-development-representation-002 \
 runs/representation-study-003/20260927-114734-development-representation-003
```

Measured on 336 candidates: 112/run; median/p90/max tokens are
3250.5/3454.9/3566, 2840.5/3101.5/3238, 2894.5/3105.5/3126. Zero overflow.
Tests with local traces skip if private artifacts are absent. No API, researcher,
benchmark loader, embedding generation, or training is invoked.
