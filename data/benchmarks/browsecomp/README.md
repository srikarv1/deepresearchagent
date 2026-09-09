# BrowseComp test set

`browse_comp_test_set.csv` — the public BrowseComp benchmark (OpenAI, 2025),
committed **encrypted**, exactly as downloaded.

- Source: https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv
- Downloaded: 2026-09-08
- SHA-256: `7b24471cd5b3eb2a46830a14802b5c029ea62f488ff75a0f88af7923d1454abf`
- Rows: 1,266. Columns: `problem`, `answer`, `problem_topic`, `canary`.

`problem` and `answer` are XOR-encrypted with `sha256(canary)` and
base64-encoded (the same scheme as `simple-evals/browsecomp_eval.py`).
`problem_topic` is plaintext. Decryption lives in
`src/adr/datasets/browsecomp.py`; the harness decrypts questions at load time
and answers only inside the judge. Do not commit decrypted answers anywhere.

Topic distribution: TV shows & movies 205 · Other 197 · Science & technology
173 · Art 127 · History 125 · Sports 123 · Music 116 · Video games 71 ·
Geography 70 · Politics 59.
