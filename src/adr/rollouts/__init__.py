"""PILOT trajectory corpus: randomized rollouts -> rewards -> BC pairs.

  driver    `adr rollouts`  runs (query x seed) rollouts as separate `adr run`
            subprocesses under GR_ORCHESTRATOR=random, resumable, optionally
            judging each finished run.
  reward    R(tau) from Eq. 14: quality (RACE), goal satisfaction hook,
            normalized token/latency cost, potential-based coverage shaping.
  bc_pairs  `adr bc-pairs`  replays every round of the top-return rollouts
            through the fork's serializer into prompt/completion rows.
"""
