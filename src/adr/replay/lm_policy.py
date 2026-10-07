"""Small-LM expansion policy for ReplayGym (Stage B, GPU).

At each decision the LM reads what a live agent would know (question, the branch's
goal and findings so far, what other branches cover, budget used) and answers
YES (research deeper) or NO. pi(YES) = softmax over the two answer tokens.

  * GRPO: sample decisions, get replay rewards per group, policy gradient (same
    advantage as Stage A).
  * DPO: offline pairs "sub-tree A beats sub-tree B" from the pairwise memo. A
    sub-tree maps to exactly one decision sequence (actions_for_subtree), so the
    pair's log-probs are exact sums of per-decision log-probs.
"""

from __future__ import annotations

import torch

from adr.replay.rl import EpisodeState

PROMPT = """You control a deep-research agent's budget. Decide whether to spend more research on this branch.
Question: {question}
Decision: {what}
Branch goal: {goal}
Findings from this branch so far: {findings}
Other branches already researched: {others}
Research budget used so far: {budget:.0%}
Will researching this deeper improve the final report enough to justify the cost? Answer YES or NO.
Answer:"""


def decision_prompt(st: EpisodeState) -> str:
    level, group = st.queue[0]
    t = st.tree
    goal = t.nodes[group[0]].research_goal if level == 1 else t.nodes[t.nodes[group[0]].parent_node_id].research_goal
    what = {1: "start researching this branch", 2: "research this branch one level deeper", 3: "research this branch to full depth"}[level]
    if level == 1:
        findings = "(not researched yet)"
    else:
        branch = {n for n in st.researched if _root(t, n) == _root(t, group[0])}
        findings = " ".join(l for n in t.order if n in branch for l in t.nodes[n].learnings)[:1500] or "(none)"
    others = "; ".join(t.nodes[r].research_goal[:120] for r in t.roots if r in st.researched and r != _root(t, group[0])) or "(none)"
    budget = t.tokens(st.researched) / st.full_tokens
    return PROMPT.format(question=t.query[:500], what=what, goal=goal[:300], findings=findings, others=others[:800], budget=budget)


def _root(t, n):
    while t.nodes[n].parent_node_id in t.nodes:
        n = t.nodes[n].parent_node_id
    return n


def actions_for_subtree(st: EpisodeState, subtree: frozenset[str]) -> list[tuple[str, int]]:
    """Deterministic (prompt, action) sequence that realizes `subtree` (consumes st)."""
    out = []
    while not st.done:
        prompt = decision_prompt(st)
        _, group = st.queue[0]
        a = int(all(n in subtree for n in group))
        out.append((prompt, a))
        st.step(a)
    return out


class LMPolicy(torch.nn.Module):
    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B", lora_r: int = 16, device: str | None = None):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device)
        self.model = get_peft_model(base, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
        self.yes = self.tok.encode(" YES", add_special_tokens=False)[0]
        self.no = self.tok.encode(" NO", add_special_tokens=False)[0]
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

    def logp(self, prompts: list[str], actions: list[int], adapter: bool = True) -> torch.Tensor:
        """Per-decision log pi(action | prompt). adapter=False gives the frozen reference."""
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536).to(self.device)
        ctx = self.model.disable_adapter() if not adapter else _null()
        with ctx:
            logits = self.model(**enc).logits[:, -1, :]
        two = torch.log_softmax(torch.stack([logits[:, self.no], logits[:, self.yes]], dim=-1).float(), dim=-1)
        a = torch.as_tensor(actions, device=two.device)
        return two.gather(1, a[:, None]).squeeze(1)

    @torch.no_grad()
    def p_yes(self, prompt: str) -> float:
        return float(self.logp([prompt], [1]).exp())


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def dpo_loss(pol: LMPolicy, win: list[tuple[str, int]], lose: list[tuple[str, int]], beta: float = 0.1) -> torch.Tensor:
    def seq(pairs, adapter):
        if not pairs:
            return torch.zeros((), device=pol.device)
        p, a = zip(*pairs)
        return pol.logp(list(p), list(a), adapter=adapter).sum()

    with torch.no_grad():
        ref_w, ref_l = seq(win, False), seq(lose, False)
    margin = (seq(win, True) - ref_w) - (seq(lose, True) - ref_l)
    return -torch.nn.functional.logsigmoid(beta * margin)
