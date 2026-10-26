"""Live-only deterministic evidence-prefix bounding; frozen serialization is unchanged."""
import json

from pydantic import model_validator

from adr.mlp.compact_metadata import encode_metadata, RULE
from adr.mlp.embedding_limits import MAX_INPUT_TOKENS, token_count
from adr.synthetic.schema import BudgetState, ResearchSnapshot, serialize_context


class UnenforcedLiveBudget(BudgetState):
    """Same fields/serialization; permit actual live usage above nominal maxima."""

    @model_validator(mode="after")
    def bounds(self):
        return self


class UnenforcedLiveSnapshot(ResearchSnapshot):
    """Live-only budget type; inherited state validation remains active."""
    budget: UnenforcedLiveBudget


def embedding_context(snapshot, *, on_diagnostics=None):
    original = serialize_context(snapshot)
    original_size = len(original.encode('utf-8'))
    info = {'rule': RULE, 'bounded': False,
            'original_bytes': original_size, 'final_bytes': original_size,
            'limit_tokens': MAX_INPUT_TOKENS, 'tokenizer': 'cl100k_base'}
    data = snapshot.model_dump()
    bodies = [e['text'] for e in data['evidence']]
    data = encode_metadata(data)

    text_index = data['evidence_fields'].index('text')

    def render(lengths):
        for evidence, body, length in zip(data['evidence'], bodies, lengths):
            evidence[text_index] = body[:length]
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

    lengths = [0] * len(bodies)
    metadata = render(lengths)
    info.update(metadata_bytes=len(metadata.encode('utf-8')),
                evidence_items=len(bodies), original_tokens=None, metadata_tokens=None)
    try:
        info.update(original_tokens=token_count(original), metadata_tokens=token_count(metadata))
        info['bounding_required'] = info['original_tokens'] > MAX_INPUT_TOKENS
    finally:
        if on_diagnostics:
            on_diagnostics(dict(info))
    if info['original_tokens'] <= MAX_INPUT_TOKENS:
        info['final_tokens'] = info['original_tokens']
        return original, info
    if info['metadata_tokens'] > MAX_INPUT_TOKENS:
        raise ValueError('Live context metadata alone exceeds embedding token bound; history/identities cannot be preserved')
    # Deterministic common-prefix search; token counts need not be monotone.
    # Every accepted candidate is checked; maximal content is not guaranteed.
    low, high = 0, max(map(len, bodies), default=0)
    while low < high:
        mid = (low + high + 1) // 2
        if token_count(render([min(mid, len(b)) for b in bodies])) <= MAX_INPUT_TOKENS:
            low = mid
        else:
            high = mid - 1
    lengths = [min(low, len(b)) for b in bodies]
    # Spend remaining tokens one character at a time in original historical order.
    while True:
        added = False
        for i, body in enumerate(bodies):
            if lengths[i] < len(body):
                lengths[i] += 1
                if token_count(render(lengths)) > MAX_INPUT_TOKENS:
                    lengths[i] -= 1
                else:
                    added = True
        if not added:
            break
    bounded = render(lengths)
    info.update(bounded=True, final_bytes=len(bounded.encode('utf-8')), final_tokens=token_count(bounded),
                evidence_items=len(bodies), truncated_items=sum(n < len(b) for n, b in zip(lengths, bodies)),
                original_body_characters=sum(map(len, bodies)), retained_body_characters=sum(lengths))
    return bounded, info
