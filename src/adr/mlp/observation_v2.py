"""Shared, deterministic MLP #2 observation. No providers, training, or I/O."""
import json
import re
from pydantic import Field
from adr.synthetic.schema import StrictModel
from adr.mlp.embedding_limits import encoding, token_count

VERSION = 'mlp-observation-v2-findings'
MAX_TOKENS = 2048

class ResultCard(StrictModel):
    goal: str = Field(min_length=1)
    findings: list[str] = Field(default_factory=list)
    followups: list[str] = Field(default_factory=list)

class ObservationInput(StrictModel):
    current_need: str = Field(min_length=1)
    pending_needs: list[str] = Field(default_factory=list)
    prior_results: list[ResultCard] = Field(default_factory=list)
    recent_queries: list[str] = Field(default_factory=list)


def _cap(text, limit):
    tokens = encoding().encode_ordinary(text)
    if len(tokens) <= limit:
        return text
    # Decode without inventing replacement characters at a partial UTF-8 boundary.
    clipped = encoding().decode_bytes(tokens[:limit]).decode('utf-8', errors='ignore')
    while token_count(clipped) > limit:
        clipped = clipped[:-1]
    return clipped


def construct(source: ObservationInput):
    diagnostics = {'version': VERSION, 'omitted_pending_needs':max(0,len(source.pending_needs)-3),
                   'omitted_result_cards':max(0,len(source.prior_results)-4),
                   'omitted_recent_queries':max(0,len(source.recent_queries)-6),
                   'omitted_findings':0,'omitted_followups':0,'truncated_fields':0,'source_references_removed':0}
    def cap(text, limit):
        text, removed = re.subn(r'url-sha256:[0-9a-fA-F]+|https?://[^\s<>]+', '[source]', text)
        diagnostics['source_references_removed'] += removed
        result = _cap(text,limit)
        diagnostics['truncated_fields'] += result != text
        return result
    cards=[]
    for card in source.prior_results[-4:]:
        diagnostics['omitted_findings'] += max(0,len(card.findings)-3)
        diagnostics['omitted_followups'] += max(0,len(card.followups)-2)
        cards.append({'goal':cap(card.goal,40), 'findings':[cap(s,64) for s in card.findings[:3]],
                      'followups':[cap(s,32) for s in card.followups[:2]]})
    # Version belongs to the artifact/diagnostics, not the embedded semantic fields.
    view={'current_need':cap(source.current_need,64),
          'pending_needs':[cap(s,40) for s in source.pending_needs[-3:]],
          'prior_results':cards,'recent_queries':[cap(s,32) for s in source.recent_queries[-6:]]}
    def render():
        return json.dumps(view,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    text=render()
    for field, counter in [('recent_queries','omitted_recent_queries'),('pending_needs','omitted_pending_needs'),('prior_results','omitted_result_cards')]:
        while token_count(text)>MAX_TOKENS and view[field]:
            view[field].pop(0);diagnostics[counter]+=1;text=render()
    if token_count(text)>MAX_TOKENS:
        raise ValueError('Current need alone exceeds v2 serialization bound')
    diagnostics['tokens']=token_count(text)
    return text,diagnostics
