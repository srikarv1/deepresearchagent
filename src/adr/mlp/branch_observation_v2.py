"""Frozen candidate-conditioned contract; pure offline construction, no gate activation.

Callers supply an immutable pre-retrieval snapshot. Completed results MUST be in
publication order, not completion order. IDs are used only for validation/audit.
"""
from dataclasses import dataclass
import json
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from adr.mlp.embedding_limits import token_count
from adr.synthetic.schema import ResearchSnapshot, derive_features, FeatureVector, FEATURE_NAMES

VERSION = 'mlp-observation-v2-branches-cited-top3-v1'
MAX_TOKENS = 4096
EMBEDDING_CONTRACT = {
    'version': VERSION, 'fields': ['original_question', 'context_before', 'candidate_subquery'],
    'model': 'text-embedding-3-small', 'dimensions_each': 1536,
    'feature_names': list(FEATURE_NAMES), 'input_dimensions': 4623,
}

class Record(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

class Task(Record):
    id: str
    parent_id: str | None = None
    tree_depth: int = Field(ge=1)

class PublishedBranch(Record):
    task_id: str
    goal: str
    findings: tuple[str, ...] = ()
    followups: tuple[str, ...] = ()
    cited_insights: dict[str, str] = Field(default_factory=dict)

class ObservationSource(Record):
    current_task_id: str
    current_need: str
    tasks: tuple[Task, ...]
    # This list is the publication boundary, not the complete result registry.
    published_branches: tuple[PublishedBranch, ...] = ()
    executed_queries: tuple[str, ...] = ()

    @model_validator(mode='after')
    def hierarchy(self):
        tasks = {t.id: t for t in self.tasks}
        if len(tasks) != len(self.tasks) or self.current_task_id not in tasks:
            raise ValueError('Duplicate/missing task identity')
        for t in self.tasks:
            if t.parent_id is None:
                if t.tree_depth != 1:
                    raise ValueError('Invalid root depth')
            elif t.parent_id not in tasks or tasks[t.parent_id].tree_depth + 1 != t.tree_depth:
                raise ValueError('Invalid parent/depth')
        ids = [b.task_id for b in self.published_branches]
        if len(ids) != len(set(ids)) or any(i not in tasks or i == self.current_task_id for i in ids):
            raise ValueError('Invalid published branch identity')
        return self

@dataclass(frozen=True)
class Observation:
    text: str
    tokens: int
    overflow: bool
    selected_task_ids: tuple[str, ...]
    scores: tuple[float, ...]
    version: str = VERSION

    @property
    def required_bypass_decision(self):
        """Future consumers must bypass embedding/MLP on overflow."""
        return 'EXECUTE' if self.overflow else None


def construct(source: ObservationSource, candidate: str) -> Observation:
    docs = ['\n'.join((b.goal, *b.findings, *b.followups)) for b in source.published_branches]
    scores = [0.0] * len(docs)
    if docs:
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english',
            ngram_range=(1, 2), sublinear_tf=True, smooth_idf=True, norm='l2')
        # Empty/all-stopword corpora are well-defined zero-score publication ties.
        if any(vectorizer.build_analyzer()(doc) for doc in docs):
            matrix = vectorizer.fit_transform(docs)
            scores = (matrix @ vectorizer.transform([candidate]).T).toarray().ravel().tolist()
    selected = sorted(range(len(docs)), key=lambda i: (-scores[i], i))[:3]
    tasks = {t.id: t for t in source.tasks}
    parent = tasks[source.current_task_id].parent_id
    ancestors = set()
    node = parent
    while node is not None:
        ancestors.add(node)
        node = tasks[node].parent_id
    packets = []
    for i in selected:
        b = source.published_branches[i]
        packets.append({'goal': b.goal, 'findings': list(b.findings),
            'followups': list(b.followups), 'cited_insights': dict(b.cited_insights),
            'relation': 'parent' if b.task_id == parent else 'ancestor' if b.task_id in ancestors else 'other',
            'tree_depth': tasks[b.task_id].tree_depth})
    text = json.dumps({'current_need': source.current_need,
        'recent_queries': list(source.executed_queries[-6:]), 'branches': packets},
        ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    tokens = token_count(text)
    return Observation(text, tokens, tokens > MAX_TOKENS,
        tuple(source.published_branches[i].task_id for i in selected), tuple(scores[i] for i in selected))


def model_inputs(question: str, candidate: str, source: ObservationSource,
                 complete_state: ResearchSnapshot) -> tuple[Observation, FeatureVector]:
    """Shared train/live seam. Selection never filters the numerical source state.

    Returns no embeddings and makes no decision. Overflow is an explicit EXECUTE
    bypass contract; this module is deliberately not wired into the live gate.
    """
    return construct(source, candidate), derive_features(question, complete_state)
