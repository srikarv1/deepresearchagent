"""Strict frozen MLP #1 inference. Never fits, writes training data, or creates a client at import."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from adr.mlp.prepare import embedding_identity, normalize
from adr.synthetic.schema import FEATURE_NAMES, derive_features
from adr.mlp.live_context import embedding_context
from adr.mlp.embedding_limits import token_count, MAX_INPUT_TOKENS

DEFAULT_RUN = Path('runs/mlp/baseline-v1-20260923T044920906450Z')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_json(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Frozen artifact checksum mismatch: {Path(path).name}')
    return json.loads(Path(path).read_text())


class FrozenMLP:
    def __init__(self, run=DEFAULT_RUN, *, embedder=None, cache_size=1024):
        run = Path(run)
        manifest = json.loads((run/'manifest.json').read_text())
        if manifest['status'] != 'complete':
            raise ValueError('Incomplete training run')
        cfg = checked_json(run/'config.json', manifest['artifacts_sha256']['config.json'])
        required = {'input_dimensions':4623, 'hidden_dimensions':[256,64], 'dropout':0.2,
                    'threshold':0.5, 'seed':42}
        if any(cfg.get(k) != v for k,v in required.items()):
            raise ValueError('Frozen MLP #1 architecture/config/threshold incompatibility')
        prepared = Path(cfg['prepared'])
        contract = checked_json(prepared/'contract.json', manifest['prepared_sha256']['contract.json'])
        self.stats = checked_json(prepared/'normalization.json', manifest['prepared_sha256']['normalization.json'])
        if (contract['feature_names'] != list(FEATURE_NAMES) or
            contract['text_fields'] != ['original_question','context_before','candidate_subquery'] or
            contract['labels'] != {'EXECUTE':1,'SKIP':0} or
            contract['embedding']['model'] != 'text-embedding-3-small' or
            contract['embedding']['dimensions'] != 1536 or
            self.stats['feature_names'] != list(FEATURE_NAMES) or self.stats['fit_split'] != 'train'):
            raise ValueError('Frozen feature/text/embedding contract incompatibility')
        for key in ('mean','scale','std','zero_variance'):
            if np.asarray(self.stats[key]).shape != (15,):
                raise ValueError('Invalid frozen normalizer dimensions')
        if not np.isfinite(self.stats['mean']).all() or not np.isfinite(self.stats['scale']).all() or np.any(np.array(self.stats['scale']) <= 0):
            raise ValueError('Invalid frozen normalizer values')
        if digest(run/'best_mlp.pt') != manifest['artifacts_sha256']['best_mlp.pt']:
            raise ValueError('Frozen checkpoint checksum mismatch')
        checkpoint = torch.load(run/'best_mlp.pt', map_location='cpu', weights_only=True)
        if checkpoint['config'] != cfg:
            raise ValueError('Checkpoint/config incompatibility')
        torch.set_num_threads(1)
        self.model = nn.Sequential(nn.Linear(4623,256), nn.ReLU(), nn.Dropout(.2),
                                   nn.Linear(256,64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64,1))
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.identity = contract['embedding']
        self.provenance = {'checkpoint_sha256':digest(run/'best_mlp.pt'),
                           'normalization_sha256':digest(prepared/'normalization.json'),
                           'contract_sha256':digest(prepared/'contract.json'),
                           'threshold':.5, 'feature_names':list(FEATURE_NAMES)}
        self.embedder = embedder
        self.client = None
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.lock = asyncio.Lock()

    def validate_endpoint(self):
        if embedding_identity('text-embedding-3-small',1536) != self.identity:
            raise ValueError('Embedding endpoint/model identity differs from frozen preparation')
        if not os.environ.get('OPENAI_API_KEY'):
            raise ValueError('OPENAI_API_KEY is required for live inference')

    async def vectors(self, texts):
        async with self.lock:
            unique = list(dict.fromkeys(texts))
            found = {t:self.cache[t] for t in unique if t in self.cache}
            missing = [t for t in unique if t not in found]
            if any(not t or token_count(t) > MAX_INPUT_TOKENS for t in missing):
                raise ValueError('Inference text empty or exceeds model embedding token limit')
            for start in range(0,len(missing),16):
                batch = missing[start:start+16]
                if self.embedder is None:
                    self.validate_endpoint()
                    if self.client is None:
                        from adr.llm.openai_compat import OpenAICompatLLM
                        self.client = OpenAICompatLLM(model='text-embedding-3-small')
                    embed = self.client.embed
                else:
                    embed = self.embedder
                values = np.asarray(await embed(batch, dimensions=1536), dtype=np.float32)
                if values.shape != (len(batch),1536) or not np.isfinite(values).all():
                    raise ValueError('Invalid inference embedding dimensions/values')
                for text, vector in zip(batch,values):
                    found[text] = vector
                    self.cache[text] = vector
                    self.cache.move_to_end(text)
                    while len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
            return np.stack([found[t] for t in texts])

    async def predict_batch(self, question, snapshot, candidates):
        context, _ = embedding_context(snapshot)
        raw = derive_features(question,snapshot).values
        normalized = normalize(np.asarray([raw]), self.stats)[0]
        if not candidates:
            return [], context, raw, normalized.tolist()
        # Exactly the three independent semantic fields; no IDs/supervision.
        vectors = await self.vectors([question,context,*candidates])
        X = np.stack([np.concatenate((vectors[0],vectors[1],v,normalized)) for v in vectors[2:]])
        if X.shape != (len(candidates),4623) or not np.isfinite(X).all():
            raise ValueError('Invalid MLP inference input')
        with torch.inference_mode():
            probabilities = torch.sigmoid(self.model(torch.from_numpy(X)).squeeze(-1)).numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError('Non-finite MLP probability')
        return probabilities.tolist(), context, raw, normalized.tolist()

    async def close(self):
        if self.client:
            await self.client.close()
