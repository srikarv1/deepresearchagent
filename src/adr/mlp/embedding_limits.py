"""Offline admission for text-embedding-3-small.

Installed OpenAI SDK embeddings.create documents 8192 tokens per input;
OpenAI tiktoken maps this model to cl100k_base. Never download tokenizer data.
"""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import tempfile

MODEL = 'text-embedding-3-small'
MAX_INPUT_TOKENS = 8192


@lru_cache(maxsize=1)
def encoding():
    import tiktoken
    url = 'https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken'
    directory = os.environ.get('TIKTOKEN_CACHE_DIR', os.environ.get('DATA_GYM_CACHE_DIR', str(Path(tempfile.gettempdir())/'data-gym-cache')))
    path = Path(directory)/hashlib.sha1(url.encode()).hexdigest()
    expected = '223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7'
    if not directory or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Verified local cl100k_base tokenizer cache required; no download attempted')
    result = tiktoken.encoding_for_model(MODEL)
    if result.name != 'cl100k_base':
        raise ValueError('Unexpected embedding tokenizer')
    return result


def token_count(text):
    # Strings are embedded as ordinary text, including apparent special-token literals.
    return len(encoding().encode_ordinary(text))
