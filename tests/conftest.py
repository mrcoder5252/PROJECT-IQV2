"""Offline fakes so the whole pipeline can be tested without downloading models."""
import hashlib
import re

import numpy as np
import pytest


class FakeEncoder:  # stands in for tiktoken
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


def fake_embed(texts, model_name=None, dim=256):
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
        n = np.linalg.norm(out[i])
        out[i] /= n if n else 1.0
    return out


@pytest.fixture
def offline(monkeypatch):
    import src.chunk_for_rag as ch
    import src.vector_store as vs
    monkeypatch.setattr(ch, "get_encoder", lambda: FakeEncoder())
    monkeypatch.setattr(vs, "encode", fake_embed)
    return True
