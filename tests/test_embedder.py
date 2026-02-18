"""
Embedder fallback 벡터 테스트
"""
from __future__ import annotations

import numpy as np

from tad_mapper.engine.embedder import Embedder, EmbeddingHealth


def test_deterministic_fallback_vector_is_reproducible():
    text = "동일한 입력 텍스트"
    v1 = Embedder._deterministic_fallback_vector(text)
    v2 = Embedder._deterministic_fallback_vector(text)

    assert np.allclose(v1, v2)
    assert len(v1) == Embedder.EMBEDDING_DIM


def test_deterministic_fallback_vector_changes_with_text():
    v1 = Embedder._deterministic_fallback_vector("text-a")
    v2 = Embedder._deterministic_fallback_vector("text-b")

    assert not np.allclose(v1, v2)


def test_embedder_switches_model_when_primary_not_found():
    embedder = Embedder.__new__(Embedder)
    embedder._candidate_models = ["primary", "backup"]
    embedder._health = EmbeddingHealth(active_model="primary", model_available=True)

    def fake_request(text: str, model: str):
        if model == "primary":
            raise RuntimeError("404 NOT_FOUND")
        return [0.1] * Embedder.EMBEDDING_DIM

    embedder._request_embedding = fake_request
    vec = embedder._try_switch_embedding_model("hello")

    assert vec is not None
    assert embedder._health.active_model == "backup"
    assert embedder._health.model_available is True
