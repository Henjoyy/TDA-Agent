"""
Embedder fallback 벡터 테스트
"""
from __future__ import annotations

import numpy as np

from tad_mapper.engine.embedder import Embedder


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
