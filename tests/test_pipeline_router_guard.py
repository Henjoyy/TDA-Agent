"""
Pipeline 라우터 가드 로직 테스트
"""
from __future__ import annotations

from tad_mapper.pipeline import TADMapperPipeline


def test_router_block_when_embedding_model_unavailable():
    reason = TADMapperPipeline._evaluate_router_block_reason(
        {
            "model_available": False,
            "active_model": "bad-model",
            "model_validation_error": "NOT_FOUND",
            "total_calls": 10,
            "fallback_ratio": 1.0,
        }
    )
    assert "임베딩 모델 사용 불가" in reason


def test_router_block_when_fallback_ratio_too_high():
    reason = TADMapperPipeline._evaluate_router_block_reason(
        {
            "model_available": True,
            "active_model": "ok-model",
            "total_calls": 10,
            "fallback_ratio": 0.95,
        }
    )
    assert "fallback 비율 과다" in reason


def test_router_allows_healthy_embedding_state():
    reason = TADMapperPipeline._evaluate_router_block_reason(
        {
            "model_available": True,
            "active_model": "ok-model",
            "total_calls": 10,
            "fallback_ratio": 0.0,
        }
    )
    assert reason == ""
