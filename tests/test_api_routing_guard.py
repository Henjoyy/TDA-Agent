"""
API 라우팅 가드 테스트
"""
from __future__ import annotations

from fastapi.testclient import TestClient

import api.main as api_main
from tad_mapper.models.topology import RoutingResult


class _ReadyRouter:
    is_ready = True


class _DummyPipeline:
    def __init__(self, routing_result: RoutingResult):
        self.router = _ReadyRouter()
        self.router_block_reason = ""
        self._routing_result = routing_result

    def route_query(self, query: str) -> RoutingResult:
        return self._routing_result


def test_route_rejects_low_confidence(monkeypatch):
    low_conf = RoutingResult(
        query_text="q",
        target_agent_id="a1",
        target_agent_name="Agent 1",
        homotopy_class_id="class_a1",
        confidence=0.01,
        top_similarity=0.1,
        alternatives=[],
        is_ambiguous=True,
        ambiguity_reason="low confidence",
    )
    pipeline = _DummyPipeline(low_conf)
    monkeypatch.setattr(api_main, "_get_session", lambda output_id: (pipeline, None))

    client = TestClient(api_main.app)
    res = client.post("/api/route", json={"output_id": "o1", "query": "test"})
    assert res.status_code == 409
    payload = res.json()
    assert "message" in payload["detail"]


def test_route_returns_503_when_router_not_ready(monkeypatch):
    class _NoRouterPipeline:
        router = None
        router_block_reason = "임베딩 모델 사용 불가"

    monkeypatch.setattr(
        api_main, "_get_session", lambda output_id: (_NoRouterPipeline(), None)
    )

    client = TestClient(api_main.app)
    res = client.post("/api/route", json={"output_id": "o1", "query": "test"})
    assert res.status_code == 503
    assert "임베딩 모델 사용 불가" in res.json()["detail"]
