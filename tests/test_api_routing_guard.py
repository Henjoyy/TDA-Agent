"""
API 라우팅 가드 테스트
"""
from __future__ import annotations

from fastapi.testclient import TestClient

import api.main as api_main
from tad_mapper.models.topology import (
    CompositionPlan,
    HierarchicalExecutionPlan,
    HierarchicalExecutionStep,
    HierarchicalRoutingPlan,
    RoutingResult,
)


class _ReadyRouter:
    is_ready = True


class _DummyPipeline:
    def __init__(self, routing_result: RoutingResult):
        self.router = _ReadyRouter()
        self.router_block_reason = ""
        self._routing_result = routing_result

    def route_query(self, query: str) -> RoutingResult:
        return self._routing_result

    def plan_hierarchy(self, query: str) -> HierarchicalRoutingPlan:
        return HierarchicalRoutingPlan(
            query_text=query,
            path_type="master_unit",
            routing=self._routing_result,
            complexity_score=0.2,
            complexity_threshold=0.45,
            selected_unit_ids=[self._routing_result.target_agent_id],
            subtasks=[query],
            assignments=[],
            rationale=["test"],
        )

    def route_hierarchy_and_compose(self, query: str) -> HierarchicalExecutionPlan:
        routing = self.plan_hierarchy(query)
        return HierarchicalExecutionPlan(
            query_text=query,
            hierarchical_routing=routing,
            execution_steps=[
                HierarchicalExecutionStep(
                    subtask_id="subtask_1",
                    subtask_text=query,
                    unit_agent_id=self._routing_result.target_agent_id,
                    composition_plan=CompositionPlan(
                        query_text=query,
                        agent_id=self._routing_result.target_agent_id,
                    ),
                )
            ],
        )


class _DummyResult:
    def __init__(self, fallback_ratio: float = 0.0):
        self.embedding_health = {"fallback_ratio": fallback_ratio}


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
    monkeypatch.setattr(
        api_main, "_get_session", lambda output_id: (pipeline, _DummyResult(0.0))
    )

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


def test_route_rejects_small_probability_gap(monkeypatch):
    confident_but_close = RoutingResult(
        query_text="q",
        target_agent_id="a1",
        target_agent_name="Agent 1",
        homotopy_class_id="class_a1",
        confidence=0.9,
        top_similarity=0.9,
        routing_probabilities={"a1": 0.51, "a2": 0.49},
        alternatives=[],
        is_ambiguous=False,
        ambiguity_reason="",
    )
    pipeline = _DummyPipeline(confident_but_close)
    monkeypatch.setattr(
        api_main, "_get_session", lambda output_id: (pipeline, _DummyResult(0.0))
    )
    monkeypatch.setattr(api_main, "ROUTE_MIN_PROB_GAP", 0.05)

    client = TestClient(api_main.app)
    res = client.post("/api/route", json={"output_id": "o1", "query": "test"})
    assert res.status_code == 409
    payload = res.json()
    assert payload["detail"]["prob_gap"] == 0.02
    assert payload["detail"]["prob_gap_threshold"] == 0.05


def test_route_allows_ambiguous_when_fallback_ratio_high(monkeypatch):
    ambiguous_but_usable = RoutingResult(
        query_text="q",
        target_agent_id="a1",
        target_agent_name="Agent 1",
        homotopy_class_id="class_a1",
        confidence=0.12,
        top_similarity=0.4,
        routing_probabilities={"a1": 0.6, "a2": 0.4},
        alternatives=[],
        is_ambiguous=True,
        ambiguity_reason="close similarities",
    )
    pipeline = _DummyPipeline(ambiguous_but_usable)
    monkeypatch.setattr(
        api_main, "_get_session", lambda output_id: (pipeline, _DummyResult(1.0))
    )
    monkeypatch.setattr(api_main, "ROUTE_MIN_CONFIDENCE", 0.35)
    monkeypatch.setattr(api_main, "ROUTE_MIN_PROB_GAP", 0.0)

    client = TestClient(api_main.app)
    res = client.post("/api/route", json={"output_id": "o1", "query": "test"})
    assert res.status_code == 200
    payload = res.json()
    assert payload["routing_policy"]["ambiguity_enforced"] is False
    assert payload["routing_policy"]["min_confidence"] == 0.1


def test_route_hierarchy_returns_plan(monkeypatch):
    routing = RoutingResult(
        query_text="q",
        target_agent_id="a1",
        target_agent_name="Agent 1",
        homotopy_class_id="class_a1",
        confidence=0.8,
        top_similarity=0.7,
        routing_probabilities={"a1": 0.8, "a2": 0.2},
        alternatives=[],
        is_ambiguous=False,
        ambiguity_reason="",
    )
    pipeline = _DummyPipeline(routing)
    monkeypatch.setattr(
        api_main, "_get_session", lambda output_id: (pipeline, _DummyResult(0.0))
    )

    client = TestClient(api_main.app)
    res = client.post("/api/route-hierarchy", json={"output_id": "o1", "query": "test"})
    assert res.status_code == 200
    payload = res.json()
    assert payload["hierarchical_plan"]["path_type"] == "master_unit"
    assert payload["hierarchical_plan"]["selected_unit_ids"] == ["a1"]


def test_route_hierarchy_and_compose_returns_execution(monkeypatch):
    routing = RoutingResult(
        query_text="q",
        target_agent_id="a1",
        target_agent_name="Agent 1",
        homotopy_class_id="class_a1",
        confidence=0.8,
        top_similarity=0.7,
        routing_probabilities={"a1": 0.8, "a2": 0.2},
        alternatives=[],
        is_ambiguous=False,
        ambiguity_reason="",
    )
    pipeline = _DummyPipeline(routing)
    monkeypatch.setattr(
        api_main, "_get_session", lambda output_id: (pipeline, _DummyResult(0.0))
    )

    client = TestClient(api_main.app)
    res = client.post(
        "/api/route-hierarchy-and-compose",
        json={"output_id": "o1", "query": "test"},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["hierarchical_execution"]["hierarchical_routing"]["path_type"] == "master_unit"
    assert payload["hierarchical_execution"]["total_steps"] == 1
