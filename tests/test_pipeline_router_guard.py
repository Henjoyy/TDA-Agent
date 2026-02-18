"""
Pipeline 라우터 가드 로직 테스트
"""
from __future__ import annotations

import tad_mapper.pipeline as pipeline_module
from tad_mapper.pipeline import TADMapperPipeline
from tad_mapper.models.topology import (
    CompositionPlan,
    HierarchicalRoutingPlan,
    RoutingResult,
    SubTaskAssignment,
)


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


def test_router_block_when_fallback_ratio_too_high(monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "ROUTER_DISABLE_ON_FALLBACK_RATIO", True
    )
    reason = TADMapperPipeline._evaluate_router_block_reason(
        {
            "model_available": True,
            "active_model": "ok-model",
            "total_calls": 10,
            "fallback_ratio": 0.95,
        }
    )
    assert "fallback 비율 과다" in reason


def test_router_allows_high_fallback_in_hybrid_mode(monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "ROUTER_DISABLE_ON_FALLBACK_RATIO", False
    )
    reason = TADMapperPipeline._evaluate_router_block_reason(
        {
            "model_available": True,
            "active_model": "ok-model",
            "total_calls": 10,
            "fallback_ratio": 0.95,
        }
    )
    assert reason == ""


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


def test_route_hierarchy_and_compose_uses_subtask_assignments(monkeypatch):
    pipeline = TADMapperPipeline.__new__(TADMapperPipeline)

    routing = RoutingResult(
        query_text="q",
        target_agent_id="unit_a",
        target_agent_name="Unit A",
        homotopy_class_id="class_a",
        confidence=0.4,
        top_similarity=0.5,
        routing_probabilities={"unit_a": 0.6, "unit_b": 0.4},
        alternatives=[],
        is_ambiguous=True,
        ambiguity_reason="close",
    )
    hierarchical = HierarchicalRoutingPlan(
        query_text="작업A 그리고 작업B",
        path_type="master_orchestrator_unit",
        routing=routing,
        complexity_score=0.8,
        complexity_threshold=0.45,
        selected_orchestrator_ids=["orch_a"],
        selected_unit_ids=["unit_a", "unit_b"],
        subtasks=["작업A", "작업B"],
        assignments=[
            SubTaskAssignment(
                subtask_id="subtask_1",
                subtask_text="작업A",
                orchestrator_id="orch_a",
                unit_agent_id="unit_a",
                score=0.7,
            ),
            SubTaskAssignment(
                subtask_id="subtask_2",
                subtask_text="작업B",
                orchestrator_id="orch_a",
                unit_agent_id="unit_b",
                score=0.6,
            ),
        ],
    )

    monkeypatch.setattr(pipeline, "plan_hierarchy", lambda query: hierarchical)

    def _compose(subtask: str, unit_id: str) -> CompositionPlan:
        return CompositionPlan(query_text=subtask, agent_id=unit_id)

    monkeypatch.setattr(pipeline, "compose_tools", _compose)

    execution = TADMapperPipeline.route_hierarchy_and_compose(
        pipeline, "작업A 그리고 작업B"
    )

    assert execution.hierarchical_routing.path_type == "master_orchestrator_unit"
    assert execution.total_steps == 2
    assert execution.execution_steps[0].unit_agent_id == "unit_a"
    assert execution.execution_steps[1].unit_agent_id == "unit_b"


def test_route_hierarchy_and_compose_merges_same_unit_subtasks(monkeypatch):
    pipeline = TADMapperPipeline.__new__(TADMapperPipeline)

    routing = RoutingResult(
        query_text="q",
        target_agent_id="unit_a",
        target_agent_name="Unit A",
        homotopy_class_id="class_a",
        confidence=0.4,
        top_similarity=0.5,
        routing_probabilities={"unit_a": 0.9},
        alternatives=[],
        is_ambiguous=True,
        ambiguity_reason="close",
    )
    hierarchical = HierarchicalRoutingPlan(
        query_text="작업A 그리고 작업B",
        path_type="master_orchestrator_unit",
        routing=routing,
        complexity_score=0.8,
        complexity_threshold=0.45,
        selected_orchestrator_ids=["orch_a"],
        selected_unit_ids=["unit_a"],
        subtasks=["작업A", "작업B"],
        assignments=[
            SubTaskAssignment(
                subtask_id="subtask_1",
                subtask_text="작업A",
                orchestrator_id="orch_a",
                unit_agent_id="unit_a",
                score=0.7,
            ),
            SubTaskAssignment(
                subtask_id="subtask_2",
                subtask_text="작업B",
                orchestrator_id="orch_a",
                unit_agent_id="unit_a",
                score=0.6,
            ),
        ],
    )

    monkeypatch.setattr(pipeline, "plan_hierarchy", lambda query: hierarchical)
    monkeypatch.setattr(
        pipeline,
        "compose_tools",
        lambda subtask, unit_id: CompositionPlan(query_text=subtask, agent_id=unit_id),
    )

    execution = TADMapperPipeline.route_hierarchy_and_compose(
        pipeline, "작업A 그리고 작업B"
    )

    assert execution.total_steps == 1
    step = execution.execution_steps[0]
    assert step.unit_agent_id == "unit_a"
    assert step.source_subtask_ids == ["subtask_1", "subtask_2"]
    assert "작업A" in step.subtask_text and "작업B" in step.subtask_text
