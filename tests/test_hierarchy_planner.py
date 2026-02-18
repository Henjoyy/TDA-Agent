from __future__ import annotations

import numpy as np

from tad_mapper.engine.hierarchy_planner import HierarchyPlanner
from tad_mapper.engine.tda_analyzer import DiscoveredAgent
from tad_mapper.models.topology import RoutingResult


def _agents() -> list[DiscoveredAgent]:
    return [
        DiscoveredAgent(
            agent_id="unit_trade_a",
            cluster_id=0,
            task_ids=["t1", "t2"],
            task_names=["환율 분석", "거래 리스크 분석"],
            centroid=np.zeros(10),
            suggested_name="무역 분석 에이전트 A",
            suggested_role="환율/리스크 분석",
            suggested_capabilities=["환율", "리스크", "분석"],
            routing_group_id="trade",
        ),
        DiscoveredAgent(
            agent_id="unit_trade_b",
            cluster_id=1,
            task_ids=["t3"],
            task_names=["무역 보고서 생성"],
            centroid=np.zeros(10),
            suggested_name="무역 분석 에이전트 B",
            suggested_role="보고서 생성",
            suggested_capabilities=["보고서", "무역"],
            routing_group_id="trade",
        ),
        DiscoveredAgent(
            agent_id="unit_alert",
            cluster_id=2,
            task_ids=["t4"],
            task_names=["알림 전송"],
            centroid=np.zeros(10),
            suggested_name="알림 에이전트",
            suggested_role="알림",
            suggested_capabilities=["알림"],
            routing_group_id="alert",
        ),
    ]


def test_build_hierarchy_with_orchestrator_and_direct_unit():
    planner = HierarchyPlanner()
    blueprint = planner.build(_agents())

    assert "orch_trade" in blueprint.orchestrator_to_units
    assert set(blueprint.orchestrator_to_units["orch_trade"]) == {
        "unit_trade_a",
        "unit_trade_b",
    }
    assert blueprint.unit_to_orchestrator["unit_trade_a"] == "orch_trade"
    assert blueprint.unit_to_orchestrator["unit_trade_b"] == "orch_trade"
    assert blueprint.unit_to_orchestrator["unit_alert"] == ""

    master = next(n for n in blueprint.nodes if n.tier == "MASTER")
    assert "orch_trade" in master.child_ids
    assert "unit_alert" in master.child_ids


def test_plan_uses_master_unit_path_for_simple_query():
    planner = HierarchyPlanner(simple_threshold=0.45)
    planner.build(_agents())
    routing = RoutingResult(
        query_text="환율 조회",
        target_agent_id="unit_trade_a",
        target_agent_name="무역 분석 에이전트 A",
        homotopy_class_id="class_trade",
        confidence=0.9,
        top_similarity=0.8,
        routing_probabilities={"unit_trade_a": 0.9, "unit_trade_b": 0.1},
        alternatives=[],
        is_ambiguous=False,
        ambiguity_reason="",
    )

    plan = planner.plan("환율 조회", routing)
    assert plan.path_type == "master_unit"
    assert plan.selected_unit_ids == ["unit_trade_a"]
    assert len(plan.assignments) == 1
    assert plan.assignments[0].unit_agent_id == "unit_trade_a"


def test_plan_uses_orchestrator_path_for_complex_query():
    planner = HierarchyPlanner(simple_threshold=0.35, max_orchestrators=2)
    planner.build(_agents())
    routing = RoutingResult(
        query_text="환율 분석 그리고 리스크 보고서 작성",
        target_agent_id="unit_trade_a",
        target_agent_name="무역 분석 에이전트 A",
        homotopy_class_id="class_trade",
        confidence=0.18,
        top_similarity=0.31,
        routing_probabilities={
            "unit_trade_a": 0.45,
            "unit_trade_b": 0.40,
            "unit_alert": 0.15,
        },
        alternatives=[],
        is_ambiguous=True,
        ambiguity_reason="close similarities",
    )

    plan = planner.plan("환율 분석 그리고 리스크 보고서 작성", routing)
    assert plan.path_type == "master_orchestrator_unit"
    assert "orch_trade" in plan.selected_orchestrator_ids
    assert len(plan.subtasks) == 2
    assert len(plan.assignments) == 2
    assert {a.unit_agent_id for a in plan.assignments}.issubset(
        {"unit_trade_a", "unit_trade_b"}
    )
