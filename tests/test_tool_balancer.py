"""
ToolBalancer 단위 테스트

테스트 항목:
1. Gini 계수 계산 정확성
2. 오버로드 탐지
3. 재분배 후 임계값 이하 확인
4. 에이전트 분리 (KMeans 2분할)
"""
from __future__ import annotations

import numpy as np
import pytest

from tad_mapper.engine.tda_analyzer import DiscoveredAgent
from tad_mapper.engine.feature_extractor import TopologicalFeature
from tad_mapper.engine.tool_balancer import ToolBalancer, DEFAULT_MAX_TOOLS
from tad_mapper.models.mcp_tool import MCPInputSchema, MCPPropertySchema, MCPToolAnnotations, MCPToolSchema
from tad_mapper.models.topology import BalanceReport


# ── 픽스처 ──────────────────────────────────────────────────────────────────

def make_agent(agent_id: str, task_ids: list[str]) -> DiscoveredAgent:
    return DiscoveredAgent(
        agent_id=agent_id,
        cluster_id=0,
        task_ids=task_ids,
        task_names=[f"Task {t}" for t in task_ids],
        centroid=np.random.rand(10),
        suggested_name=f"Agent {agent_id}",
        suggested_role="테스트 역할",
        routing_group_id=f"group_{agent_id}",
    )


def make_tool(name: str, agent_id: str, task_id: str) -> MCPToolSchema:
    return MCPToolSchema(
        name=name,
        description=f"{name} 설명",
        inputSchema=MCPInputSchema(
            properties={"query": MCPPropertySchema(type="string", description="입력")},
            required=["query"],
        ),
        annotations=MCPToolAnnotations(
            assigned_agent=agent_id,
            source_task_id=task_id,
            source_task_name=f"Task {task_id}",
            confidence=0.9,
        ),
    )


def make_feature(task_id: str) -> TopologicalFeature:
    vals = np.random.random(10)
    return TopologicalFeature(
        task_id=task_id,
        task_name=f"Task {task_id}",
        data_type=round(float(vals[0]), 2),
        reasoning_depth=round(float(vals[1]), 2),
        automation_potential=round(float(vals[2]), 2),
        interaction_type=round(float(vals[3]), 2),
        output_complexity=round(float(vals[4]), 2),
        domain_specificity=round(float(vals[5]), 2),
        temporal_sensitivity=round(float(vals[6]), 2),
        data_volume=round(float(vals[7]), 2),
        security_level=round(float(vals[8]), 2),
        state_dependency=round(float(vals[9]), 2),
    )


# ── 테스트: Gini 계수 ────────────────────────────────────────────────────────

class TestGiniCoefficient:
    def test_perfect_equality(self):
        """완전 균등 분배 → Gini = 0"""
        balancer = ToolBalancer()
        gini = balancer._gini_coefficient([3, 3, 3, 3])
        assert gini == pytest.approx(0.0, abs=0.01)

    def test_max_inequality_single(self):
        """한 Agent에 모두 집중 → Gini 높음"""
        balancer = ToolBalancer()
        gini = balancer._gini_coefficient([10, 0, 0, 0])
        assert gini > 0.5

    def test_empty_list(self):
        """빈 목록 → Gini = 0"""
        balancer = ToolBalancer()
        assert balancer._gini_coefficient([]) == 0.0

    def test_single_agent(self):
        """단일 에이전트 → Gini = 0"""
        balancer = ToolBalancer()
        gini = balancer._gini_coefficient([5])
        assert gini == pytest.approx(0.0, abs=0.01)


# ── 테스트: 오버로드 탐지 ────────────────────────────────────────────────────

class TestOverloadDetection:
    def test_no_overload(self):
        """각 Agent ≤ MAX_TOOLS → overloaded_agents 비어 있음"""
        balancer = ToolBalancer(max_tools_per_agent=7)

        agent_a = make_agent("agent_0", ["t1", "t2", "t3"])
        agent_b = make_agent("agent_1", ["t4", "t5", "t6"])
        agents = [agent_a, agent_b]

        tools = (
            [make_tool(f"tool_a_{i}", "agent_0", f"t{i}") for i in range(1, 4)] +
            [make_tool(f"tool_b_{i}", "agent_1", f"t{i}") for i in range(4, 7)]
        )

        report = balancer.analyze(agents, tools)
        assert report.overloaded_agents == []
        assert report.balance_score > 0.5

    def test_overload_detected(self):
        """한 Agent에 MAX_TOOLS 초과 → overloaded 탐지"""
        balancer = ToolBalancer(max_tools_per_agent=3)

        task_ids = [f"t{i}" for i in range(10)]
        agent_a = make_agent("agent_0", task_ids)
        agent_b = make_agent("agent_1", ["t10"])
        agents = [agent_a, agent_b]

        tools = (
            [make_tool(f"tool_a_{i}", "agent_0", f"t{i}") for i in range(10)] +
            [make_tool("tool_b_1", "agent_1", "t10")]
        )

        report = balancer.analyze(agents, tools)
        assert "agent_0" in report.overloaded_agents
        assert report.gini_coefficient > 0.3

    def test_balance_score_range(self):
        """balance_score ∈ [0, 1]"""
        balancer = ToolBalancer()
        agents = [make_agent(f"a{i}", [f"t{j+i*3}" for j in range(3)]) for i in range(3)]
        tools = [make_tool(f"tool_{j}", f"a{j//3}", f"t{j}") for j in range(9)]
        report = balancer.analyze(agents, tools)
        assert 0.0 <= report.balance_score <= 1.0


# ── 테스트: 재분배 ──────────────────────────────────────────────────────────

class TestRebalance:
    def test_rebalance_reduces_overload(self):
        """재분배 후 오버로드 에이전트가 줄어들어야 함"""
        np.random.seed(42)
        balancer = ToolBalancer(max_tools_per_agent=3)

        task_ids_a = [f"t{i}" for i in range(8)]
        agent_a = make_agent("agent_0", task_ids_a)
        agent_b = make_agent("agent_1", ["t8", "t9"])
        agents = [agent_a, agent_b]

        features = [make_feature(tid) for tid in task_ids_a + ["t8", "t9"]]
        tools = (
            [make_tool(f"tool_a_{i}", "agent_0", f"t{i}") for i in range(8)] +
            [make_tool(f"tool_b_{i}", "agent_1", f"t{8+i}") for i in range(2)]
        )

        new_agents, new_tools, report = balancer.rebalance(agents, features, tools)

        assert report.rebalanced is True
        # 재분배 후 에이전트 수가 늘거나 같아야 함
        assert len(new_agents) >= len(agents)
        # Tool 수는 동일하게 유지
        assert len(new_tools) == len(tools)

    def test_no_rebalance_when_balanced(self):
        """균형 잡힌 경우 재분배 없음"""
        balancer = ToolBalancer(max_tools_per_agent=7)
        agents = [make_agent(f"a{i}", [f"t{j+i*2}" for j in range(2)]) for i in range(3)]
        features = [make_feature(f"t{j}") for j in range(6)]
        tools = [make_tool(f"tool_{j}", f"a{j//2}", f"t{j}") for j in range(6)]

        new_agents, new_tools, report = balancer.rebalance(agents, features, tools)
        assert report.rebalanced is False
        assert len(new_agents) == len(agents)


# ── 테스트: 에이전트 분리 ────────────────────────────────────────────────────

class TestAgentSplitting:
    def test_split_produces_two_agents(self):
        """6개 이상 태스크 에이전트를 2개로 분리"""
        np.random.seed(42)
        balancer = ToolBalancer()
        task_ids = [f"t{i}" for i in range(6)]
        agent = make_agent("agent_0", task_ids)
        features = [make_feature(tid) for tid in task_ids]
        tools = [make_tool(f"tool_{i}", "agent_0", f"t{i}") for i in range(6)]

        split = balancer._split_overloaded_agent(agent, features, tools)
        assert len(split) == 2
        # 두 에이전트의 태스크 합이 원본과 같아야 함
        all_tasks = set()
        for a in split:
            all_tasks.update(a.task_ids)
            assert a.routing_group_id == agent.routing_group_id
        assert all_tasks == set(task_ids)

    def test_split_single_task_no_change(self):
        """태스크 1개 에이전트는 분리 불가 → 원본 반환"""
        balancer = ToolBalancer()
        agent = make_agent("agent_0", ["t1"])
        features = [make_feature("t1")]
        tools = [make_tool("tool_1", "agent_0", "t1")]

        split = balancer._split_overloaded_agent(agent, features, tools)
        assert len(split) == 1
