"""
Tool Balancer — Agile MCP Tool 최적 분배 알고리즘

하나의 Agent에 MCP Tool이 너무 많으면 LLM 컨텍스트 초과,
라우팅 혼란, 성능 저하가 발생합니다.

이 모듈은 에이전트 간 Tool 부하를 분석하고,
Agile하게 threshold를 조정하여 최적 분배를 유지합니다.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans

from tad_mapper.engine.feature_extractor import TopologicalFeature
from tad_mapper.engine.tda_analyzer import DiscoveredAgent
from tad_mapper.models.mcp_tool import MCPToolSchema
from tad_mapper.models.topology import BalanceReport

logger = logging.getLogger(__name__)

# ── 기본 임계값 상수 ─────────────────────────────────────────────────────────
DEFAULT_MAX_TOOLS = 7        # Agent당 최대 MCP Tool 수 (LLM 성능 최적 범위)
DEFAULT_MIN_TOOLS = 2        # Agent당 최소 MCP Tool 수
MAX_REBALANCE_ITERATIONS = 3 # 재분배 최대 반복 횟수
GINI_WARN_THRESHOLD = 0.3    # Gini 계수 경고 임계값 (0.3 이상이면 불균형)


@dataclass
class _AgentToolStats:
    """에이전트별 Tool 통계"""
    agent_id: str
    agent_name: str
    tool_count: int
    tool_names: list[str] = field(default_factory=list)


class ToolBalancer:
    """
    MCP Tool 에이전트 간 부하 균형 분석 및 조정.

    알고리즘:
    1. 현재 분포 분석 (Gini coefficient)
    2. 오버로드 에이전트 탐지 (> MAX_TOOLS)
    3. n_agents를 +1 증가시켜 TDA 재클러스터링 (Agile threshold 조정)
    4. 수렴 조건 충족 시 종료
    5. 최대 반복 초과 시 강제 분리
    """

    def __init__(
        self,
        max_tools_per_agent: int = DEFAULT_MAX_TOOLS,
        min_tools_per_agent: int = DEFAULT_MIN_TOOLS,
    ) -> None:
        self.max_tools = max_tools_per_agent
        self.min_tools = min_tools_per_agent

    # ── 분석 ─────────────────────────────────────────────────────────────────

    def analyze(
        self,
        agents: list[DiscoveredAgent],
        mcp_tools: list[MCPToolSchema],
    ) -> BalanceReport:
        """
        에이전트별 Tool 분포를 분석하고 균형 보고서를 반환합니다.

        Args:
            agents: 발견된 Unit Agent 목록
            mcp_tools: 생성된 MCP Tool 스키마 목록

        Returns:
            BalanceReport: 분포 통계, 오버/언더로드 목록, Gini 계수
        """
        stats = self._compute_stats(agents, mcp_tools)
        counts = {s.agent_id: s.tool_count for s in stats}

        overloaded = [s.agent_id for s in stats if s.tool_count > self.max_tools]
        underloaded = [s.agent_id for s in stats if s.tool_count < self.min_tools]

        gini = self._gini_coefficient([s.tool_count for s in stats])
        balance_score = 1.0 - gini

        recommended_n = self._recommend_n_agents(stats)

        report = BalanceReport(
            agent_tool_counts=counts,
            overloaded_agents=overloaded,
            underloaded_agents=underloaded,
            gini_coefficient=gini,
            recommended_n_agents=recommended_n,
            current_max_tools=self.max_tools,
            rebalanced=False,
            balance_score=balance_score,
            summary=self._build_summary(stats, overloaded, underloaded, gini),
        )

        if overloaded:
            logger.warning(
                f"오버로드 Agent 탐지: {overloaded} "
                f"(기준: {self.max_tools}개 초과)"
            )
        else:
            logger.info(
                f"Tool 분배 균형 양호 (Gini: {gini:.3f}, "
                f"분포: {counts})"
            )

        return report

    # ── Agile 재분배 ──────────────────────────────────────────────────────────

    def rebalance(
        self,
        agents: list[DiscoveredAgent],
        features: list[TopologicalFeature],
        mcp_tools: list[MCPToolSchema],
    ) -> tuple[list[DiscoveredAgent], list[MCPToolSchema], BalanceReport]:
        """
        Agile Threshold 조정 알고리즘으로 Tool 재분배.

        1. 오버로드 에이전트에 대해 n_agents+1 재클러스터링 시도
        2. 3회 반복 후에도 미수렴 시 강제 분리

        Returns:
            (재분배된 agents, 재할당된 mcp_tools, BalanceReport)
        """
        report = self.analyze(agents, mcp_tools)
        if not report.overloaded_agents:
            logger.info("재분배 불필요: 모든 Agent가 임계값 이하입니다.")
            return agents, mcp_tools, report

        logger.info(
            f"Agile 재분배 시작: {len(report.overloaded_agents)}개 오버로드 Agent"
        )

        current_agents = list(agents)
        current_tools = list(mcp_tools)
        n_iterations = 0

        for iteration in range(MAX_REBALANCE_ITERATIONS):
            n_iterations += 1
            current_report = self.analyze(current_agents, current_tools)

            if not current_report.overloaded_agents:
                logger.info(f"재분배 수렴: {iteration + 1}회 반복")
                break

            logger.info(f"재분배 반복 {iteration + 1}/{MAX_REBALANCE_ITERATIONS}")

            # 오버로드 에이전트를 분리
            new_agents: list[DiscoveredAgent] = []
            split_map: dict[str, list[str]] = {}  # old_id → [new_id1, new_id2]

            for agent in current_agents:
                if agent.agent_id in current_report.overloaded_agents:
                    # 해당 에이전트 분리
                    agent_tools = self._get_agent_tools(agent, current_tools)
                    split_agents = self._split_overloaded_agent(
                        agent, features, agent_tools
                    )
                    new_agents.extend(split_agents)
                    split_map[agent.agent_id] = [a.agent_id for a in split_agents]
                    logger.info(
                        f"  '{agent.suggested_name or agent.agent_id}' "
                        f"→ {len(split_agents)}개 분리 "
                        f"({[a.suggested_name or a.agent_id for a in split_agents]})"
                    )
                else:
                    new_agents.append(agent)

            # Tool 재할당
            current_tools = self._reassign_tools(new_agents, current_tools, split_map)
            current_agents = new_agents

        # 최종 보고서 생성
        final_report = self.analyze(current_agents, current_tools)
        final_report.rebalanced = True
        final_report.rebalance_iterations = n_iterations
        final_report.current_max_tools = self.max_tools

        logger.info(
            f"재분배 완료: {len(agents)}개 → {len(current_agents)}개 Agent, "
            f"Gini: {report.gini_coefficient:.3f} → {final_report.gini_coefficient:.3f}"
        )
        return current_agents, current_tools, final_report

    # ── 에이전트 분리 ─────────────────────────────────────────────────────────

    def _split_overloaded_agent(
        self,
        agent: DiscoveredAgent,
        features: list[TopologicalFeature],
        agent_tools: list[MCPToolSchema],
    ) -> list[DiscoveredAgent]:
        """
        오버로드 에이전트를 2개의 서브 에이전트로 분리합니다.

        해당 에이전트의 태스크에 2차 KMeans 클러스터링을 적용합니다.
        """
        # 에이전트 태스크의 특징 벡터 수집
        feature_map = {f.task_id: f for f in features}
        agent_features = [
            feature_map[tid]
            for tid in agent.task_ids
            if tid in feature_map
        ]

        if len(agent_features) < 2:
            # 분리 불가 → 원본 반환
            return [agent]

        # 2분할 KMeans
        matrix = np.stack([f.vector for f in agent_features])
        n_split = min(2, len(agent_features))
        kmeans = KMeans(n_clusters=n_split, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix)

        split_agents: list[DiscoveredAgent] = []
        suffix_labels = ["A", "B", "C"]
        base_name = agent.suggested_name or agent.agent_id

        for k in range(n_split):
            mask = labels == k
            sub_features = [f for f, m in zip(agent_features, mask) if m]
            if not sub_features:
                continue

            sub_centroid = matrix[mask].mean(axis=0)
            sub_agent = DiscoveredAgent(
                agent_id=f"{agent.agent_id}_split_{k}",
                cluster_id=agent.cluster_id * 10 + k,
                task_ids=[f.task_id for f in sub_features],
                task_names=[f.task_name for f in sub_features],
                centroid=sub_centroid,
                suggested_name=f"{base_name} {suffix_labels[k]}",
                suggested_role=agent.suggested_role,
                suggested_capabilities=agent.suggested_capabilities,
                routing_group_id=agent.routing_group_id or agent.agent_id,
            )
            split_agents.append(sub_agent)

        return split_agents if split_agents else [agent]

    # ── Tool 재할당 ───────────────────────────────────────────────────────────

    def _reassign_tools(
        self,
        agents: list[DiscoveredAgent],
        tools: list[MCPToolSchema],
        split_map: dict[str, list[str]],
    ) -> list[MCPToolSchema]:
        """
        분리된 에이전트에 맞게 Tool의 assigned_agent를 재할당합니다.
        """
        # agent_id → task_id 집합 매핑
        agent_task_sets = {
            agent.agent_id: set(agent.task_ids) for agent in agents
        }

        reassigned: list[MCPToolSchema] = []
        for tool in tools:
            if tool.annotations is None:
                reassigned.append(tool)
                continue

            source_task = tool.annotations.source_task_id
            # 이 태스크를 가진 에이전트 탐색
            new_agent_id = None
            for agent_id, task_ids in agent_task_sets.items():
                if source_task in task_ids:
                    new_agent_id = agent_id
                    break

            if new_agent_id and new_agent_id != tool.annotations.assigned_agent:
                # 새 에이전트로 재할당
                new_tool = tool.model_copy(deep=True)
                new_tool.annotations.assigned_agent = new_agent_id
                reassigned.append(new_tool)
            else:
                reassigned.append(tool)

        return reassigned

    # ── 통계 유틸리티 ─────────────────────────────────────────────────────────

    def _compute_stats(
        self,
        agents: list[DiscoveredAgent],
        mcp_tools: list[MCPToolSchema],
    ) -> list[_AgentToolStats]:
        """에이전트별 Tool 통계 계산"""
        # tool의 assigned_agent로 카운트
        agent_tool_map: dict[str, list[str]] = {a.agent_id: [] for a in agents}

        for tool in mcp_tools:
            if tool.annotations and tool.annotations.assigned_agent in agent_tool_map:
                agent_tool_map[tool.annotations.assigned_agent].append(tool.name)

        stats = []
        for agent in agents:
            tool_names = agent_tool_map.get(agent.agent_id, [])
            stats.append(_AgentToolStats(
                agent_id=agent.agent_id,
                agent_name=agent.suggested_name or agent.agent_id,
                tool_count=len(tool_names),
                tool_names=tool_names,
            ))
        return stats

    @staticmethod
    def _get_agent_tools(
        agent: DiscoveredAgent,
        mcp_tools: list[MCPToolSchema],
    ) -> list[MCPToolSchema]:
        """특정 에이전트에 할당된 Tool 목록 반환"""
        return [
            t for t in mcp_tools
            if t.annotations and t.annotations.assigned_agent == agent.agent_id
        ]

    @staticmethod
    def _gini_coefficient(counts: list[int]) -> float:
        """
        Gini 계수 계산 (불균형 지수).

        0 = 완전 균등 분배
        1 = 완전 불균등 (한 에이전트에 모든 Tool 집중)
        """
        if not counts or sum(counts) == 0:
            return 0.0
        n = len(counts)
        arr = np.array(counts, dtype=float)
        arr_sorted = np.sort(arr)
        idx = np.arange(1, n + 1)
        gini = (2 * np.sum(idx * arr_sorted) / (n * np.sum(arr_sorted))) - (n + 1) / n
        return float(np.clip(gini, 0.0, 1.0))

    @staticmethod
    def _recommend_n_agents(stats: list[_AgentToolStats]) -> int:
        """
        최적 에이전트 수 권장값 계산.

        총 Tool 수 / MAX_TOOLS_PER_AGENT 를 반올림한 값.
        """
        total_tools = sum(s.tool_count for s in stats)
        if total_tools == 0:
            return len(stats)
        return max(len(stats), int(np.ceil(total_tools / DEFAULT_MAX_TOOLS)))

    @staticmethod
    def _build_summary(
        stats: list[_AgentToolStats],
        overloaded: list[str],
        underloaded: list[str],
        gini: float,
    ) -> str:
        """분석 요약 메시지 생성"""
        total = sum(s.tool_count for s in stats)
        avg = total / len(stats) if stats else 0
        max_count = max((s.tool_count for s in stats), default=0)
        min_count = min((s.tool_count for s in stats), default=0)

        status = "✅ 균형" if not overloaded else f"⚠️ {len(overloaded)}개 오버로드"
        return (
            f"{status} | "
            f"총 {total}개 Tool, {len(stats)}개 Agent | "
            f"평균 {avg:.1f}개/Agent (최대 {max_count}, 최소 {min_count}) | "
            f"Gini={gini:.3f}"
        )
