"""
Hierarchy Planner

Master Agent - Orchestrator Agent - Unit Agent 계층을 구성하고,
쿼리마다 단순/복합 경로를 선택해 실행 계획을 생성합니다.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from tad_mapper.engine.tda_analyzer import DiscoveredAgent
from tad_mapper.models.topology import (
    HierarchicalRoutingPlan,
    HierarchyBlueprint,
    HierarchyNode,
    RoutingResult,
    SubTaskAssignment,
)

_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]{2,}")
_SPLIT_PATTERN = re.compile(
    r"\s*(?:그리고|및|,|;| then | and | 후에 | 다음으로 )\s*",
    flags=re.IGNORECASE,
)


class HierarchyPlanner:
    """
    계층형 라우팅 플래너.

    - build(): Unit Agent 집합에서 Orchestrator 계층 자동 구성
    - plan(): Master의 경로 선택 + Orchestrator의 서브태스크 분해/배정
    """

    def __init__(
        self,
        simple_threshold: float = 0.45,
        max_orchestrators: int = 2,
        min_orchestrator_prob: float = 0.12,
    ) -> None:
        self._simple_threshold = float(np.clip(simple_threshold, 0.0, 1.0))
        self._max_orchestrators = max(1, int(max_orchestrators))
        self._min_orchestrator_prob = float(np.clip(min_orchestrator_prob, 0.0, 1.0))

        self._blueprint = HierarchyBlueprint()
        self._agents_by_id: dict[str, DiscoveredAgent] = {}
        self._unit_tokens: dict[str, set[str]] = {}
        self._orchestrator_tokens: dict[str, set[str]] = {}
        self._unit_task_count: dict[str, int] = {}

    def build(self, agents: list[DiscoveredAgent]) -> HierarchyBlueprint:
        """Unit Agent 목록에서 계층 구조를 생성합니다."""
        self._agents_by_id = {a.agent_id: a for a in agents}
        self._unit_tokens = {}
        self._orchestrator_tokens = {}
        self._unit_task_count = {a.agent_id: len(a.task_ids) for a in agents}

        nodes: list[HierarchyNode] = []
        master = HierarchyNode(
            node_id="master_agent",
            tier="MASTER",
            name="Master Agent",
            description="전략적 분배/감독",
        )
        nodes.append(master)

        unit_to_orchestrator: dict[str, str] = {}
        orchestrator_to_units: dict[str, list[str]] = {}

        grouped: dict[str, list[DiscoveredAgent]] = defaultdict(list)
        for agent in agents:
            group_id = agent.routing_group_id or agent.agent_id
            grouped[group_id].append(agent)

        for group_id in sorted(grouped):
            members = sorted(grouped[group_id], key=lambda a: a.agent_id)

            if len(members) == 1:
                unit = members[0]
                master.child_ids.append(unit.agent_id)
                nodes.append(
                    HierarchyNode(
                        node_id=unit.agent_id,
                        tier="UNIT",
                        name=unit.suggested_name or unit.agent_id,
                        parent_id=master.node_id,
                        description=unit.suggested_role or "단일 기능 실행",
                    )
                )
                unit_to_orchestrator[unit.agent_id] = ""
                self._unit_tokens[unit.agent_id] = self._agent_tokens(unit)
                continue

            orch_id = f"orch_{group_id}"
            orch_name = self._derive_orchestrator_name(group_id, members)
            master.child_ids.append(orch_id)

            unit_ids = [m.agent_id for m in members]
            orchestrator_to_units[orch_id] = unit_ids
            nodes.append(
                HierarchyNode(
                    node_id=orch_id,
                    tier="ORCHESTRATOR",
                    name=orch_name,
                    parent_id=master.node_id,
                    child_ids=unit_ids.copy(),
                    unit_agent_ids=unit_ids.copy(),
                    description="복합 태스크 분해 및 Unit 배정",
                )
            )
            self._orchestrator_tokens[orch_id] = self._orchestrator_text_tokens(members)

            for unit in members:
                unit_to_orchestrator[unit.agent_id] = orch_id
                nodes.append(
                    HierarchyNode(
                        node_id=unit.agent_id,
                        tier="UNIT",
                        name=unit.suggested_name or unit.agent_id,
                        parent_id=orch_id,
                        description=unit.suggested_role or "단일 기능 실행",
                    )
                )
                self._unit_tokens[unit.agent_id] = self._agent_tokens(unit)

        self._blueprint = HierarchyBlueprint(
            master_id=master.node_id,
            nodes=nodes,
            unit_to_orchestrator=unit_to_orchestrator,
            orchestrator_to_units=orchestrator_to_units,
        )
        return self._blueprint

    def plan(
        self,
        query_text: str,
        routing: RoutingResult,
    ) -> HierarchicalRoutingPlan:
        """Master의 경로 선택 + Orchestrator의 서브태스크 분해 계획 생성."""
        subtasks = self._split_subtasks(query_text)
        complexity = self._complexity_score(query_text, routing, subtasks)
        target_orch = self._blueprint.unit_to_orchestrator.get(
            routing.target_agent_id, ""
        )

        use_orchestrator = bool(
            target_orch
            and (
                complexity >= self._simple_threshold
                or len(subtasks) > 1
                or routing.is_ambiguous
            )
        )

        if not use_orchestrator:
            assignments = [
                SubTaskAssignment(
                    subtask_id=f"subtask_{i}",
                    subtask_text=subtask,
                    unit_agent_id=routing.target_agent_id,
                    score=float(np.clip(routing.confidence, 0.0, 1.0)),
                    reason="단순 경로 선택 (Master → Unit)",
                )
                for i, subtask in enumerate(subtasks, start=1)
            ]
            return HierarchicalRoutingPlan(
                query_text=query_text,
                path_type="master_unit",
                routing=routing,
                complexity_score=complexity,
                complexity_threshold=self._simple_threshold,
                selected_unit_ids=[routing.target_agent_id],
                subtasks=subtasks,
                assignments=assignments,
                rationale=[
                    (
                        f"complexity={complexity:.3f} < "
                        f"threshold={self._simple_threshold:.3f}"
                    ),
                    "고신뢰/저복잡 쿼리는 Master가 Unit에 직접 위임",
                ],
            )

        orch_probs = self._aggregate_orchestrator_probabilities(routing, target_orch)
        selected_orchestrators = self._select_orchestrators(orch_probs, target_orch)

        assignments: list[SubTaskAssignment] = []
        selected_units: list[str] = []
        for i, subtask in enumerate(subtasks, start=1):
            orch_id = self._choose_orchestrator(subtask, selected_orchestrators, orch_probs)
            unit_id, score, reason = self._choose_unit(subtask, orch_id, routing)
            assignments.append(
                SubTaskAssignment(
                    subtask_id=f"subtask_{i}",
                    subtask_text=subtask,
                    orchestrator_id=orch_id,
                    unit_agent_id=unit_id,
                    score=score,
                    reason=reason,
                )
            )
            if unit_id not in selected_units:
                selected_units.append(unit_id)

        return HierarchicalRoutingPlan(
            query_text=query_text,
            path_type="master_orchestrator_unit",
            routing=routing,
            complexity_score=complexity,
            complexity_threshold=self._simple_threshold,
            orchestrator_probabilities=orch_probs,
            selected_orchestrator_ids=selected_orchestrators,
            selected_unit_ids=selected_units,
            subtasks=subtasks,
            assignments=assignments,
            rationale=[
                (
                    f"complexity={complexity:.3f} >= "
                    f"threshold={self._simple_threshold:.3f}"
                ),
                "복합/모호 쿼리는 Master가 Orchestrator에 분해 위임",
                "Orchestrator가 서브태스크별 Unit을 배정",
            ],
        )

    @property
    def blueprint(self) -> HierarchyBlueprint:
        return self._blueprint

    def _complexity_score(
        self,
        query_text: str,
        routing: RoutingResult,
        subtasks: list[str],
    ) -> float:
        tokens = self._tokenize(query_text)
        uncertainty = 1.0 - float(np.clip(routing.confidence, 0.0, 1.0))
        ambiguous = 1.0 if routing.is_ambiguous else 0.0
        multi_intent = float(np.clip((len(subtasks) - 1) / 3.0, 0.0, 1.0))
        long_query = float(np.clip(len(tokens) / 18.0, 0.0, 1.0))

        probs = sorted(routing.routing_probabilities.values(), reverse=True)
        if len(probs) >= 2:
            prob_gap = float(np.clip(probs[0] - probs[1], 0.0, 1.0))
        else:
            prob_gap = 0.0
        low_prob_gap = 1.0 - prob_gap

        score = (
            0.35 * uncertainty
            + 0.20 * ambiguous
            + 0.25 * multi_intent
            + 0.10 * long_query
            + 0.10 * low_prob_gap
        )
        return float(np.clip(score, 0.0, 1.0))

    def _aggregate_orchestrator_probabilities(
        self,
        routing: RoutingResult,
        target_orchestrator: str,
    ) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        for unit_id, prob in routing.routing_probabilities.items():
            orch_id = self._blueprint.unit_to_orchestrator.get(unit_id, "")
            if orch_id:
                scores[orch_id] += float(prob)

        if not scores and target_orchestrator:
            scores[target_orchestrator] = 1.0

        total = sum(scores.values())
        if total <= 0:
            return {}
        return {k: float(v / total) for k, v in scores.items()}

    def _select_orchestrators(
        self,
        orchestrator_probs: dict[str, float],
        target_orchestrator: str,
    ) -> list[str]:
        ranked = sorted(
            orchestrator_probs.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        selected = [
            orch_id
            for orch_id, prob in ranked
            if prob >= self._min_orchestrator_prob
        ][: self._max_orchestrators]

        if not selected and ranked:
            selected = [ranked[0][0]]
        if target_orchestrator and target_orchestrator not in selected:
            selected = [target_orchestrator, *selected]
            selected = selected[: self._max_orchestrators]
        return selected

    def _choose_orchestrator(
        self,
        subtask_text: str,
        orchestrators: list[str],
        orchestrator_probs: dict[str, float],
    ) -> str:
        if not orchestrators:
            return ""
        if len(orchestrators) == 1:
            return orchestrators[0]

        subtask_tokens = self._tokenize(subtask_text)
        scored: list[tuple[str, float]] = []
        for orch_id in orchestrators:
            lex = self._lexical_similarity(
                subtask_tokens, self._orchestrator_tokens.get(orch_id, set())
            )
            prob = orchestrator_probs.get(orch_id, 0.0)
            score = 0.6 * prob + 0.4 * lex
            scored.append((orch_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def _choose_unit(
        self,
        subtask_text: str,
        orchestrator_id: str,
        routing: RoutingResult,
    ) -> tuple[str, float, str]:
        candidates = self._blueprint.orchestrator_to_units.get(orchestrator_id, [])
        if not candidates:
            return (
                routing.target_agent_id,
                float(np.clip(routing.confidence, 0.0, 1.0)),
                "후보 Unit이 없어 라우팅 타겟 Unit으로 폴백",
            )

        subtask_tokens = self._tokenize(subtask_text)
        scored: list[tuple[str, float, float, float]] = []
        for unit_id in candidates:
            prob = float(routing.routing_probabilities.get(unit_id, 0.0))
            lex = self._lexical_similarity(subtask_tokens, self._unit_tokens.get(unit_id, set()))
            score = 0.65 * prob + 0.35 * lex
            scored.append((unit_id, score, prob, lex))
        scored.sort(key=lambda x: x[1], reverse=True)

        unit_id, score, prob, lex = scored[0]
        if score <= 0.0:
            # 증거가 약하면 태스크 수가 많은 Unit을 허브로 사용
            hub_id = max(candidates, key=lambda uid: self._unit_task_count.get(uid, 0))
            hub_score = 0.2 + 0.8 * float(
                self._unit_task_count.get(hub_id, 0) / max(1, sum(self._unit_task_count.get(uid, 0) for uid in candidates))
            )
            return hub_id, float(np.clip(hub_score, 0.0, 1.0)), "증거 약함: 허브 Unit 우선"

        reason = f"prob={prob:.3f}, lexical={lex:.3f}"
        return unit_id, float(np.clip(score, 0.0, 1.0)), reason

    @staticmethod
    def _derive_orchestrator_name(
        group_id: str,
        members: list[DiscoveredAgent],
    ) -> str:
        base = members[0].suggested_name or group_id
        base = base.replace("에이전트", "").strip()
        return f"{base} 오케스트레이터"

    def _orchestrator_text_tokens(self, members: list[DiscoveredAgent]) -> set[str]:
        texts = []
        for member in members:
            texts.append(member.suggested_name or member.agent_id)
            texts.append(member.suggested_role or "")
            texts.extend(member.task_names[:8])
            texts.extend((member.suggested_capabilities or [])[:6])
        return self._tokenize(" ".join(texts))

    def _agent_tokens(self, agent: DiscoveredAgent) -> set[str]:
        texts = [
            agent.suggested_name or agent.agent_id,
            agent.suggested_role or "",
            " ".join(agent.task_names[:12]),
            " ".join((agent.suggested_capabilities or [])[:8]),
        ]
        return self._tokenize(" ".join(texts))

    @staticmethod
    def _split_subtasks(query_text: str) -> list[str]:
        pieces = [p.strip() for p in _SPLIT_PATTERN.split(query_text or "") if p.strip()]
        return pieces or [query_text.strip()]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token.lower() for token in _TOKEN_PATTERN.findall(text or "")}

    @staticmethod
    def _lexical_similarity(a_tokens: set[str], b_tokens: set[str]) -> float:
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens & b_tokens)
        if inter <= 0:
            return 0.0
        union = len(a_tokens | b_tokens)
        if union <= 0:
            return 0.0
        recall = inter / max(1, len(a_tokens))
        jaccard = inter / union
        return float(np.clip(0.8 * recall + 0.2 * jaccard, 0.0, 1.0))
