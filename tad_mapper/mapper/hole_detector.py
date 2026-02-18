"""
Hole Detector - 위상학적 구멍(논리적 공백) 탐지
"""
from __future__ import annotations

import logging

import numpy as np

from tad_mapper.engine.tda_analyzer import DiscoveredAgent, MapperGraph
from tad_mapper.models.agent import HoleWarning, OverlapWarning
from tad_mapper.models.journey import UserJourney

logger = logging.getLogger(__name__)


class HoleDetector:
    """
    위상 공간에서 논리적 구멍과 중복 구간을 탐지합니다.

    탐지 유형:
    1. Unassigned Hole: 어떤 Agent에도 속하지 않는 태스크
    2. Connectivity Hole: 의존성 체인이 끊긴 구간
    3. Overlap Warning: 여러 Agent에 중복 할당된 태스크
    """

    def detect_holes(
        self,
        journey: UserJourney,
        agents: list[DiscoveredAgent],
        graph: MapperGraph,
    ) -> list[HoleWarning]:
        """모든 유형의 Hole 탐지"""
        holes: list[HoleWarning] = []
        holes.extend(self._detect_unassigned(journey, agents))
        holes.extend(self._detect_connectivity_holes(journey, agents))
        return holes

    def detect_overlaps(
        self,
        agents: list[DiscoveredAgent],
        graph: MapperGraph,
    ) -> list[OverlapWarning]:
        """중복 할당 탐지 (여러 Agent의 Mapper 노드에 동시 포함된 태스크)"""
        overlaps: list[OverlapWarning] = []

        # task_id → 속한 agent_id 목록
        task_agents: dict[str, list[str]] = {}
        for agent in agents:
            for tid in agent.task_ids:
                task_agents.setdefault(tid, []).append(agent.agent_id)

        for tid, agent_ids in task_agents.items():
            if len(agent_ids) > 1:
                # 태스크 이름 찾기
                task_name = next(
                    (name for a in agents for i, name in zip(a.task_ids, a.task_names) if i == tid),
                    tid
                )
                overlaps.append(OverlapWarning(
                    task_id=tid,
                    candidate_agents=agent_ids,
                    description=(
                        f"태스크 '{task_name}'이(가) {len(agent_ids)}개 Agent에 중복 할당되었습니다: "
                        f"{', '.join(agent_ids)}"
                    ),
                ))

        if overlaps:
            logger.warning(f"중복 할당 탐지: {len(overlaps)}개 태스크")
        return overlaps

    def _detect_unassigned(
        self,
        journey: UserJourney,
        agents: list[DiscoveredAgent],
    ) -> list[HoleWarning]:
        """어떤 Agent에도 할당되지 않은 태스크 탐지"""
        assigned_ids = {tid for agent in agents for tid in agent.task_ids}
        all_ids = {step.id for step in journey.steps}
        unassigned = all_ids - assigned_ids

        if not unassigned:
            return []

        unassigned_names = [
            step.name for step in journey.steps if step.id in unassigned
        ]
        return [HoleWarning(
            hole_type="unassigned",
            affected_tasks=list(unassigned),
            description=(
                f"{len(unassigned)}개 태스크가 어떤 Agent에도 할당되지 않았습니다: "
                f"{', '.join(unassigned_names)}"
            ),
            suggestion=(
                "해당 태스크의 설명을 보완하거나, 새로운 Agent 유형을 추가하는 것을 검토하세요."
            ),
        )]

    def _detect_connectivity_holes(
        self,
        journey: UserJourney,
        agents: list[DiscoveredAgent],
    ) -> list[HoleWarning]:
        """의존성 체인에서 Agent 간 연결이 끊긴 구간 탐지"""
        holes: list[HoleWarning] = []

        # task_id → agent_id 매핑
        task_to_agent: dict[str, str] = {}
        for agent in agents:
            for tid in agent.task_ids:
                task_to_agent[tid] = agent.agent_id

        # 의존성 체인 검사
        for step in journey.steps:
            if not step.dependencies:
                continue
            current_agent = task_to_agent.get(step.id)
            for dep_id in step.dependencies:
                dep_agent = task_to_agent.get(dep_id)
                if dep_agent and current_agent and dep_agent != current_agent:
                    # 다른 Agent 간 데이터 전달 → 인터페이스 필요 (경고)
                    dep_step = journey.get_step_by_id(dep_id)
                    dep_name = dep_step.name if dep_step else dep_id
                    holes.append(HoleWarning(
                        hole_type="connectivity",
                        affected_tasks=[dep_id, step.id],
                        description=(
                            f"'{dep_name}'({dep_agent}) → '{step.name}'({current_agent}) "
                            f"간 Agent 경계를 넘는 데이터 전달이 필요합니다."
                        ),
                        suggestion=(
                            f"{dep_agent}의 출력을 {current_agent}의 입력으로 연결하는 "
                            "MCP Tool 또는 메시지 버스 인터페이스를 설계하세요."
                        ),
                    ))

        return holes
