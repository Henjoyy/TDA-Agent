"""Unit Agent 데이터 모델"""
from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureBounds(BaseModel):
    """에이전트 특징 벡터 경계값"""
    reasoning_depth: list[float] | None = None
    data_type: list[float] | None = None
    risk_level: list[float] | None = None
    autonomy: list[float] | None = None
    data_sensitivity: list[float] | None = None
    complexity: list[float] | None = None


class UnitAgentDefinition(BaseModel):
    """Unit Agent 정의"""
    id: str
    name: str
    name_ko: str
    description: str
    capabilities: list[str]
    feature_bounds: FeatureBounds


class AgentAssignment(BaseModel):
    """태스크 → 에이전트 할당 결과"""
    task_id: str
    task_name: str
    agent_id: str
    agent_name: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="할당 신뢰도 (0~1)")
    cluster_id: int = Field(default=-1, description="Mapper 클러스터 ID")
    reasoning: str = Field(default="", description="할당 근거 설명")


class HoleWarning(BaseModel):
    """논리적 구멍(Hole) 경고"""
    hole_type: str  # "unassigned" | "connectivity" | "coverage_gap"
    affected_tasks: list[str]
    description: str
    suggestion: str


class OverlapWarning(BaseModel):
    """중복 구간(Overlap) 경고"""
    task_id: str
    candidate_agents: list[str]
    description: str


class MappingResult(BaseModel):
    """전체 매핑 결과"""
    journey_id: str
    journey_title: str
    assignments: list[AgentAssignment]
    holes: list[HoleWarning] = Field(default_factory=list)
    overlaps: list[OverlapWarning] = Field(default_factory=list)
    agent_summary: dict[str, list[str]] = Field(
        default_factory=dict,
        description="에이전트별 담당 태스크 ID 목록"
    )

    def build_agent_summary(self) -> None:
        """assignments에서 agent_summary 생성"""
        summary: dict[str, list[str]] = {}
        for a in self.assignments:
            summary.setdefault(a.agent_id, []).append(a.task_id)
        self.agent_summary = summary
