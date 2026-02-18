"""
수학적 정식화 기반 위상 데이터 모델

1. QueryPoint     — 사용자 쿼리 q ∈ Q (User Query Manifold)
2. HomotopyClass  — 동치류 [x] (호모토피 동치 쿼리 집합)
3. CoverageRegion — 에이전트 커버리지 영역 Ui
4. CoverageMetrics — Q ⊆ ∪Ui 커버리지 측정값
5. RoutingResult  — Φ(x) = Uk 라우팅 결과
6. ToolExecutionStep / CompositionPlan — 툴 합성 y = (t_π(m) ∘ ... ∘ t_π(1))(x)
7. BalanceReport  — MCP Tool 부하 균형 분석
"""
from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────────────
# 1. Query Manifold
# ─────────────────────────────────────────────────────────────────────────────

class QueryPoint(BaseModel):
    """사용자 쿼리 q ∈ Q — 쿼리 매니폴드 위의 한 점"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    embedding: list[float] = Field(default_factory=list, description="768차원 임베딩 벡터")
    homotopy_class_id: str = Field(default="", description="분류된 호모토피 클래스 ID")

    def embedding_array(self) -> np.ndarray:
        return np.array(self.embedding)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Homotopy Class
# ─────────────────────────────────────────────────────────────────────────────

class HomotopyClass(BaseModel):
    """
    호모토피 동치류 [x]

    표현은 달라도 본질적 의도가 같은 쿼리들의 집합.
    e.g. "자료 찾아줘", "데이터 검색해", "정보 줘" → 동일 클래스
    """
    class_id: str                              # "class_agent_0" 등
    agent_id: str                              # 해당 클래스를 처리하는 Agent
    agent_name: str = ""
    representative_text: str = ""              # 이 클래스의 대표 설명
    centroid_embedding: list[float] = Field(default_factory=list)
    radius: float = Field(default=0.3, description="클래스 반경 (코사인 거리 기준)")

    def centroid_array(self) -> np.ndarray:
        return np.array(self.centroid_embedding)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Agent Coverage Region
# ─────────────────────────────────────────────────────────────────────────────

class CoverageRegion(BaseModel):
    """
    Unit Agent의 커버리지 영역 Ui

    Q ⊆ ∪ Ui 를 구성하는 개별 열린 집합.
    """
    agent_id: str
    agent_name: str = ""
    centroid_embedding: list[float] = Field(default_factory=list)
    task_embeddings: list[list[float]] = Field(default_factory=list)
    radius: float = Field(default=0.0, description="커버리지 반경 (최대 태스크 거리)")
    voronoi_area: float = Field(default=0.0, description="Voronoi 셀 면적 (2D 투영)")
    projected_centroid_2d: list[float] = Field(default_factory=list, description="PCA 2D 투영 중심")

    def centroid_array(self) -> np.ndarray:
        return np.array(self.centroid_embedding)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Coverage Metrics
# ─────────────────────────────────────────────────────────────────────────────

class CoverageMetrics(BaseModel):
    """
    Q ⊆ ∪ Ui 커버리지 측정값

    coverage_ratio = 1.0 이면 완전 피복(Open Cover 조건 충족)
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    coverage_ratio: float = Field(..., ge=0.0, le=1.0, description="전체 공간 대비 커버된 비율")
    overlap_ratio: float = Field(..., ge=0.0, le=1.0, description="Agent 간 중첩 비율")
    gap_ratio: float = Field(..., ge=0.0, le=1.0, description="미커버 영역 비율")
    agent_coverage_areas: dict[str, float] = Field(
        default_factory=dict, description="에이전트별 커버리지 면적"
    )
    uncovered_task_ids: list[str] = Field(default_factory=list, description="커버되지 않은 태스크")
    overlap_agent_pairs: list[tuple[str, str]] = Field(
        default_factory=list, description="중첩된 에이전트 쌍"
    )
    coverage_complete: bool = Field(default=False, description="Q ⊆ ∪Ui 조건 충족 여부")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Routing Result
# ─────────────────────────────────────────────────────────────────────────────

class RoutingCandidate(BaseModel):
    """라우팅 후보 (대안 Agent)"""
    agent_id: str
    agent_name: str = ""
    similarity: float
    homotopy_class_id: str = ""


class RoutingResult(BaseModel):
    """
    Φ(x) = Uk 라우팅 결과

    Master Agent가 쿼리 x를 분석하여 적절한 Unit Agent Uk로 라우팅.
    """
    query_text: str
    target_agent_id: str
    target_agent_name: str = ""
    homotopy_class_id: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="라우팅 신뢰도")
    top_similarity: float = Field(..., ge=0.0, le=1.0)
    alternatives: list[RoutingCandidate] = Field(
        default_factory=list, description="Top-2~3 대안 Agent"
    )
    is_ambiguous: bool = Field(default=False, description="신뢰도 낮아 모호한 경우")
    ambiguity_reason: str = Field(default="", description="모호성 원인 설명")

    def to_dict(self) -> dict:
        return self.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Tool Composition Plan
# ─────────────────────────────────────────────────────────────────────────────

class ToolExecutionStep(BaseModel):
    """
    합성 체인의 한 단계 t_π(k)

    y = (t_π(m) ∘ ... ∘ t_π(2) ∘ t_π(1))(x)
    """
    order: int                                  # π(k) — 실행 순서 (1부터 시작)
    tool_name: str
    tool_description: str = ""
    input_from: str = Field(
        default="query",
        description="입력 출처: 'query'(사용자 입력) 또는 이전 단계 tool_name"
    )
    output_to: str = Field(
        default="next",
        description="출력 목적지: 'next'(다음 단계) 또는 'final'(최종 결과)"
    )
    input_params: dict[str, Any] = Field(
        default_factory=dict, description="이 단계에 전달할 파라미터"
    )
    rationale: str = Field(default="", description="이 단계가 필요한 이유")


class CompositionPlan(BaseModel):
    """
    MCP Tool 합성 실행 계획

    특정 Agent가 쿼리를 처리하기 위해 MCP Tool을 어떤 순서로 합성할지 정의.
    y = (t_π(m) ∘ ... ∘ t_π(1))(x)
    """
    query_text: str
    agent_id: str
    agent_name: str = ""
    steps: list[ToolExecutionStep] = Field(default_factory=list)
    total_steps: int = 0
    estimated_output: str = Field(default="", description="예상 최종 출력 형태")
    dependency_graph: dict[str, list[str]] = Field(
        default_factory=dict, description="tool_name → 의존 tool_name 목록"
    )
    is_valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self.total_steps = len(self.steps)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Tool Balance Report
# ─────────────────────────────────────────────────────────────────────────────

class BalanceReport(BaseModel):
    """
    MCP Tool 에이전트 간 부하 균형 분석 결과

    하나의 Agent에 Tool이 너무 많으면 LLM 성능이 저하되므로
    Agile하게 threshold를 조정하여 재분배합니다.
    """
    agent_tool_counts: dict[str, int] = Field(
        default_factory=dict, description="agent_id → MCP Tool 수"
    )
    overloaded_agents: list[str] = Field(
        default_factory=list, description="MAX_TOOLS 초과 에이전트"
    )
    underloaded_agents: list[str] = Field(
        default_factory=list, description="MIN_TOOLS 미만 에이전트"
    )
    gini_coefficient: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="툴 수 분배 불균형 지수 (0=완전 균등, 1=완전 불균등)"
    )
    recommended_n_agents: int = Field(
        default=0, description="최적 에이전트 수 권장값"
    )
    current_max_tools: int = Field(default=7, description="현재 적용 중인 임계값")
    rebalanced: bool = Field(default=False, description="재분배 실행 여부")
    rebalance_iterations: int = Field(default=0, description="재분배 반복 횟수")
    balance_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="균형 점수 (1=완벽 균등, 0=극심한 불균형)"
    )
    summary: str = Field(default="", description="분석 요약 메시지")
