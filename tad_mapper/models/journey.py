"""User Journey 데이터 모델"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TaskStep(BaseModel):
    """사용자 여정의 개별 단계(태스크)"""

    id: str = Field(..., description="태스크 고유 ID (예: 'task_001')")
    name: str = Field(..., description="태스크 이름 (예: '수출량 통계 조회')")
    description: str = Field(..., description="태스크 상세 설명")
    actor: str = Field(default="user", description="수행 주체 (예: 'user', '기획자', 'system')")
    input_data: list[str] = Field(default_factory=list, description="입력 데이터 유형 목록")
    output_data: list[str] = Field(default_factory=list, description="출력 데이터 유형 목록")
    dependencies: list[str] = Field(default_factory=list, description="선행 태스크 ID 목록")
    tags: list[str] = Field(default_factory=list, description="분류 태그")

    @field_validator("id")
    @classmethod
    def id_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("태스크 ID는 비어있을 수 없습니다.")
        return v.strip()

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("태스크 이름은 비어있을 수 없습니다.")
        return v.strip()


class UserJourney(BaseModel):
    """전체 사용자 여정"""

    id: str = Field(..., description="여정 고유 ID")
    title: str = Field(..., description="여정 제목")
    domain: str = Field(default="general", description="도메인 (예: '무역', '금융', '의료')")
    description: str = Field(default="", description="여정 전체 설명")
    steps: list[TaskStep] = Field(..., description="태스크 단계 목록", min_length=1)
    metadata: dict = Field(default_factory=dict, description="추가 메타데이터")

    @field_validator("steps")
    @classmethod
    def steps_must_have_unique_ids(cls, v: list[TaskStep]) -> list[TaskStep]:
        ids = [step.id for step in v]
        if len(ids) != len(set(ids)):
            raise ValueError("태스크 ID가 중복되어 있습니다.")
        return v

    def get_step_by_id(self, step_id: str) -> TaskStep | None:
        """ID로 태스크 단계 조회"""
        return next((s for s in self.steps if s.id == step_id), None)

    def get_dependency_graph(self) -> dict[str, list[str]]:
        """의존성 그래프 반환 {task_id: [dependent_task_ids]}"""
        return {step.id: step.dependencies for step in self.steps}
