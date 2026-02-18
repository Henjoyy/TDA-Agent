"""MCP Tool 스키마 데이터 모델"""
from __future__ import annotations

from pydantic import BaseModel, Field


class MCPPropertySchema(BaseModel):
    """MCP Tool 파라미터 속성 스키마"""
    type: str = "string"
    description: str = ""
    enum: list[str] | None = None
    default: str | int | float | bool | None = None


class MCPInputSchema(BaseModel):
    """MCP Tool 입력 스키마 (JSON Schema 형식)"""
    type: str = "object"
    properties: dict[str, MCPPropertySchema] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class MCPToolAnnotations(BaseModel):
    """MCP Tool 메타 어노테이션"""
    assigned_agent: str
    source_task_id: str
    source_task_name: str
    source_task_ids: list[str] = Field(
        default_factory=list,
        description="이 Tool이 커버하는 원본 태스크 ID 목록(공유 Tool용)",
    )
    source_task_names: list[str] = Field(
        default_factory=list,
        description="이 Tool이 커버하는 원본 태스크 이름 목록(공유 Tool용)",
    )
    confidence: float

    def all_task_ids(self) -> list[str]:
        if self.source_task_ids:
            return list(dict.fromkeys(self.source_task_ids))
        return [self.source_task_id] if self.source_task_id else []

    def all_task_names(self) -> list[str]:
        if self.source_task_names:
            return list(dict.fromkeys(self.source_task_names))
        return [self.source_task_name] if self.source_task_name else []


class MCPToolSchema(BaseModel):
    """MCP 프로토콜 Tool 스키마 (공식 규격 준수)"""
    name: str = Field(..., description="Tool 고유 식별자 (snake_case)")
    description: str = Field(..., description="Tool 기능 설명")
    inputSchema: MCPInputSchema = Field(default_factory=MCPInputSchema)
    annotations: MCPToolAnnotations | None = None

    def to_dict(self) -> dict:
        """JSON 직렬화용 딕셔너리 변환"""
        return self.model_dump(exclude_none=True)
