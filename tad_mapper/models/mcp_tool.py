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
    confidence: float


class MCPToolSchema(BaseModel):
    """MCP 프로토콜 Tool 스키마 (공식 규격 준수)"""
    name: str = Field(..., description="Tool 고유 식별자 (snake_case)")
    description: str = Field(..., description="Tool 기능 설명")
    inputSchema: MCPInputSchema = Field(default_factory=MCPInputSchema)
    annotations: MCPToolAnnotations | None = None

    def to_dict(self) -> dict:
        """JSON 직렬화용 딕셔너리 변환"""
        return self.model_dump(exclude_none=True)
