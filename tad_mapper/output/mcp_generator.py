"""
MCP Tool 스키마 자동 생성기
Gemini LLM을 사용하여 각 태스크에 필요한 MCP Tool JSON 스키마를 생성합니다.
"""
from __future__ import annotations

import json
import logging
import re

import google.genai as genai

from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from tad_mapper.engine.tda_analyzer import DiscoveredAgent
from tad_mapper.models.journey import TaskStep, UserJourney
from tad_mapper.models.mcp_tool import (
    MCPInputSchema,
    MCPPropertySchema,
    MCPToolAnnotations,
    MCPToolSchema,
)

logger = logging.getLogger(__name__)

_MCP_PROMPT = """
당신은 MCP(Model Context Protocol) Tool 설계 전문가입니다.
아래 태스크를 수행하기 위한 MCP Tool JSON 스키마를 설계해주세요.

Agent 정보:
- Agent 이름: {agent_name}
- Agent 역할: {agent_role}

태스크 정보:
- 이름: {task_name}
- 설명: {task_description}
- 입력 데이터: {input_data}
- 출력 데이터: {output_data}

MCP Tool 설계 규칙:
1. tool name은 snake_case, 동사_명사 형태 (예: get_export_stats, analyze_trade_trend)
2. 파라미터는 실제로 필요한 것만 포함
3. 각 파라미터에 명확한 description 작성
4. required 배열에 필수 파라미터 명시

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "name": "tool_name_in_snake_case",
  "description": "Tool이 하는 일을 명확하게 설명 (한국어 가능)",
  "inputSchema": {{
    "type": "object",
    "properties": {{
      "param_name": {{
        "type": "string",
        "description": "파라미터 설명"
      }}
    }},
    "required": ["param_name"]
  }}
}}
"""


class MCPGenerator:
    """Gemini LLM 기반 MCP Tool 스키마 자동 생성"""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    def generate(
        self,
        journey: UserJourney,
        agents: list[DiscoveredAgent],
    ) -> list[MCPToolSchema]:
        """모든 Agent의 태스크에 대해 MCP Tool 스키마 생성"""
        tools: list[MCPToolSchema] = []
        task_map = {step.id: step for step in journey.steps}

        for agent in agents:
            agent_name = agent.suggested_name or agent.agent_id
            agent_role = agent.suggested_role or "태스크 처리"

            for task_id in agent.task_ids:
                task = task_map.get(task_id)
                if not task:
                    continue
                logger.info(f"MCP Tool 생성 중: {task.name} → {agent_name}")
                tool = self._generate_single(task, agent_name, agent_role, agent.agent_id)
                tools.append(tool)

        return tools

    def _generate_single(
        self,
        task: TaskStep,
        agent_name: str,
        agent_role: str,
        agent_id: str,
    ) -> MCPToolSchema:
        """단일 태스크에 대한 MCP Tool 스키마 생성"""
        prompt = _MCP_PROMPT.format(
            agent_name=agent_name,
            agent_role=agent_role,
            task_name=task.name,
            task_description=task.description,
            input_data=", ".join(task.input_data) or "미지정",
            output_data=", ".join(task.output_data) or "미지정",
        )

        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            raw = response.text.strip()
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                raise ValueError("JSON 응답을 찾을 수 없습니다.")
            data = json.loads(json_match.group())

            # inputSchema 파싱
            schema_data = data.get("inputSchema", {})
            properties: dict[str, MCPPropertySchema] = {}
            for prop_name, prop_data in schema_data.get("properties", {}).items():
                properties[prop_name] = MCPPropertySchema(
                    type=prop_data.get("type", "string"),
                    description=prop_data.get("description", ""),
                    enum=prop_data.get("enum"),
                    default=prop_data.get("default"),
                )

            return MCPToolSchema(
                name=data.get("name", self._task_to_tool_name(task.name)),
                description=data.get("description", task.description),
                inputSchema=MCPInputSchema(
                    properties=properties,
                    required=schema_data.get("required", []),
                ),
                annotations=MCPToolAnnotations(
                    assigned_agent=agent_id,
                    source_task_id=task.id,
                    source_task_name=task.name,
                    confidence=0.9,
                ),
            )

        except Exception as e:
            logger.warning(f"MCP Tool 생성 실패 ({task.name}): {e}. 기본 스키마 사용.")
            return self._fallback_schema(task, agent_id)

    def _fallback_schema(self, task: TaskStep, agent_id: str) -> MCPToolSchema:
        """LLM 실패 시 기본 스키마 생성"""
        return MCPToolSchema(
            name=self._task_to_tool_name(task.name),
            description=task.description or task.name,
            inputSchema=MCPInputSchema(
                properties={"query": MCPPropertySchema(
                    type="string",
                    description="처리할 요청 내용",
                )},
                required=["query"],
            ),
            annotations=MCPToolAnnotations(
                assigned_agent=agent_id,
                source_task_id=task.id,
                source_task_name=task.name,
                confidence=0.5,
            ),
        )

    @staticmethod
    def _task_to_tool_name(task_name: str) -> str:
        """태스크 이름을 snake_case Tool 이름으로 변환"""
        import unicodedata
        # 한글 등 유니코드 → ASCII 근사 변환 시도
        name = task_name.lower()
        # 공백/특수문자 → 언더스코어
        name = re.sub(r"[^a-z0-9가-힣]", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return f"execute_{name}" if name else "execute_task"
