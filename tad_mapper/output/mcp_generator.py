"""
MCP Tool 스키마 자동 생성기 (배치 처리 지원)
Gemini LLM을 사용하여 각 태스크에 필요한 MCP Tool JSON 스키마를 생성합니다.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import google.genai as genai
from google.genai import types

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

DEFAULT_MCP_TIMEOUT_MS = 45_000
DEFAULT_MCP_BATCH_SIZE = 12
DEFAULT_MCP_RETRIES = 1

# ── 배치 프롬프트 ─────────────────────────────────────────────────────────────

_BATCH_MCP_PROMPT = """\
당신은 MCP(Model Context Protocol) Tool 설계 전문가입니다.
아래 {n}개의 태스크를 수행하기 위한 MCP Tool JSON 스키마를 **한꺼번에** 설계해주세요.

## 설계 규칙
1. tool name은 snake_case, 동사_명사 형태 (예: get_export_stats, analyze_trade_trend)
2. 파라미터는 실제로 필요한 것만 포함하고 명확한 description 작성 (한국어 가능)
3. required 배열에 필수 파라미터 명시

## 태스크 목록
{task_list}

## 출력 형식 (반드시 아래 JSON 배열 형식으로만 응답하세요. 마크다운 코드블록 없이)
[
  {{
    "task_id": "task_001",
    "name": "tool_name",
    "description": "Tool 설명",
    "inputSchema": {{
      "type": "object",
      "properties": {{
        "param_name": {{ "type": "string", "description": "설명" }}
      }},
      "required": ["param_name"]
    }}
  }},
  ...
]
"""

_TASK_ITEM_TEMPLATE = """\
[Task {idx}] task_id={task_id}
  이름: {name}
  설명: {description}
  입력 데이터: {input_data}
  출력 데이터: {output_data}
  소속 Agent: {agent_name} ({agent_role})"""


class MCPGenerator:
    """Gemini LLM 기반 MCP Tool 스키마 자동 생성 (배치 처리)"""

    def __init__(self) -> None:
        self._timeout_ms = max(
            1_000, int(os.getenv("TAD_MCP_TIMEOUT_MS", str(DEFAULT_MCP_TIMEOUT_MS)))
        )
        self._batch_size = max(
            1, int(os.getenv("TAD_MCP_BATCH_SIZE", str(DEFAULT_MCP_BATCH_SIZE)))
        )
        self._max_retries = max(
            0, int(os.getenv("TAD_MCP_RETRIES", str(DEFAULT_MCP_RETRIES)))
        )
        self._client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=self._timeout_ms),
        )

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def generate(
        self,
        journey: UserJourney,
        agents: list[DiscoveredAgent],
    ) -> list[MCPToolSchema]:
        """
        모든 Agent의 태스크에 대해 MCP Tool 스키마를 단 1회 LLM 호출로 생성.

        파싱 실패 시 해당 태스크만 기본 스키마(fallback)로 복구합니다.
        """
        if not journey.steps:
            return []

        logger.info(f"[배치 MCP Tool 생성] {len(journey.steps)}개 Tool 처리 시작...")

        # 태스크-에이전트 매핑 준비
        task_map = {step.id: step for step in journey.steps}
        task_to_agent = {}
        for agent in agents:
            for tid in agent.task_ids:
                task_to_agent[tid] = agent

        # LLM 배치 호출 (대용량 입력 시 타임아웃/지연 방지를 위해 chunk 처리)
        llm_results: dict[str, dict] = {}
        chunks = list(self._chunk_tasks(journey.steps))
        logger.info(
            "  MCP 배치 분할: 총 %s개 태스크, chunk=%s개, timeout=%sms, retries=%s",
            len(journey.steps),
            len(chunks),
            self._timeout_ms,
            self._max_retries,
        )
        for idx, chunk in enumerate(chunks, start=1):
            logger.info("  MCP chunk %s/%s 처리 중... (%s개 태스크)", idx, len(chunks), len(chunk))
            llm_results.update(self._generate_batch(chunk, task_to_agent))

        # 결과 조합
        tools: list[MCPToolSchema] = []
        success_count = 0
        for task in journey.steps:
            agent = task_to_agent.get(task.id)
            agent_id = agent.agent_id if agent else "unknown"
            
            raw_data = llm_results.get(task.id)
            if raw_data:
                try:
                    tool = self._parse_single_tool(raw_data, task, agent_id)
                    tools.append(tool)
                    success_count += 1
                    continue
                except Exception as e:
                    logger.warning(f"  [{task.id}] Tool 파싱 실패: {e} → fallback 사용")

            # fallback
            tools.append(self._fallback_schema(task, agent_id))

        logger.info(
            f"[배치 MCP Tool 생성 완료] 성공: {success_count}/{len(journey.steps)}개 "
            f"(fallback: {len(journey.steps) - success_count}개)"
        )
        return tools

    # ── 내부 메서드 ───────────────────────────────────────────────────────────

    def _generate_batch(
        self, tasks: list[TaskStep], task_to_agent: dict[str, DiscoveredAgent]
    ) -> dict[str, dict]:
        """모든 Tool 설계를 단 1회 LLM 호출로 처리"""
        task_items = []
        for idx, task in enumerate(tasks, start=1):
            agent = task_to_agent.get(task.id)
            agent_name = agent.suggested_name if agent else "Unknown Agent"
            agent_role = agent.suggested_role if agent else "General Task Processing"
            
            task_items.append(_TASK_ITEM_TEMPLATE.format(
                idx=idx,
                task_id=task.id,
                name=task.name,
                description=task.description or "미지정",
                input_data=", ".join(task.input_data) or "미지정",
                output_data=", ".join(task.output_data) or "미지정",
                agent_name=agent_name,
                agent_role=agent_role,
            ))

        prompt = _BATCH_MCP_PROMPT.format(
            n=len(tasks),
            task_list="\n\n".join(task_items),
        )

        for attempt in range(self._max_retries + 1):
            try:
                logger.info(
                    "  LLM 배치 호출 중 (모델: %s, %s개 Tool, attempt=%s/%s)...",
                    GEMINI_MODEL,
                    len(tasks),
                    attempt + 1,
                    self._max_retries + 1,
                )
                response = self._client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        responseMimeType="application/json",
                    ),
                )
                raw = (response.text or "").strip()
                logger.info("  LLM 응답 수신 완료. JSON 파싱 중...")
                return self._parse_batch_response(raw, tasks)
            except Exception as e:
                if attempt < self._max_retries:
                    wait = min(2 ** attempt, 4)
                    logger.warning(
                        "  LLM 배치 호출 실패(재시도 예정): %s (sleep=%ss)",
                        e,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                logger.warning(f"  LLM 배치 호출 실패: {e}. 전체 Tool fallback 처리.")
                return {}

        return {}

    def _parse_batch_response(
        self, raw: str, tasks: list[TaskStep]
    ) -> dict[str, dict]:
        """LLM 응답(JSON 배열)을 파싱하여 {task_id: data} 딕셔너리로 변환"""
        if "```" in raw:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
            if match:
                raw = match.group(1)

        json_array = self._extract_json_array(raw)
        if not json_array:
            logger.warning("  배치 응답에서 JSON 배열을 찾을 수 없습니다.")
            return {}

        try:
            items: list[dict] = json.loads(json_array)
        except json.JSONDecodeError as e:
            logger.warning(f"  JSON 파싱 실패: {e}")
            return {}

        result: dict[str, dict] = {}
        task_ids = [t.id for t in tasks]

        for item in items:
            tid = item.get("task_id", "").strip()
            if tid in task_ids:
                result[tid] = item

        if len(result) < len(tasks) // 2:
            logger.warning("  task_id 매핑 성공률 낮음. 순서 기반 매핑 사용.")
            result = {task.id: item for task, item in zip(tasks, items)}

        return result

    @staticmethod
    def _extract_json_array(raw: str) -> str | None:
        """
        문자열에서 첫 JSON 배열 블록을 안전하게 추출합니다.
        """
        start = raw.find("[")
        if start < 0:
            return None

        depth = 0
        in_str = False
        escaped = False
        for idx in range(start, len(raw)):
            ch = raw[idx]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue
            if ch == "[":
                depth += 1
                continue
            if ch == "]":
                depth -= 1
                if depth == 0:
                    return raw[start:idx + 1]
        return None

    def _chunk_tasks(self, tasks: list[TaskStep]) -> list[list[TaskStep]]:
        """고정 크기 chunk로 태스크 분할"""
        return [
            tasks[i:i + self._batch_size] for i in range(0, len(tasks), self._batch_size)
        ]

    def _parse_single_tool(self, data: dict, task: TaskStep, agent_id: str) -> MCPToolSchema:
        """JSON 데이터로부터 MCPToolSchema 객체 생성"""
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
            description=data.get("description", task.description or task.name),
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
        name = task_name.lower()
        name = re.sub(r"[^a-z0-9가-힣]", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return f"execute_{name}" if name else "execute_task"
