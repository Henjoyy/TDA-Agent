"""
MCP Tool 스키마 자동 생성기 (배치 처리 지원)
Gemini LLM으로 태스크용 MCP Tool JSON 스키마를 생성하고,
유사 태스크는 공유 Tool로 병합합니다.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# 기본값을 "실무 우선(타임아웃 방지 + 속도)"으로 조정
DEFAULT_MCP_TIMEOUT_MS = 0
DEFAULT_MCP_BATCH_SIZE = 10
DEFAULT_MCP_RETRIES = 0
DEFAULT_MCP_MAX_WORKERS = 4
DEFAULT_MCP_ENABLE_TOOL_MERGE = True
DEFAULT_MCP_MERGE_MIN_SIMILARITY = 0.55
DEFAULT_MCP_MAX_TASKS_PER_TOOL = 4
DEFAULT_MCP_ENABLE_TASK_DEDUP = True
DEFAULT_MCP_TASK_DEDUP_MIN_TASKS = 12
DEFAULT_MCP_TASK_DEDUP_MIN_SIMILARITY = 0.72
DEFAULT_MCP_TASK_DEDUP_LOOSE_SIMILARITY = 0.45
DEFAULT_MCP_TASK_DEDUP_MIN_TAG_OVERLAP = 0.5
DEFAULT_MCP_TASK_DEDUP_TEMPLATE_SIMILARITY = 0.32
_MERGE_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]{2,}")
_MERGE_STOPWORDS = {
    "execute", "task", "tool", "agent",
    "request", "result", "description",
    "처리", "작업", "태스크", "도구", "에이전트", "설명", "결과",
}
_INTENT_KEYWORDS = {
    "lookup": {"조회", "검색", "확인", "추적", "fetch", "get", "search", "query", "check"},
    "analysis": {"분석", "평가", "진단", "검토", "analyze", "assess", "evaluate", "review"},
    "generate": {"생성", "작성", "요약", "초안", "generate", "create", "draft", "summarize"},
    "notify": {"알림", "통지", "전송", "send", "notify", "alert"},
    "recommend": {"추천", "최적화", "제안", "recommend", "optimize", "suggest"},
    "execute": {"실행", "처리", "자동화", "run", "execute", "process", "automate"},
}

# ── 배치 프롬프트 ─────────────────────────────────────────────────────────────

_BATCH_MCP_PROMPT = """\
당신은 MCP(Model Context Protocol) Tool 설계 전문가입니다.
아래 {n}개의 태스크를 수행하기 위한 MCP Tool JSON 스키마를 **한꺼번에** 설계해주세요.

## 설계 규칙
1. tool name은 snake_case, 동사_명사 형태 (예: get_export_stats, analyze_trade_trend)
2. 파라미터는 실제로 필요한 것만 포함하고, description은 1문장으로 짧게 작성
3. required 배열에 필수 파라미터 명시
4. 출력은 간결하게, 불필요한 설명/중복 문구 금지

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
[Task {idx}] id={task_id} | name={name} | desc={description} | in={input_data} | out={output_data} | agent={agent_name}"""


class MCPGenerator:
    """Gemini LLM 기반 MCP Tool 스키마 자동 생성 (배치 처리)"""

    def __init__(self) -> None:
        raw_timeout_ms = int(os.getenv("TAD_MCP_TIMEOUT_MS", str(DEFAULT_MCP_TIMEOUT_MS)))
        self._timeout_ms = None if raw_timeout_ms <= 0 else max(1_000, raw_timeout_ms)
        self._batch_size = max(
            1, int(os.getenv("TAD_MCP_BATCH_SIZE", str(DEFAULT_MCP_BATCH_SIZE)))
        )
        self._max_retries = max(
            0, int(os.getenv("TAD_MCP_RETRIES", str(DEFAULT_MCP_RETRIES)))
        )
        self._max_workers = max(
            1, int(os.getenv("TAD_MCP_MAX_WORKERS", str(DEFAULT_MCP_MAX_WORKERS)))
        )
        self._merge_enabled = (
            os.getenv(
                "TAD_MCP_ENABLE_TOOL_MERGE",
                "true" if DEFAULT_MCP_ENABLE_TOOL_MERGE else "false",
            ).strip().lower() in {"1", "true", "yes", "y", "on"}
        )
        self._merge_min_similarity = float(
            os.getenv(
                "TAD_MCP_MERGE_MIN_SIMILARITY",
                str(DEFAULT_MCP_MERGE_MIN_SIMILARITY),
            )
        )
        self._max_tasks_per_tool = max(
            1, int(os.getenv("TAD_MCP_MAX_TASKS_PER_TOOL", str(DEFAULT_MCP_MAX_TASKS_PER_TOOL)))
        )
        self._dedup_enabled = (
            os.getenv(
                "TAD_MCP_ENABLE_TASK_DEDUP",
                "true" if DEFAULT_MCP_ENABLE_TASK_DEDUP else "false",
            ).strip().lower() in {"1", "true", "yes", "y", "on"}
        )
        self._dedup_min_tasks = max(
            2,
            int(
                os.getenv(
                    "TAD_MCP_TASK_DEDUP_MIN_TASKS",
                    str(DEFAULT_MCP_TASK_DEDUP_MIN_TASKS),
                )
            ),
        )
        self._dedup_min_similarity = float(
            os.getenv(
                "TAD_MCP_TASK_DEDUP_MIN_SIMILARITY",
                str(DEFAULT_MCP_TASK_DEDUP_MIN_SIMILARITY),
            )
        )
        self._dedup_loose_similarity = float(
            os.getenv(
                "TAD_MCP_TASK_DEDUP_LOOSE_SIMILARITY",
                str(DEFAULT_MCP_TASK_DEDUP_LOOSE_SIMILARITY),
            )
        )
        self._dedup_min_tag_overlap = float(
            os.getenv(
                "TAD_MCP_TASK_DEDUP_MIN_TAG_OVERLAP",
                str(DEFAULT_MCP_TASK_DEDUP_MIN_TAG_OVERLAP),
            )
        )
        self._dedup_template_similarity = float(
            os.getenv(
                "TAD_MCP_TASK_DEDUP_TEMPLATE_SIMILARITY",
                str(DEFAULT_MCP_TASK_DEDUP_TEMPLATE_SIMILARITY),
            )
        )

        client_kwargs = {"api_key": GEMINI_API_KEY}
        if self._timeout_ms is not None:
            client_kwargs["http_options"] = types.HttpOptions(timeout=self._timeout_ms)
        self._client = genai.Client(**client_kwargs)

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def generate(
        self,
        journey: UserJourney,
        agents: list[DiscoveredAgent],
    ) -> list[MCPToolSchema]:
        """
        모든 Agent의 태스크에 대해 MCP Tool 스키마를 배치 생성하고,
        가능하면 공유 Tool로 병합합니다.

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

        llm_tasks, task_alias = self._prepare_llm_tasks(journey.steps, task_to_agent)
        if len(llm_tasks) < len(journey.steps):
            logger.info(
                "  MCP 사전 압축: %s개 태스크 → %s개 대표 태스크 (절감 %s개)",
                len(journey.steps),
                len(llm_tasks),
                len(journey.steps) - len(llm_tasks),
            )

        # LLM 배치 호출 (대용량 입력 시 타임아웃/지연 방지를 위해 chunk 처리)
        llm_results: dict[str, dict] = {}
        chunks = list(self._chunk_tasks(llm_tasks))
        timeout_label = (
            f"{self._timeout_ms}ms"
            if self._timeout_ms is not None
            else "unbounded"
        )
        max_workers = max(1, int(getattr(self, "_max_workers", 1)))
        logger.info(
            "  MCP 배치 분할: 총 %s개 태스크, chunk=%s개, timeout=%s, retries=%s, workers=%s",
            len(llm_tasks),
            len(chunks),
            timeout_label,
            self._max_retries,
            max_workers,
        )
        if len(chunks) == 1 or max_workers == 1:
            for idx, chunk in enumerate(chunks, start=1):
                logger.info("  MCP chunk %s/%s 처리 중... (%s개 태스크)", idx, len(chunks), len(chunk))
                llm_results.update(self._generate_batch(chunk, task_to_agent))
        else:
            logger.info("  MCP chunk 병렬 처리 시작...")
            with ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as executor:
                future_map = {
                    executor.submit(self._generate_batch, chunk, task_to_agent): idx
                    for idx, chunk in enumerate(chunks, start=1)
                }
                for future in as_completed(future_map):
                    idx = future_map[future]
                    try:
                        result = future.result()
                        llm_results.update(result)
                        logger.info("  MCP chunk %s/%s 완료", idx, len(chunks))
                    except Exception as e:
                        logger.warning(
                            "  MCP chunk %s/%s 처리 실패: %s (fallback 예정)",
                            idx,
                            len(chunks),
                            e,
                        )

        # 결과 조합
        tools: list[MCPToolSchema] = []
        success_count = 0
        for task in journey.steps:
            agent = task_to_agent.get(task.id)
            agent_id = agent.agent_id if agent else "unknown"

            lookup_task_id = task_alias.get(task.id, task.id)
            raw_data = llm_results.get(lookup_task_id)
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

        if bool(getattr(self, "_merge_enabled", False)):
            before = len(tools)
            tools = self._merge_shared_tools(tools)
            logger.info(
                "  MCP 공유 Tool 병합: %s개 → %s개 (절감 %s개)",
                before,
                len(tools),
                before - len(tools),
            )

        logger.info(
            f"[배치 MCP Tool 생성 완료] 성공: {success_count}/{len(journey.steps)}개 "
            f"(fallback: {len(journey.steps) - success_count}개, 최종 Tool: {len(tools)}개)"
        )
        return tools

    def _prepare_llm_tasks(
        self,
        tasks: list[TaskStep],
        task_to_agent: dict[str, DiscoveredAgent],
    ) -> tuple[list[TaskStep], dict[str, str]]:
        """
        유사 태스크를 대표 태스크로 압축해 LLM 호출량을 줄입니다.
        반환값:
        - 대표 태스크 목록 (LLM 호출 대상)
        - task_id -> 대표 task_id 매핑
        """
        aliases = {task.id: task.id for task in tasks}
        dedup_enabled = bool(getattr(self, "_dedup_enabled", False))
        dedup_min_tasks = int(getattr(self, "_dedup_min_tasks", 1_000_000))
        dedup_min_similarity = float(getattr(self, "_dedup_min_similarity", 1.01))
        dedup_loose_similarity = float(
            getattr(
                self,
                "_dedup_loose_similarity",
                max(0.35, dedup_min_similarity - 0.22),
            )
        )
        dedup_template_similarity = float(
            getattr(self, "_dedup_template_similarity", 0.32)
        )

        if (not dedup_enabled) or len(tasks) < dedup_min_tasks:
            return tasks, aliases

        representatives: list[TaskStep] = []
        rep_by_agent: dict[str, list[TaskStep]] = defaultdict(list)

        for task in tasks:
            agent = task_to_agent.get(task.id)
            agent_id = agent.agent_id if agent else "unknown"
            candidates = rep_by_agent[agent_id]

            best_rep_id = ""
            best_score = 0.0
            best_rep_task: TaskStep | None = None
            for rep in candidates:
                score = self._task_similarity(task, rep)
                if score > best_score:
                    best_score = score
                    best_rep_id = rep.id
                    best_rep_task = rep

            if (
                best_rep_id
                and (
                    best_score >= dedup_min_similarity
                    or (
                        best_score >= dedup_loose_similarity
                        and best_rep_task is not None
                        and self._task_dedup_compatible(task, best_rep_task)
                    )
                    or (
                        best_score >= dedup_template_similarity
                        and best_rep_task is not None
                        and self._task_template_match(task, best_rep_task)
                    )
                )
            ):
                aliases[task.id] = best_rep_id
                continue

            aliases[task.id] = task.id
            representatives.append(task)
            candidates.append(task)

        if len(representatives) >= len(tasks):
            return tasks, aliases

        return representatives, aliases

    def _task_dedup_compatible(self, a: TaskStep, b: TaskStep) -> bool:
        tags_sim = self._jaccard(set(a.tags), set(b.tags))
        in_sim = self._jaccard(set(a.input_data), set(b.input_data))
        out_sim = self._jaccard(set(a.output_data), set(b.output_data))
        io_sim = 0.5 * in_sim + 0.5 * out_sim

        a_name = self._tokenize_merge(a.name.replace("_", " "))
        b_name = self._tokenize_merge(b.name.replace("_", " "))
        name_overlap = len(a_name & b_name)

        a_intent = self._task_intent(a)
        b_intent = self._task_intent(b)
        same_intent = bool(a_intent and b_intent and a_intent == b_intent)
        min_tag_overlap = float(getattr(self, "_dedup_min_tag_overlap", 0.5))

        if same_intent and (tags_sim >= 0.34 or io_sim >= 0.45):
            if name_overlap >= 2 or out_sim >= 0.5:
                return True

        if tags_sim >= min_tag_overlap and (name_overlap >= 2 or io_sim >= 0.5 or out_sim >= 0.5):
            return True

        return False

    def _task_template_match(self, a: TaskStep, b: TaskStep) -> bool:
        """
        도메인 명사(견적/발주 등)는 달라도 템플릿이 같은 작업인지 판별합니다.
        예: '견적 승인 라우팅' ~ '발주 승인 라우팅'
        """
        a_name = self._tokenize_merge(a.name.replace("_", " "))
        b_name = self._tokenize_merge(b.name.replace("_", " "))
        name_overlap = len(a_name & b_name)

        out_sim = self._jaccard(set(a.output_data), set(b.output_data))
        in_sim = self._jaccard(set(a.input_data), set(b.input_data))
        tags_sim = self._jaccard(set(a.tags), set(b.tags))

        if name_overlap >= 2 and out_sim >= 0.5:
            return True
        if name_overlap >= 2 and (in_sim >= 0.34 or tags_sim >= 0.34):
            return True
        return False

    # ── 내부 메서드 ───────────────────────────────────────────────────────────

    def _generate_batch(
        self, tasks: list[TaskStep], task_to_agent: dict[str, DiscoveredAgent]
    ) -> dict[str, dict]:
        """모든 Tool 설계를 단 1회 LLM 호출로 처리"""
        task_items = []
        for idx, task in enumerate(tasks, start=1):
            agent = task_to_agent.get(task.id)
            agent_name = agent.suggested_name if agent else "Unknown Agent"
            # 프롬프트 크기를 줄여 생성 지연/타임아웃 가능성을 낮춤
            desc = self._shorten(task.description or "미지정", 120)
            input_data = self._shorten(", ".join(task.input_data) or "미지정", 80)
            output_data = self._shorten(", ".join(task.output_data) or "미지정", 80)

            task_items.append(_TASK_ITEM_TEMPLATE.format(
                idx=idx,
                task_id=task.id,
                name=self._shorten(task.name, 60),
                description=desc,
                input_data=input_data,
                output_data=output_data,
                agent_name=agent_name,
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
                source_task_ids=[task.id],
                source_task_names=[task.name],
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
                source_task_ids=[task.id],
                source_task_names=[task.name],
                confidence=0.5,
            ),
        )

    def _merge_shared_tools(self, tools: list[MCPToolSchema]) -> list[MCPToolSchema]:
        """
        유사한 태스크 Tool을 Agent별로 병합해 공유 Tool로 축약합니다.

        병합 기준:
        - 같은 assigned_agent
        - 이름/설명/입력 스키마 유사도 >= 임계값
        - 병합된 Tool당 태스크 수 <= max_tasks_per_tool
        """
        by_agent: dict[str, list[MCPToolSchema]] = defaultdict(list)
        passthrough: list[MCPToolSchema] = []
        for tool in tools:
            ann = tool.annotations
            if ann is None or not ann.assigned_agent:
                passthrough.append(tool)
                continue
            by_agent[ann.assigned_agent].append(tool)

        merged_tools: list[MCPToolSchema] = []
        for agent_id, agent_tools in by_agent.items():
            clusters: list[list[MCPToolSchema]] = []
            for tool in agent_tools:
                placed = False
                for cluster in clusters:
                    if self._cluster_task_count(cluster) >= self._max_tasks_per_tool:
                        continue
                    if self._can_merge(tool, cluster):
                        cluster.append(tool)
                        placed = True
                        break
                if not placed:
                    clusters.append([tool])

            for cluster in clusters:
                merged_tools.append(self._build_merged_tool(cluster, agent_id))

        all_tools = [*merged_tools, *passthrough]
        return self._ensure_unique_names(all_tools)

    def _can_merge(self, tool: MCPToolSchema, cluster: list[MCPToolSchema]) -> bool:
        # cluster 대표와의 유사도 + cluster 내부 최대 유사도로 안정화
        sims = [self._tool_similarity(tool, other) for other in cluster]
        if not sims:
            return False
        max_score, max_name, max_desc, max_schema = max(sims, key=lambda x: x[0])
        avg_score = sum(s[0] for s in sims) / len(sims)
        max_task = max(
            self._jaccard(self._task_name_tokens(tool), self._task_name_tokens(other))
            for other in cluster
        )

        tool_intent = self._intent_signature(tool)
        same_intent = False
        if tool_intent:
            for other in cluster:
                if self._intent_signature(other) == tool_intent:
                    same_intent = True
                    break

        # 동일 intent + schema가 거의 동일하면 task 토큰 겹침이 낮아도 병합 허용
        if same_intent and max_schema >= 0.8:
            return True
        # 동일 intent + schema 호환이면 task명 기준으로 병합 허용
        if same_intent and max_schema >= 0.5 and max_task >= 0.2:
            return True

        # 과병합 방지: 단순 schema 동일만으로는 merge 금지
        lexical_gate = (max_name >= 0.45) or (max_name >= 0.25 and max_desc >= 0.55)
        if not lexical_gate:
            return False
        return (
            max_score >= self._merge_min_similarity
            or (avg_score >= (self._merge_min_similarity + 0.08) and max_schema >= 0.5)
            or (max_task >= 0.4 and max_schema >= 0.5)
        )

    def _tool_similarity(self, a: MCPToolSchema, b: MCPToolSchema) -> tuple[float, float, float, float]:
        a_name = self._tokenize_merge(a.name.replace("_", " "))
        b_name = self._tokenize_merge(b.name.replace("_", " "))
        a_desc = self._tokenize_merge(a.description)
        b_desc = self._tokenize_merge(b.description)
        a_task = self._task_name_tokens(a)
        b_task = self._task_name_tokens(b)

        a_schema = set(a.inputSchema.properties.keys()) | set(a.inputSchema.required)
        b_schema = set(b.inputSchema.properties.keys()) | set(b.inputSchema.required)

        name_sim = self._jaccard(a_name, b_name)
        desc_sim = self._jaccard(a_desc, b_desc)
        task_sim = self._jaccard(a_task, b_task)
        schema_sim = self._jaccard(a_schema, b_schema)
        score = float(
            0.30 * name_sim
            + 0.25 * desc_sim
            + 0.25 * task_sim
            + 0.20 * schema_sim
        )
        return score, name_sim, desc_sim, schema_sim

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        if inter <= 0:
            return 0.0
        union = len(a | b)
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _tokenize_merge(text: str) -> set[str]:
        return {
            t.lower()
            for t in _MERGE_TOKEN_PATTERN.findall(text or "")
            if t.lower() not in _MERGE_STOPWORDS
        }

    def _task_similarity(self, a: TaskStep, b: TaskStep) -> float:
        a_name = self._tokenize_merge(a.name.replace("_", " "))
        b_name = self._tokenize_merge(b.name.replace("_", " "))
        a_desc = self._tokenize_merge(a.description)
        b_desc = self._tokenize_merge(b.description)
        a_io = self._tokenize_merge(" ".join([*a.input_data, *a.output_data]))
        b_io = self._tokenize_merge(" ".join([*b.input_data, *b.output_data]))

        name_sim = self._jaccard(a_name, b_name)
        desc_sim = self._jaccard(a_desc, b_desc)
        io_sim = self._jaccard(a_io, b_io)
        sim = 0.45 * name_sim + 0.35 * desc_sim + 0.20 * io_sim

        a_intent = self._task_intent(a)
        b_intent = self._task_intent(b)
        if a_intent and b_intent:
            if a_intent == b_intent:
                sim += 0.10
            else:
                sim *= 0.80
        return float(min(1.0, max(0.0, sim)))

    def _task_intent(self, task: TaskStep) -> str:
        tokens = self._tokenize_merge(f"{task.name} {task.description}")
        if not tokens:
            return ""
        best_intent = ""
        best_score = 0
        for intent, keywords in _INTENT_KEYWORDS.items():
            score = len(tokens & {k.lower() for k in keywords})
            if score > best_score:
                best_intent = intent
                best_score = score
        return best_intent

    def _task_name_tokens(self, tool: MCPToolSchema) -> set[str]:
        ann = tool.annotations
        if ann is None:
            return set()
        return self._tokenize_merge(" ".join(ann.all_task_names()))

    def _intent_signature(self, tool: MCPToolSchema) -> str:
        ann = tool.annotations
        texts: list[str] = [tool.name, tool.description]
        if ann is not None:
            texts.extend(ann.all_task_names())
        tokens = self._tokenize_merge(" ".join(texts))
        if not tokens:
            return ""

        best_intent = ""
        best_score = 0
        for intent, keywords in _INTENT_KEYWORDS.items():
            keyword_tokens = {k.lower() for k in keywords}
            score = len(tokens & keyword_tokens)
            if score > best_score:
                best_intent = intent
                best_score = score
        return best_intent

    @staticmethod
    def _cluster_task_count(cluster: list[MCPToolSchema]) -> int:
        task_ids: set[str] = set()
        for tool in cluster:
            ann = tool.annotations
            if ann is None:
                continue
            task_ids.update(ann.all_task_ids())
        return len(task_ids)

    def _build_merged_tool(
        self, cluster: list[MCPToolSchema], agent_id: str
    ) -> MCPToolSchema:
        if len(cluster) == 1:
            return cluster[0]

        # 대표 tool은 confidence가 가장 높은 항목 우선
        representative = max(
            cluster,
            key=lambda t: t.annotations.confidence if t.annotations else 0.0,
        )

        merged_properties: dict[str, MCPPropertySchema] = {}
        required_sets: list[set[str]] = []
        task_ids: list[str] = []
        task_names: list[str] = []
        confidences: list[float] = []

        for tool in cluster:
            for key, prop in tool.inputSchema.properties.items():
                if key not in merged_properties:
                    merged_properties[key] = prop
            required_sets.append(set(tool.inputSchema.required))
            ann = tool.annotations
            if ann is not None:
                task_ids.extend(ann.all_task_ids())
                task_names.extend(ann.all_task_names())
                confidences.append(float(ann.confidence))

        task_ids = list(dict.fromkeys(task_ids))
        task_names = list(dict.fromkeys(task_names))

        required = (
            sorted(set.intersection(*required_sets))
            if required_sets else []
        )
        if not required and "query" in merged_properties:
            required = ["query"]
        if "query" not in merged_properties:
            merged_properties["query"] = MCPPropertySchema(
                type="string", description="처리할 요청 내용"
            )
            if "query" not in required:
                required.append("query")

        merged_description = (
            f"{representative.description} "
            f"(통합 처리 태스크: {', '.join(task_names[:4])}"
        )
        if len(task_names) > 4:
            merged_description += f" 외 {len(task_names) - 4}개"
        merged_description += ")"

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.7
        return MCPToolSchema(
            name=representative.name,
            description=merged_description,
            inputSchema=MCPInputSchema(
                properties=merged_properties,
                required=required,
            ),
            annotations=MCPToolAnnotations(
                assigned_agent=agent_id,
                source_task_id=task_ids[0] if task_ids else "",
                source_task_name=task_names[0] if task_names else "",
                source_task_ids=task_ids,
                source_task_names=task_names,
                confidence=float(avg_conf),
            ),
        )

    @staticmethod
    def _ensure_unique_names(tools: list[MCPToolSchema]) -> list[MCPToolSchema]:
        seen: dict[str, int] = {}
        uniqued: list[MCPToolSchema] = []
        for tool in tools:
            base = tool.name
            idx = seen.get(base, 0)
            if idx == 0:
                seen[base] = 1
                uniqued.append(tool)
                continue
            # 충돌 시 접미사 부여
            new_tool = tool.model_copy(deep=True)
            new_name = f"{base}_{idx+1}"
            while new_name in seen:
                idx += 1
                new_name = f"{base}_{idx+1}"
            new_tool.name = new_name
            seen[base] = idx + 1
            seen[new_name] = 1
            uniqued.append(new_tool)
        return uniqued

    @staticmethod
    def _task_to_tool_name(task_name: str) -> str:
        """태스크 이름을 snake_case Tool 이름으로 변환"""
        name = task_name.lower()
        name = re.sub(r"[^a-z0-9가-힣]", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return f"execute_{name}" if name else "execute_task"

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        """프롬프트 과대화를 막기 위해 텍스트를 제한 길이로 축약"""
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: max(0, limit - 3)] + "..."
