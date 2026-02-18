"""
Tool Composition Engine — 수학적 정식화 #3

y = (t_π(m) ∘ ... ∘ t_π(2) ∘ t_π(1))(x)

선택된 Agent가 쿼리를 처리하기 위해
MCP Tool을 어떤 순서로 합성(Compose)할지 결정합니다.

Agent는 고정된 코드를 실행하는 게 아니라,
상황에 맞춰 도구(t)들을 조립(∘)하여 문제를 해결합니다.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict, deque

import google.genai as genai

from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from tad_mapper.engine.tda_analyzer import DiscoveredAgent
from tad_mapper.models.mcp_tool import MCPToolSchema
from tad_mapper.models.topology import CompositionPlan, ToolExecutionStep

logger = logging.getLogger(__name__)

_GENERIC_PARAM_KEYS = {"query", "text", "input", "prompt", "question", "request"}

_COMPOSE_PROMPT = """
당신은 AI 에이전트 시스템의 도구 합성 전문가입니다.

사용자의 쿼리를 처리하기 위해 아래 Agent의 MCP Tool 중 필요한 것을 선택하고,
실행 순서를 결정해주세요.

각 Tool의 출력이 다음 Tool의 입력으로 연결되도록 합성 체인을 설계합니다.
수식으로 표현하면: y = (t_π(m) ∘ ... ∘ t_π(1))(x)

═══ Agent 정보 ═══
- 이름: {agent_name}
- 역할: {agent_role}

═══ 사용자 쿼리 ═══
{query}

═══ 사용 가능한 MCP Tools ═══
{tools_list}

═══ 설계 규칙 ═══
1. 쿼리 처리에 필요한 Tool만 선택하세요 (모든 Tool을 쓸 필요 없음)
2. 실행 순서는 데이터 흐름을 고려하여 결정 (이전 Tool의 출력이 다음의 입력)
3. 첫 번째 Tool의 input_from은 "query"
4. 마지막 Tool의 output_to는 "final"
5. 중간 Tool의 output_to는 다음 단계 Tool 이름

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "selected_tools": [
    {{
      "order": 1,
      "tool_name": "tool_name_here",
      "input_from": "query",
      "output_to": "next_tool_name",
      "rationale": "이 단계가 필요한 이유"
    }}
  ],
  "estimated_output": "최종 결과물의 형태 설명 (한국어)"
}}
"""


class ToolComposer:
    """
    MCP Tool 합성 엔진.

    Agent가 쿼리에 맞게 MCP Tool을 동적으로 조립하여
    실행 계획(CompositionPlan)을 수립합니다.
    """

    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def compose(
        self,
        query: str,
        agent: DiscoveredAgent,
        tools: list[MCPToolSchema],
    ) -> CompositionPlan:
        """
        쿼리 + Agent + MCP Tools → 합성 실행 계획 생성.

        Args:
            query: 사용자 쿼리
            agent: 라우팅된 Unit Agent
            tools: 이 Agent에 할당된 MCP Tool 목록

        Returns:
            CompositionPlan: 순서대로 실행할 Tool 체인
        """
        if not tools:
            logger.warning(f"Agent '{agent.agent_id}'에 할당된 Tool이 없습니다.")
            return CompositionPlan(
                query_text=query,
                agent_id=agent.agent_id,
                agent_name=agent.suggested_name or agent.agent_id,
                is_valid=False,
                validation_errors=["해당 Agent에 MCP Tool이 없습니다."],
            )

        agent_name = agent.suggested_name or agent.agent_id
        agent_role = agent.suggested_role or "태스크 처리"

        # 의존성 그래프 사전 계산
        dep_graph = self._build_dependency_graph(tools)

        # LLM으로 합성 순서 결정
        plan = self._llm_compose(query, agent_name, agent_role, tools)

        # 의존성 그래프 추가
        plan.dependency_graph = dep_graph

        # 검증
        errors = self._validate(plan, tools)
        plan.is_valid = len(errors) == 0
        plan.validation_errors = errors

        if not plan.is_valid:
            logger.warning(f"합성 계획 검증 실패: {errors}")
            # 폴백: 위상 정렬 기반 자동 계획
            plan = self._fallback_compose(query, agent, tools, dep_graph)

        logger.info(
            f"합성 계획 생성 완료: {agent_name}, "
            f"{plan.total_steps}단계 Tool 체인"
        )
        return plan

    def validate_plan(self, plan: CompositionPlan) -> bool:
        """합성 계획 유효성 검증 (외부 호출용)"""
        return plan.is_valid

    # ── LLM 기반 합성 ─────────────────────────────────────────────────────────

    def _llm_compose(
        self,
        query: str,
        agent_name: str,
        agent_role: str,
        tools: list[MCPToolSchema],
    ) -> CompositionPlan:
        """Gemini LLM으로 Tool 합성 순서 결정"""
        tools_list = self._format_tools(tools)
        prompt = _COMPOSE_PROMPT.format(
            agent_name=agent_name,
            agent_role=agent_role,
            query=query,
            tools_list=tools_list,
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
            selected = data.get("selected_tools", [])
            estimated_output = data.get("estimated_output", "")

            # Tool 스키마 매핑
            tool_map = {t.name: t for t in tools}
            steps: list[ToolExecutionStep] = []

            for item in selected:
                tool_name = item.get("tool_name", "")
                tool = tool_map.get(tool_name)
                step = ToolExecutionStep(
                    order=int(item.get("order", len(steps) + 1)),
                    tool_name=tool_name,
                    tool_description=tool.description if tool else "",
                    input_from=item.get("input_from", "query"),
                    output_to=item.get("output_to", "final"),
                    rationale=item.get("rationale", ""),
                )
                steps.append(step)

            # 순서 정렬
            steps.sort(key=lambda s: s.order)

            return CompositionPlan(
                query_text=query,
                agent_id="",
                agent_name=agent_name,
                steps=steps,
                estimated_output=estimated_output,
            )

        except Exception as e:
            logger.warning(f"LLM 합성 실패: {e}. 폴백 사용.")
            return CompositionPlan(
                query_text=query,
                agent_id="",
                agent_name=agent_name,
                is_valid=False,
            )

    # ── 의존성 그래프 ─────────────────────────────────────────────────────────

    def _build_dependency_graph(self, tools: list[MCPToolSchema]) -> dict[str, list[str]]:
        """
        Tool 간 의존성 그래프 구성.

        현재 스키마에는 명시적 output 정의가 없으므로,
        각 Tool의 입력 파라미터 키를 기준으로 약한 의존성을 추론합니다.
        반환 형식: {tool_name: [해당 tool 실행 전에 필요한 선행 tool 이름]}
        """
        graph: dict[str, list[str]] = {t.name: [] for t in tools}

        for tool_a in tools:
            # 명시적 output schema가 없으므로 input key를 대리 신호로 사용
            output_keys = (
                set(tool_a.inputSchema.properties.keys()) | set(tool_a.inputSchema.required)
            ) - _GENERIC_PARAM_KEYS
            for tool_b in tools:
                if tool_b.name == tool_a.name:
                    continue
                input_keys = set(tool_b.inputSchema.required) - _GENERIC_PARAM_KEYS
                # 출력 키와 입력 키가 겹치면 의존성 있음
                if output_keys & input_keys:
                    graph[tool_b.name].append(tool_a.name)

        # 중복 제거 + 정렬로 결정적 동작 보장
        for tool_name, deps in graph.items():
            graph[tool_name] = sorted(set(deps))

        return graph

    def _topological_sort(
        self, tools: list[MCPToolSchema], dep_graph: dict[str, list[str]]
    ) -> list[str]:
        """의존성 그래프 위상 정렬 (실행 가능한 순서 도출)"""
        all_nodes = [t.name for t in tools]
        in_degree: dict[str, int] = {
            node: len(dep_graph.get(node, [])) for node in all_nodes
        }
        reverse_adj: dict[str, list[str]] = defaultdict(list)
        for node, deps in dep_graph.items():
            for dep in deps:
                if dep in in_degree:
                    reverse_adj[dep].append(node)

        queue = deque(sorted(name for name, deg in in_degree.items() if deg == 0))
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in sorted(reverse_adj.get(node, [])):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 순환 의존성이 있으면 남은 노드 추가
        remaining = [t.name for t in tools if t.name not in result]
        result.extend(remaining)

        return result

    # ── 폴백 합성 ──────────────────────────────────────────────────────────────

    def _fallback_compose(
        self,
        query: str,
        agent: DiscoveredAgent,
        tools: list[MCPToolSchema],
        dep_graph: dict[str, list[str]],
    ) -> CompositionPlan:
        """위상 정렬 기반 자동 합성 계획 (LLM 실패 시)"""
        sorted_names = self._topological_sort(tools, dep_graph)
        tool_map = {t.name: t for t in tools}

        steps: list[ToolExecutionStep] = []
        for idx, name in enumerate(sorted_names):
            tool = tool_map.get(name)
            is_last = idx == len(sorted_names) - 1
            step = ToolExecutionStep(
                order=idx + 1,
                tool_name=name,
                tool_description=tool.description if tool else "",
                input_from="query" if idx == 0 else sorted_names[idx - 1],
                output_to="final" if is_last else sorted_names[idx + 1],
                rationale="위상 정렬 기반 자동 순서",
            )
            steps.append(step)

        return CompositionPlan(
            query_text=query,
            agent_id=agent.agent_id,
            agent_name=agent.suggested_name or agent.agent_id,
            steps=steps,
            dependency_graph=dep_graph,
            estimated_output="자동 생성된 실행 계획",
            is_valid=True,
        )

    # ── 검증 ──────────────────────────────────────────────────────────────────

    def _validate(
        self, plan: CompositionPlan, tools: list[MCPToolSchema]
    ) -> list[str]:
        """합성 계획 유효성 검증"""
        errors: list[str] = []
        tool_names = {t.name for t in tools}

        # 1. 존재하지 않는 Tool 참조 검증
        for step in plan.steps:
            if step.tool_name not in tool_names:
                errors.append(f"존재하지 않는 Tool 참조: '{step.tool_name}'")

        # 2. 순환 의존성 검증
        name_to_order = {s.tool_name: s.order for s in plan.steps}
        for step in plan.steps:
            if step.input_from not in ("query", "") and step.input_from in name_to_order:
                src_order = name_to_order.get(step.input_from, 0)
                if src_order >= step.order:
                    errors.append(
                        f"순환 의존성: '{step.input_from}'(순서 {src_order})가 "
                        f"'{step.tool_name}'(순서 {step.order}) 이후에 실행됩니다."
                    )

        # 3. 중복 순서 검증
        orders = [s.order for s in plan.steps]
        if len(orders) != len(set(orders)):
            errors.append("중복된 실행 순서(order)가 있습니다.")

        return errors

    # ── 포맷 유틸리티 ─────────────────────────────────────────────────────────

    @staticmethod
    def _format_tools(tools: list[MCPToolSchema]) -> str:
        """LLM 프롬프트용 Tool 목록 포맷"""
        lines: list[str] = []
        for i, tool in enumerate(tools, 1):
            params = list(tool.inputSchema.properties.keys())
            required = tool.inputSchema.required
            lines.append(
                f"{i}. {tool.name}\n"
                f"   설명: {tool.description}\n"
                f"   파라미터: {', '.join(params) or '없음'}\n"
                f"   필수: {', '.join(required) or '없음'}"
            )
        return "\n\n".join(lines)
