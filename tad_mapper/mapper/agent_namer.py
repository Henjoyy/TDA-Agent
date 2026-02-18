"""
Agent Namer - Gemini LLM으로 자동 발견된 Agent 클러스터에 이름/역할 부여
"""
from __future__ import annotations

import json
import logging
import re

import google.genai as genai

from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from tad_mapper.engine.tda_analyzer import DiscoveredAgent

logger = logging.getLogger(__name__)

_NAMING_PROMPT = """
당신은 AI 시스템 아키텍트입니다.
아래 태스크 그룹을 분석하여 이 그룹을 담당할 Unit Agent의 이름과 역할을 정의해주세요.

태스크 목록:
{task_list}

이 태스크들의 공통점을 파악하여 하나의 Agent로 정의하세요.

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "name": "Agent 이름 (예: 'Trade Statistics Agent')",
  "name_ko": "한국어 이름 (예: '무역 통계 에이전트')",
  "role": "이 Agent의 핵심 역할 (1~2문장)",
  "capabilities": ["capability_1", "capability_2", "capability_3"],
  "mcp_tool_prefix": "snake_case 접두사 (예: 'trade_stats')"
}}
"""


class AgentNamer:
    """Gemini LLM으로 클러스터에 Agent 이름/역할 자동 부여"""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    def name_agents(self, agents: list[DiscoveredAgent]) -> list[DiscoveredAgent]:
        """모든 Agent 클러스터에 이름과 역할 부여"""
        named: list[DiscoveredAgent] = []
        for agent in agents:
            logger.info(f"Agent 명명 중: {agent.agent_id} ({len(agent.task_ids)}개 태스크)")
            named.append(self._name_single(agent))
        return named

    def _name_single(self, agent: DiscoveredAgent) -> DiscoveredAgent:
        """단일 Agent 클러스터 명명"""
        task_list = "\n".join(
            f"- {name}" for name in agent.task_names
        )
        prompt = _NAMING_PROMPT.format(task_list=task_list)

        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            raw = response.text.strip()
            
            # Markdown code block 제거 (```json ... ```)
            if "```" in raw:
                # ```json 또는 ``` 로 시작하는 블록 추출
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
                if match:
                    raw = match.group(1)
            
            # JSON 파싱 시도
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # JSON 파싱 실패 시, 정규식으로 필요한 필드만이라도 추출 시도
                logger.warning(f"JSON 파싱 실패. 정규식 추출 시도: {raw[:100]}...")
                name_match = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
                role_match = re.search(r'"role"\s*:\s*"([^"]+)"', raw)
                
                data = {}
                if name_match:
                    data["name"] = name_match.group(1)
                if role_match:
                    data["role"] = role_match.group(1)
                
                if not data:
                    raise ValueError("유효한 JSON 또는 필드를 찾을 수 없습니다.")

            agent.suggested_name = data.get("name", f"Agent {agent.cluster_id}")
            agent.suggested_role = data.get("role", "")
            agent.suggested_capabilities = data.get("capabilities", [])
            # mcp_tool_prefix를 agent_id에 반영
            prefix = data.get("mcp_tool_prefix", f"agent_{agent.cluster_id}")
            agent.agent_id = prefix

        except Exception as e:
            logger.warning(f"Agent 명명 실패 ({agent.agent_id}): {e}. 기본값 사용.")
            agent.suggested_name = f"Agent {agent.cluster_id}"
            agent.suggested_role = f"태스크 그룹 {agent.cluster_id} 처리"
            agent.suggested_capabilities = ["task_execution"]

        return agent
