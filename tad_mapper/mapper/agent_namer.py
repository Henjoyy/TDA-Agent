"""
Agent Namer - Gemini LLM으로 자동 발견된 Agent 클러스터에 이름/역할 부여

배치 처리 방식:
- N개 에이전트를 단 1회 LLM 호출로 명명 (순차 N회 → 배치 1회)
- Chain-of-Thought(CoT) 프롬프트로 구체적인 이름 생성
- 키워드 빈도 분석 기반 스마트 폴백 (더 이상 "범용 처리 에이전트" 없음)
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter

import google.genai as genai

from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from tad_mapper.engine.tda_analyzer import DiscoveredAgent

logger = logging.getLogger(__name__)

# ── 도메인 키워드 → 에이전트 역할 매핑 (폴백용) ─────────────────────────────
_DOMAIN_KEYWORD_MAP: dict[str, tuple[str, str]] = {
    # 키워드: (한국어 이름, 영어 prefix)
    "분석": ("분석 에이전트", "analysis"),
    "통계": ("통계 분석 에이전트", "statistics"),
    "예측": ("예측 모델 에이전트", "prediction"),
    "추천": ("추천 시스템 에이전트", "recommendation"),
    "검색": ("검색 에이전트", "search"),
    "분류": ("분류 에이전트", "classification"),
    "모니터링": ("모니터링 에이전트", "monitoring"),
    "자동화": ("자동화 에이전트", "automation"),
    "최적화": ("최적화 에이전트", "optimization"),
    "관리": ("관리 에이전트", "management"),
    "처리": ("데이터 처리 에이전트", "processing"),
    "생성": ("콘텐츠 생성 에이전트", "generation"),
    "번역": ("번역 에이전트", "translation"),
    "요약": ("요약 에이전트", "summarization"),
    "감지": ("이상 감지 에이전트", "detection"),
    "물류": ("물류 에이전트", "logistics"),
    "무역": ("무역 에이전트", "trade"),
    "금융": ("금융 에이전트", "finance"),
    "고객": ("고객 서비스 에이전트", "customer_service"),
    "재고": ("재고 관리 에이전트", "inventory"),
    "공급": ("공급망 에이전트", "supply_chain"),
    "마케팅": ("마케팅 에이전트", "marketing"),
    "리스크": ("리스크 관리 에이전트", "risk"),
    "보안": ("보안 에이전트", "security"),
    "품질": ("품질 관리 에이전트", "quality"),
    "수요": ("수요 예측 에이전트", "demand_forecast"),
    "가격": ("가격 최적화 에이전트", "pricing"),
    "계약": ("계약 관리 에이전트", "contract"),
    "규정": ("규정 준수 에이전트", "compliance"),
    "사기": ("사기 탐지 에이전트", "fraud_detection"),
}

# ── 배치 프롬프트 ─────────────────────────────────────────────────────────────

_BATCH_NAMING_PROMPT = """\
당신은 AI 시스템 아키텍트입니다.
아래 {n}개의 Agent 클러스터를 **한꺼번에** 분석하여 각각에 이름과 역할을 부여하세요.

## 출력 규칙
- 이름은 반드시 구체적이어야 함 (예: "무역 리스크 분석 에이전트" ✅, "범용 처리 에이전트" ❌)
- 각 클러스터의 태스크 목록을 보고 공통 도메인 + 핵심 기능을 결합하여 이름 결정
- mcp_tool_prefix는 snake_case로 작성

## Agent 클러스터 목록
{agent_list}

## 출력 형식 (반드시 아래 JSON 배열 형식으로만 응답하세요. 마크다운 코드블록 없이)
[
  {{
    "agent_id": "agent_0",
    "name": "영문 Agent 이름 (예: 'Trade Risk Analysis Agent')",
    "name_ko": "한국어 이름 (예: '무역 리스크 분석 에이전트')",
    "role": "이 Agent의 핵심 역할 (1~2문장, 구체적으로)",
    "capabilities": ["capability_1", "capability_2", "capability_3"],
    "mcp_tool_prefix": "snake_case 접두사 (예: 'trade_risk')"
  }},
  ...
]
"""

_AGENT_ITEM_TEMPLATE = """\
[Agent {idx}] agent_id={agent_id}
  태스크 목록:
{task_list}"""


class AgentNamer:
    """Gemini LLM으로 클러스터에 Agent 이름/역할 자동 부여 (배치 처리)"""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def name_agents(self, agents: list[DiscoveredAgent]) -> list[DiscoveredAgent]:
        """
        모든 Agent 클러스터에 이름과 역할을 단 1회 LLM 호출로 부여.

        LLM 배치 호출 실패 시 각 에이전트를 키워드 폴백으로 처리합니다.
        """
        if not agents:
            return []

        logger.info(f"[배치 Agent 명명] {len(agents)}개 에이전트를 1회 LLM 호출로 처리 시작...")

        # LLM 배치 호출
        llm_results: dict[str, dict] = self._name_batch(agents)

        # 결과 적용
        named: list[DiscoveredAgent] = []
        success_count = 0
        for agent in agents:
            data = llm_results.get(agent.agent_id)
            if data:
                try:
                    self._apply_naming(agent, data)
                    named.append(agent)
                    success_count += 1
                    continue
                except Exception as e:
                    logger.warning(f"  [{agent.agent_id}] 결과 적용 실패: {e} → 키워드 폴백")

            # 폴백
            self._apply_keyword_fallback(agent)
            named.append(agent)

        logger.info(
            f"[배치 Agent 명명 완료] 성공: {success_count}/{len(agents)}개 "
            f"(fallback: {len(agents) - success_count}개)"
        )
        return named

    # ── 내부 메서드 ───────────────────────────────────────────────────────────

    def _name_batch(self, agents: list[DiscoveredAgent]) -> dict[str, dict]:
        """
        모든 에이전트를 단 1회 LLM 호출로 명명.

        Returns:
            {agent_id: {name, name_ko, role, capabilities, mcp_tool_prefix}} 딕셔너리.
        """
        agent_items = []
        for idx, agent in enumerate(agents, start=1):
            task_list_str = "\n".join(f"    - {name}" for name in agent.task_names)
            agent_items.append(_AGENT_ITEM_TEMPLATE.format(
                idx=idx,
                agent_id=agent.agent_id,
                task_list=task_list_str,
            ))

        prompt = _BATCH_NAMING_PROMPT.format(
            n=len(agents),
            agent_list="\n\n".join(agent_items),
        )

        try:
            logger.info(f"  LLM 배치 호출 중 (모델: {GEMINI_MODEL}, {len(agents)}개 에이전트)...")
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            raw = response.text.strip()
            logger.info("  LLM 응답 수신 완료. JSON 파싱 중...")
            return self._parse_batch_response(raw, agents)

        except Exception as e:
            logger.warning(f"  LLM 배치 호출 실패: {e}. 전체 에이전트 키워드 폴백 처리.")
            return {}

    def _parse_batch_response(
        self, raw: str, agents: list[DiscoveredAgent]
    ) -> dict[str, dict]:
        """
        LLM 응답(JSON 배열)을 파싱하여 {agent_id: data} 딕셔너리로 변환.
        """
        # 마크다운 코드블록 제거
        if "```" in raw:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
            if match:
                raw = match.group(1)

        # JSON 배열 추출
        array_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not array_match:
            logger.warning("  배치 응답에서 JSON 배열을 찾을 수 없습니다.")
            return {}

        try:
            items: list[dict] = json.loads(array_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"  JSON 파싱 실패: {e}")
            return {}

        # agent_id 기준으로 매핑
        result: dict[str, dict] = {}
        agent_ids = [a.agent_id for a in agents]

        for item in items:
            aid = item.get("agent_id", "").strip()
            if aid in agent_ids:
                result[aid] = item

        # agent_id 매핑 실패 시 순서 기반 fallback 매핑
        if len(result) < len(agents) // 2:
            logger.warning(
                f"  agent_id 매핑 성공률 낮음 ({len(result)}/{len(agents)}). "
                "순서 기반 매핑으로 전환..."
            )
            result = {}
            for agent, item in zip(agents, items):
                result[agent.agent_id] = item

        logger.info(f"  파싱 성공: {len(result)}/{len(agents)}개 에이전트")
        return result

    def _apply_naming(self, agent: DiscoveredAgent, data: dict) -> None:
        """LLM 결과를 에이전트에 적용"""
        name_ko = data.get("name_ko", "").strip()
        name_en = data.get("name", "").strip()
        candidate_name = name_ko or name_en

        # "범용" 또는 너무 짧은 이름은 키워드 폴백으로 교체
        if not candidate_name or "범용" in candidate_name or len(candidate_name) < 5:
            candidate_name = self._keyword_fallback_name(agent)

        agent.suggested_name = candidate_name
        agent.suggested_role = data.get("role", self._keyword_fallback_role(agent))
        agent.suggested_capabilities = data.get("capabilities", [])

        # mcp_tool_prefix는 내부 식별자(agent_id)와 분리해 저장
        prefix = data.get("mcp_tool_prefix", "").strip()
        if not prefix or prefix == "agent":
            prefix = self._keyword_fallback_prefix(agent)
        agent.tool_prefix = prefix

    def _apply_keyword_fallback(self, agent: DiscoveredAgent) -> None:
        """키워드 폴백으로 에이전트 명명"""
        agent.suggested_name = self._keyword_fallback_name(agent)
        agent.suggested_role = self._keyword_fallback_role(agent)
        agent.suggested_capabilities = ["task_execution", "data_processing"]
        agent.tool_prefix = self._keyword_fallback_prefix(agent)

    # ── 스마트 폴백 메서드 ────────────────────────────────────────────────────

    def _keyword_fallback_name(self, agent: DiscoveredAgent) -> str:
        """
        태스크 이름에서 키워드 빈도를 분석하여 구체적인 에이전트 이름 생성.
        "범용 처리 에이전트" 대신 실제 도메인 기반 이름을 반환합니다.
        """
        all_text = " ".join(agent.task_names)

        keyword_scores: Counter[str] = Counter()
        for keyword in _DOMAIN_KEYWORD_MAP:
            count = all_text.count(keyword)
            if count > 0:
                keyword_scores[keyword] += count

        if keyword_scores:
            top_keyword = keyword_scores.most_common(1)[0][0]
            name_ko, _ = _DOMAIN_KEYWORD_MAP[top_keyword]
            logger.info(f"키워드 폴백 이름 생성: '{top_keyword}' → '{name_ko}'")
            return name_ko

        return f"태스크 그룹 {agent.cluster_id + 1} 에이전트"

    def _keyword_fallback_role(self, agent: DiscoveredAgent) -> str:
        """태스크 목록 기반 역할 설명 생성"""
        if not agent.task_names:
            return f"태스크 그룹 {agent.cluster_id + 1}번의 작업을 처리하는 에이전트"
        sample = agent.task_names[:3]
        return f"주요 태스크: {', '.join(sample)} 등 {len(agent.task_names)}개 작업 처리"

    def _keyword_fallback_prefix(self, agent: DiscoveredAgent) -> str:
        """태스크 이름 기반 snake_case prefix 생성"""
        all_text = " ".join(agent.task_names)
        for keyword, (_, prefix) in _DOMAIN_KEYWORD_MAP.items():
            if keyword in all_text:
                return f"{prefix}_{agent.cluster_id}"
        return f"agent_{agent.cluster_id}"
