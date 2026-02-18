"""
LLM(Gemini) 기반 Feature Extractor
각 AI 적용 기회(태스크)에서 위상학적 특징 벡터를 추출합니다.

배치 처리 방식:
- N개 태스크를 단 1회 LLM 호출로 처리 (순차 N회 → 배치 1회)
- 응답 파싱 실패 시 해당 태스크만 태그 기반 fallback으로 복구
"""
from __future__ import annotations

import json
import logging
import re

import google.genai as genai
from google.genai import types
import numpy as np
from pydantic import BaseModel, Field

from config.settings import GEMINI_API_KEY, GEMINI_MODEL, PROJECT_ROOT
from tad_mapper.models.journey import TaskStep

logger = logging.getLogger(__name__)

SKILLS_PATH = PROJECT_ROOT / "tad_mapper" / "resources" / "skills.md"


class TopologicalFeature(BaseModel):
    """태스크의 위상학적 특징 벡터 (각 값 0.0 ~ 1.0)"""
    task_id: str
    task_name: str

    # ── 핵심 특징 차원 ──────────────────────────────────────
    data_type: float = Field(..., ge=0.0, le=1.0,
        description="데이터 성격: 0=비정형(텍스트/이미지), 1=정형(DB/수치)")
    reasoning_depth: float = Field(..., ge=0.0, le=1.0,
        description="추론 깊이: 0=단순조회, 0.5=분석, 1=복합추론/창의")
    automation_potential: float = Field(..., ge=0.0, le=1.0,
        description="자동화 가능성: 0=사람 개입 필수, 1=완전 자동화 가능")
    interaction_type: float = Field(..., ge=0.0, le=1.0,
        description="상호작용 유형: 0=읽기전용, 1=외부 시스템 쓰기/실행")
    output_complexity: float = Field(..., ge=0.0, le=1.0,
        description="출력 복잡도: 0=단순값/불리언, 1=복합 문서/보고서")
    domain_specificity: float = Field(..., ge=0.0, le=1.0,
        description="도메인 특화도: 0=범용, 1=고도 전문 지식 필요")
    temporal_sensitivity: float = Field(..., ge=0.0, le=1.0,
        description="시간 민감도: 0=배치/지연 허용, 1=실시간 대응 필요")
    data_volume: float = Field(..., ge=0.0, le=1.0,
        description="데이터 규모: 0=단건/소량, 1=대용량/고처리량")
    security_level: float = Field(..., ge=0.0, le=1.0,
        description="보안 수준: 0=공개 데이터, 1=보안/규제 중요")
    state_dependency: float = Field(..., ge=0.0, le=1.0,
        description="상태 의존성: 0=stateless, 1=세션/워크플로 상태 강함")

    @property
    def vector(self) -> np.ndarray:
        """특징 벡터를 numpy 배열로 반환"""
        return np.array([
            self.data_type,
            self.reasoning_depth,
            self.automation_potential,
            self.interaction_type,
            self.output_complexity,
            self.domain_specificity,
            self.temporal_sensitivity,
            self.data_volume,
            self.security_level,
            self.state_dependency,
        ])


# ── 배치 프롬프트 ─────────────────────────────────────────────────────────────

_BATCH_PROMPT_HEADER = """\
당신은 AI 시스템 설계 전문가입니다.
아래 {n}개의 AI 적용 기회(태스크)를 **한꺼번에** 분석하여 각각의 위상학적 특징 벡터를 추출해주세요.

## 분석 가이드라인 (Skills)
{skills}

## 평가 기준 (각 항목: 0.0 ~ 1.0)
1. data_type          : 데이터 성격 (0=비정형/텍스트/이미지, 1=정형/DB/수치/API)
2. reasoning_depth    : 추론 깊이 (0=단순 조회/필터, 0.5=통계/분석, 1=복합추론/예측/창의)
3. automation_potential: 자동화 가능성 (0=사람 판단 필수, 1=규칙 기반 완전 자동화)
4. interaction_type   : 상호작용 (0=읽기/조회만, 1=외부 시스템 쓰기/실행/알림)
5. output_complexity  : 출력 복잡도 (0=단순 수치/예/아니오, 1=복합 보고서/문서/시각화)
6. domain_specificity : 도메인 특화 (0=누구나 이해 가능, 1=전문가 지식 필수)
7. temporal_sensitivity: 시간 민감도 (0=배치 처리 가능, 1=실시간 응답 필수)
8. data_volume        : 데이터 규모 (0=소량, 1=대용량/고처리량)
9. security_level     : 보안 중요도 (0=공개 데이터, 1=민감/규제 데이터)
10. state_dependency  : 상태 의존성 (0=stateless, 1=세션/트랜잭션 상태 강함)

## 태스크 목록
{task_list}

## 출력 형식 (반드시 아래 JSON 배열 형식으로만 응답하세요. 마크다운 코드블록 없이)
[
  {{
    "task_id": "task_001",
    "data_type": 0.0,
    "reasoning_depth": 0.0,
    "automation_potential": 0.0,
    "interaction_type": 0.0,
    "output_complexity": 0.0,
    "domain_specificity": 0.0,
    "temporal_sensitivity": 0.0,
    "data_volume": 0.0,
    "security_level": 0.0,
    "state_dependency": 0.0
  }},
  ...
]
"""

_TASK_ITEM_TEMPLATE = """\
[{idx}] task_id={task_id}
  이름: {name}
  설명: {description}
  입력: {input_data}
  출력: {output_data}
  태그: {tags}"""


class FeatureExtractor:
    """Gemini LLM을 사용하여 태스크에서 위상학적 특징 벡터 배치 추출"""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._skills = ""
        if SKILLS_PATH.exists():
            self._skills = SKILLS_PATH.read_text(encoding="utf-8")
        else:
            logger.warning(f"Skills file not found at {SKILLS_PATH}")

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def extract(self, tasks: list[TaskStep]) -> list[TopologicalFeature]:
        """
        태스크 목록에서 특징 벡터 배치 추출.

        N개 태스크를 단 1회 LLM 호출로 처리합니다.
        파싱 실패한 태스크는 태그 기반 fallback으로 개별 복구합니다.
        """
        if not tasks:
            return []

        logger.info(f"[배치 특징 추출] {len(tasks)}개 태스크를 1회 LLM 호출로 처리 시작...")

        # LLM 배치 호출
        llm_results: dict[str, dict] = self._extract_batch(tasks)

        # 결과 조합 (LLM 성공 → 파싱, 실패 → fallback)
        features: list[TopologicalFeature] = []
        success_count = 0
        for task in tasks:
            raw_data = llm_results.get(task.id)
            if raw_data:
                try:
                    feature = TopologicalFeature(
                        task_id=task.id,
                        task_name=task.name,
                        data_type=float(raw_data["data_type"]),
                        reasoning_depth=float(raw_data["reasoning_depth"]),
                        automation_potential=float(raw_data["automation_potential"]),
                        interaction_type=float(raw_data["interaction_type"]),
                        output_complexity=float(raw_data["output_complexity"]),
                        domain_specificity=float(raw_data["domain_specificity"]),
                        temporal_sensitivity=float(raw_data["temporal_sensitivity"]),
                        data_volume=float(raw_data["data_volume"]),
                        security_level=float(raw_data["security_level"]),
                        state_dependency=float(raw_data["state_dependency"]),
                    )
                    features.append(feature)
                    success_count += 1
                    continue
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"  [{task.id}] 파싱 오류: {e} → fallback 사용")

            # fallback
            features.append(self._fallback_feature(task))

        logger.info(
            f"[배치 특징 추출 완료] 성공: {success_count}/{len(tasks)}개 "
            f"(fallback: {len(tasks) - success_count}개)"
        )
        return features

    # ── 내부 메서드 ───────────────────────────────────────────────────────────

    def _extract_batch(self, tasks: list[TaskStep]) -> dict[str, dict]:
        """
        모든 태스크를 단 1회 LLM 호출로 처리.

        Returns:
            {task_id: {data_type, reasoning_depth, ...}} 딕셔너리.
            파싱 실패 시 빈 딕셔너리 반환.
        """
        # 태스크 목록 텍스트 구성
        task_items = []
        for idx, task in enumerate(tasks, start=1):
            task_items.append(_TASK_ITEM_TEMPLATE.format(
                idx=idx,
                task_id=task.id,
                name=task.name,
                description=task.description or "미지정",
                input_data=", ".join(task.input_data) or "미지정",
                output_data=", ".join(task.output_data) or "미지정",
                tags=", ".join(task.tags) or "없음",
            ))

        prompt = _BATCH_PROMPT_HEADER.format(
            n=len(tasks),
            skills=self._skills,
            task_list="\n\n".join(task_items),
        )

        try:
            logger.info(f"  LLM 배치 호출 중 (모델: {GEMINI_MODEL})...")
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            raw = response.text.strip()
            logger.info("  LLM 응답 수신 완료. JSON 파싱 중...")
            return self._parse_batch_response(raw, tasks)

        except Exception as e:
            logger.warning(f"  LLM 배치 호출 실패: {e}. 전체 태스크 fallback 처리.")
            return {}

    def _parse_batch_response(
        self, raw: str, tasks: list[TaskStep]
    ) -> dict[str, dict]:
        """
        LLM 응답(JSON 배열)을 파싱하여 {task_id: data} 딕셔너리로 변환.

        마크다운 코드블록, 앞뒤 텍스트 등을 자동으로 제거합니다.
        """
        # 마크다운 코드블록 제거
        if "```" in raw:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
            if match:
                raw = match.group(1)

        # JSON 배열 추출 (앞뒤 불필요한 텍스트 제거)
        array_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not array_match:
            logger.warning("  배치 응답에서 JSON 배열을 찾을 수 없습니다.")
            return {}

        try:
            items: list[dict] = json.loads(array_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"  JSON 파싱 실패: {e}")
            return {}

        # task_id 기준으로 매핑
        result: dict[str, dict] = {}
        task_ids = [t.id for t in tasks]

        for item in items:
            tid = item.get("task_id", "").strip()
            if tid in task_ids:
                result[tid] = item
            else:
                # task_id가 없거나 순서 기반 매핑 시도
                logger.debug(f"  task_id 불일치: '{tid}' → 순서 기반 매핑 시도")

        # task_id 매핑 실패 시 순서 기반 fallback 매핑
        if len(result) < len(tasks) // 2:
            logger.warning(
                f"  task_id 매핑 성공률 낮음 ({len(result)}/{len(tasks)}). "
                "순서 기반 매핑으로 전환..."
            )
            result = {}
            for task, item in zip(tasks, items):
                result[task.id] = item

        logger.info(f"  파싱 성공: {len(result)}/{len(tasks)}개 태스크")
        return result

    @staticmethod
    def _fallback_feature(task: TaskStep) -> TopologicalFeature:
        """LLM 실패 시 태그 기반 휴리스틱 특징 추출"""
        tags_lower = [t.lower() for t in task.tags]
        return TopologicalFeature(
            task_id=task.id,
            task_name=task.name,
            data_type=0.8 if any(t in tags_lower for t in ["db", "api", "data", "정형"]) else 0.3,
            reasoning_depth=0.8 if any(t in tags_lower for t in ["분석", "추론", "예측"]) else 0.3,
            automation_potential=0.7 if any(t in tags_lower for t in ["자동화", "규칙"]) else 0.4,
            interaction_type=0.8 if any(t in tags_lower for t in ["쓰기", "실행", "알림"]) else 0.2,
            output_complexity=0.7 if any(t in tags_lower for t in ["보고서", "문서", "시각화"]) else 0.3,
            domain_specificity=0.7 if any(t in tags_lower for t in ["전문", "무역", "금융"]) else 0.3,
            temporal_sensitivity=0.85 if any(t in tags_lower for t in ["실시간", "모니터링", "스트리밍", "알림"]) else 0.35,
            data_volume=0.8 if any(t in tags_lower for t in ["대용량", "배치", "빅데이터", "로그"]) else 0.4,
            security_level=0.85 if any(t in tags_lower for t in ["보안", "규정", "컴플라이언스", "민감", "개인정보"]) else 0.3,
            state_dependency=0.8 if any(t in tags_lower for t in ["세션", "상태", "트랜잭션", "워크플로", "대화"]) else 0.25,
        )

    @staticmethod
    def to_matrix(features: list[TopologicalFeature]) -> np.ndarray:
        """특징 목록을 행렬로 변환 (shape: [n_tasks, 10])"""
        return np.stack([f.vector for f in features])
