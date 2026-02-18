"""
LLM(Gemini) 기반 Feature Extractor
각 AI 적용 기회(태스크)에서 위상학적 특징 벡터를 추출합니다.
"""
from __future__ import annotations

import json
import logging
import re

import google.genai as genai
from google.genai import types
import numpy as np
from pydantic import BaseModel, Field

from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from tad_mapper.models.journey import TaskStep

logger = logging.getLogger(__name__)


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
        ])


_EXTRACTION_PROMPT = """
당신은 AI 시스템 설계 전문가입니다.
아래 AI 적용 기회(태스크)를 분석하여 위상학적 특징 벡터를 추출해주세요.

각 항목을 0.0 ~ 1.0 사이의 실수로 평가하세요.

태스크 정보:
- 이름: {name}
- 설명: {description}
- 입력 데이터: {input_data}
- 출력 데이터: {output_data}
- 태그: {tags}

평가 기준:
1. data_type: 데이터 성격 (0=비정형/텍스트/이미지, 1=정형/DB/수치/API)
2. reasoning_depth: 추론 깊이 (0=단순 조회/필터, 0.5=통계/분석, 1=복합추론/예측/창의)
3. automation_potential: 자동화 가능성 (0=사람 판단 필수, 1=규칙 기반 완전 자동화)
4. interaction_type: 상호작용 (0=읽기/조회만, 1=외부 시스템 쓰기/실행/알림)
5. output_complexity: 출력 복잡도 (0=단순 수치/예/아니오, 1=복합 보고서/문서/시각화)
6. domain_specificity: 도메인 특화 (0=누구나 이해 가능, 1=전문가 지식 필수)

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "data_type": 0.0,
  "reasoning_depth": 0.0,
  "automation_potential": 0.0,
  "interaction_type": 0.0,
  "output_complexity": 0.0,
  "domain_specificity": 0.0,
  "reasoning": "평가 근거를 한 문장으로"
}}
"""


class FeatureExtractor:
    """Gemini LLM을 사용하여 태스크에서 위상학적 특징 벡터 추출"""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    def extract(self, tasks: list[TaskStep]) -> list[TopologicalFeature]:
        """태스크 목록에서 특징 벡터 추출"""
        features: list[TopologicalFeature] = []
        for task in tasks:
            logger.info(f"특징 추출 중: [{task.id}] {task.name}")
            feature = self._extract_single(task)
            features.append(feature)
        return features

    def _extract_single(self, task: TaskStep) -> TopologicalFeature:
        """단일 태스크 특징 추출"""
        prompt = _EXTRACTION_PROMPT.format(
            name=task.name,
            description=task.description,
            input_data=", ".join(task.input_data) or "미지정",
            output_data=", ".join(task.output_data) or "미지정",
            tags=", ".join(task.tags) or "없음",
        )

        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            raw = response.text.strip()
            # JSON 블록 추출 (마크다운 코드블록 처리)
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                raise ValueError("JSON 응답을 찾을 수 없습니다.")
            data = json.loads(json_match.group())

            return TopologicalFeature(
                task_id=task.id,
                task_name=task.name,
                data_type=float(data["data_type"]),
                reasoning_depth=float(data["reasoning_depth"]),
                automation_potential=float(data["automation_potential"]),
                interaction_type=float(data["interaction_type"]),
                output_complexity=float(data["output_complexity"]),
                domain_specificity=float(data["domain_specificity"]),
            )
        except Exception as e:
            logger.warning(f"LLM 특징 추출 실패 ({task.id}): {e}. 기본값 사용.")
            return self._fallback_feature(task)

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
        )

    @staticmethod
    def to_matrix(features: list[TopologicalFeature]) -> np.ndarray:
        """특징 목록을 행렬로 변환 (shape: [n_tasks, 6])"""
        return np.stack([f.vector for f in features])
