"""
Gemini Embedding 기반 벡터 생성기
태스크 설명을 의미론적 임베딩 벡터로 변환합니다.

확장된 기능:
- embed_query()          : 단일 사용자 쿼리 임베딩 (라우팅용)
- embed_batch()          : 배치 임베딩
- embed_agent_profile()  : Agent 프로파일 텍스트 임베딩 (호모토피 클래스 centroid)
- cosine_similarity()    : 두 벡터 유사도 계산
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, replace

import google.genai as genai
import numpy as np

from config.settings import (
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_CANDIDATES,
    GEMINI_API_KEY,
)
from tad_mapper.models.journey import TaskStep

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingHealth:
    active_model: str
    model_available: bool
    model_validation_error: str = ""
    total_calls: int = 0
    fallback_calls: int = 0
    last_error: str = ""

    @property
    def fallback_ratio(self) -> float:
        if self.total_calls <= 0:
            return 0.0
        return self.fallback_calls / self.total_calls

    def to_dict(self) -> dict:
        data = asdict(self)
        data["fallback_ratio"] = round(self.fallback_ratio, 4)
        return data


class Embedder:
    """Gemini text-embedding-004로 태스크/쿼리 텍스트 임베딩 생성"""

    EMBEDDING_DIM = 768  # text-embedding-004 출력 차원
    _FALLBACK_WARN_FIRST = 3
    _FALLBACK_WARN_EVERY = 25

    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._candidate_models = list(dict.fromkeys(
            [EMBEDDING_MODEL, *EMBEDDING_MODEL_CANDIDATES]
        ))
        self._health = EmbeddingHealth(
            active_model=EMBEDDING_MODEL,
            model_available=True,
        )
        self._fallback_warn_count = 0
        self._validate_embedding_model()

    # ── 기존 메서드 ──────────────────────────────────────────────────────────

    def embed_tasks(self, tasks: list[TaskStep]) -> np.ndarray:
        """
        태스크 목록을 임베딩 행렬로 변환.
        Returns: shape [n_tasks, embedding_dim]
        """
        embeddings: list[list[float]] = []
        for task in tasks:
            text = self._task_to_text(task)
            logger.info(f"임베딩 생성 중: [{task.id}] {task.name}")
            vec = self._embed_text(text)
            embeddings.append(vec)
        return np.array(embeddings)

    # ── 신규 메서드 ──────────────────────────────────────────────────────────

    def embed_query(self, query_text: str) -> np.ndarray:
        """
        사용자 쿼리 단일 텍스트 임베딩.
        Homotopy Router의 Φ(x) 함수 입력으로 사용.

        Returns: shape [EMBEDDING_DIM]
        """
        preview = query_text[:50] + "..." if len(query_text) > 50 else query_text
        logger.info(f"쿼리 임베딩 생성: '{preview}'")
        prefixed = f"사용자 요청: {query_text}"
        return np.array(self._embed_text(prefixed))

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        텍스트 목록 배치 임베딩.

        Returns: shape [n_texts, EMBEDDING_DIM]
        """
        embeddings: list[list[float]] = []
        for text in texts:
            vec = self._embed_text(text)
            embeddings.append(vec)
        return np.array(embeddings)

    def embed_agent_profile(self, agent_name: str, agent_role: str,
                             task_names: list[str]) -> np.ndarray:
        """
        Agent 프로파일 전체를 하나의 임베딩 벡터로 변환.
        HomotopyClass의 centroid로 사용됨.

        Returns: shape [EMBEDDING_DIM]
        """
        profile_text = self._build_agent_profile_text(agent_name, agent_role, task_names)
        logger.info(f"Agent 프로파일 임베딩: {agent_name}")
        return np.array(self._embed_text(profile_text))

    # ── 내부 유틸리티 ────────────────────────────────────────────────────────

    def _embed_text(self, text: str) -> list[float]:
        """단일 텍스트 임베딩 (API 호출)"""
        self._health.total_calls += 1
        if not self._health.model_available:
            self._health.fallback_calls += 1
            return self._deterministic_fallback_vector(text)

        try:
            return self._request_embedding(text, model=self._health.active_model)
        except Exception as e:
            err = str(e)
            self._health.last_error = err

            # 모델 미지원 시 후보 모델 자동 전환
            if self._is_model_not_found_error(e):
                switched = self._try_switch_embedding_model(text)
                if switched is not None:
                    return switched
                self._health.model_available = False
                self._health.model_validation_error = err

            self._health.fallback_calls += 1
            self._fallback_warn_count += 1
            if self._should_log_fallback_warning(self._fallback_warn_count):
                logger.warning(
                    "임베딩 생성 실패: %s. 결정적 fallback 벡터 사용. (count=%s)",
                    e,
                    self._fallback_warn_count,
                )
            return self._deterministic_fallback_vector(text)

    def _request_embedding(self, text: str, model: str) -> list[float]:
        result = self._client.models.embed_content(
            model=model,
            contents=text,
        )
        return result.embeddings[0].values

    def _validate_embedding_model(self) -> None:
        """
        초기화 시 임베딩 모델 동작 여부를 점검하고,
        미지원 모델이면 자동으로 대체 모델을 탐색합니다.
        """
        probe = "embedding model validation"
        try:
            _ = self._request_embedding(probe, model=self._health.active_model)
            self._health.model_available = True
            return
        except Exception as e:
            if not self._is_model_not_found_error(e):
                logger.warning(
                    "임베딩 모델 초기 검증 실패(일시적 가능): %s", e
                )
                return

            logger.warning(
                "임베딩 모델 '%s' 사용 불가. 대체 모델 탐색 시작...",
                self._health.active_model,
            )
            switched = self._try_switch_embedding_model(probe)
            if switched is not None:
                return

            self._health.model_available = False
            self._health.model_validation_error = str(e)
            logger.error(
                "지원되는 임베딩 모델을 찾지 못했습니다. fallback 모드로 동작합니다: %s",
                e,
            )

    def _try_switch_embedding_model(self, text: str) -> list[float] | None:
        current = self._health.active_model
        for candidate in self._candidate_models:
            if candidate == current:
                continue
            try:
                vec = self._request_embedding(text, model=candidate)
                self._health.active_model = candidate
                self._health.model_available = True
                self._health.model_validation_error = ""
                logger.warning(
                    "임베딩 모델 자동 전환: '%s' → '%s'", current, candidate
                )
                return vec
            except Exception as e:
                self._health.last_error = str(e)
                continue
        return None

    @staticmethod
    def _is_model_not_found_error(error: Exception) -> bool:
        message = str(error).lower()
        return "not_found" in message or "is not found" in message or "not supported" in message

    @classmethod
    def _should_log_fallback_warning(cls, count: int) -> bool:
        if count <= cls._FALLBACK_WARN_FIRST:
            return True
        return count % cls._FALLBACK_WARN_EVERY == 0

    @staticmethod
    def _task_to_text(task: TaskStep) -> str:
        """태스크를 임베딩용 텍스트로 변환"""
        parts = [
            f"태스크: {task.name}",
            f"설명: {task.description}",
        ]
        if task.input_data:
            parts.append(f"입력: {', '.join(task.input_data)}")
        if task.output_data:
            parts.append(f"출력: {', '.join(task.output_data)}")
        if task.tags:
            parts.append(f"태그: {', '.join(task.tags)}")
        return " | ".join(parts)

    @staticmethod
    def _build_agent_profile_text(agent_name: str, agent_role: str,
                                   task_names: list[str]) -> str:
        """Agent 프로파일 텍스트 생성"""
        tasks_str = ", ".join(task_names[:10])  # 최대 10개 태스크만
        return (
            f"에이전트: {agent_name} | "
            f"역할: {agent_role} | "
            f"처리 태스크: {tasks_str}"
        )

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """두 벡터의 코사인 유사도 계산"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @classmethod
    def _deterministic_fallback_vector(cls, text: str) -> list[float]:
        """
        텍스트 해시 기반 결정적 fallback 벡터.
        외부 API 실패 시에도 재현 가능한 결과를 유지합니다.
        """
        seed = int.from_bytes(
            hashlib.sha256(text.encode("utf-8")).digest()[:8], "big"
        )
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(cls.EMBEDDING_DIM)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def reset_health_stats(self) -> None:
        """요청 단위 통계 초기화 (모델 상태는 유지)"""
        self._health.total_calls = 0
        self._health.fallback_calls = 0
        self._health.last_error = ""

    def get_health(self) -> EmbeddingHealth:
        return replace(self._health)

    def get_health_dict(self) -> dict:
        return self._health.to_dict()
