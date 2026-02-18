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

import logging

import google.genai as genai
import numpy as np

from config.settings import GEMINI_API_KEY, EMBEDDING_MODEL
from tad_mapper.models.journey import TaskStep

logger = logging.getLogger(__name__)


class Embedder:
    """Gemini text-embedding-004로 태스크/쿼리 텍스트 임베딩 생성"""

    EMBEDDING_DIM = 768  # text-embedding-004 출력 차원

    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY)

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
        try:
            result = self._client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
            )
            return result.embeddings[0].values
        except Exception as e:
            logger.warning(f"임베딩 생성 실패: {e}. 랜덤 벡터 사용.")
            return list(np.random.randn(self.EMBEDDING_DIM).tolist())

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
