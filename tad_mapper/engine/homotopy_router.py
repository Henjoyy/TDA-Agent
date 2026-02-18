"""
Homotopy Router — 수학적 정식화 #2

Φ : Q → {U_Search, U_Stats, ..., U_Report}
Φ(x) = U_k ⟺ [x] = [x_k]

사용자 쿼리 x의 호모토피 클래스 [x]를 분류하여
적절한 Unit Agent U_k로 라우팅합니다.

"자료 찾아줘", "데이터 검색해", "정보 좀 줘"는
표현(x)은 다르지만 호모토피 클래스([x])가 같으므로
모두 동일한 U_Search로 매핑됩니다.
"""
from __future__ import annotations

import logging

import numpy as np

from tad_mapper.engine.embedder import Embedder
from tad_mapper.engine.query_manifold import QueryManifold
from tad_mapper.engine.tda_analyzer import DiscoveredAgent
from tad_mapper.models.topology import (
    HomotopyClass,
    QueryPoint,
    RoutingCandidate,
    RoutingResult,
)

logger = logging.getLogger(__name__)

# 모호성 판단 임계값: top1과 top2 신뢰도 차이가 이 값보다 작으면 모호
AMBIGUITY_THRESHOLD = 0.1
# 라우팅 최소 신뢰도: 이 값 미만이면 "unknown" 처리
MIN_CONFIDENCE = 0.3
SOFTMAX_TEMPERATURE = 0.2
RADIUS_MARGIN = 0.05


class HomotopyRouter:
    """
    Master Agent의 라우팅 함수 Φ를 구현합니다.

    표현은 달라도 본질적 의도가 같은 쿼리들(호모토피 동치류)을
    동일한 Agent로 매핑합니다.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._homotopy_classes: list[HomotopyClass] = []
        self._manifold: QueryManifold | None = None
        self._ready = False

    # ── 초기화 ───────────────────────────────────────────────────────────────

    def build(
        self,
        agents: list[DiscoveredAgent],
        manifold: QueryManifold | None = None,
    ) -> None:
        """
        Agent 목록에서 호모토피 클래스를 구성합니다.

        각 Agent → 하나의 HomotopyClass (대표 의도 클러스터)
        Agent의 centroid 임베딩 = 클래스의 대표 벡터

        Args:
            agents: 발견된 Unit Agent 목록 (suggested_name/role 필요)
            manifold: 구축된 QueryManifold (centroid 임베딩 추출용)
        """
        self._manifold = manifold
        self._homotopy_classes = []

        for agent in agents:
            agent_name = agent.suggested_name or agent.agent_id
            agent_role = agent.suggested_role or "태스크 처리"

            # centroid 임베딩 결정
            # 1순위: QueryManifold의 Coverage Region centroid
            # 2순위: Agent 프로파일 텍스트를 새로 임베딩
            centroid_emb: np.ndarray

            if manifold is not None:
                region = next(
                    (r for r in manifold.regions if r.agent_id == agent.agent_id),
                    None
                )
                if region and region.centroid_embedding:
                    centroid_emb = np.array(region.centroid_embedding)
                    radius = self._adaptive_radius(
                        centroid_emb,
                        [np.array(v) for v in region.task_embeddings],
                        margin=RADIUS_MARGIN,
                        fallback=region.radius if region.radius > 0 else 0.3,
                    )
                else:
                    centroid_emb = self._embedder.embed_agent_profile(
                        agent_name, agent_role, agent.task_names
                    )
                    radius = 0.3
            else:
                centroid_emb = self._embedder.embed_agent_profile(
                    agent_name, agent_role, agent.task_names
                )
                radius = 0.3

            # 호모토피 클래스 생성
            homotopy_class = HomotopyClass(
                class_id=f"class_{agent.agent_id}",
                agent_id=agent.agent_id,
                agent_name=agent_name,
                representative_text=(
                    f"{agent_name}: {agent_role}. "
                    f"처리 태스크: {', '.join(agent.task_names[:5])}"
                ),
                centroid_embedding=centroid_emb.tolist(),
                radius=radius,
            )
            self._homotopy_classes.append(homotopy_class)

        self._ready = True
        logger.info(
            f"HomotopyRouter 구축 완료: {len(self._homotopy_classes)}개 호모토피 클래스"
        )
        for hc in self._homotopy_classes:
            logger.info(f"  [{hc.class_id}] → {hc.agent_name}")

    # ── 핵심 라우팅 함수 Φ ──────────────────────────────────────────────────

    def route(self, query_text: str) -> RoutingResult:
        """
        라우팅 함수 Φ(x) = U_k 실행.

        쿼리 텍스트를 임베딩하고 호모토피 클래스를 분류하여
        적절한 Unit Agent로 라우팅합니다.

        Args:
            query_text: 사용자 입력 쿼리

        Returns:
            RoutingResult: 라우팅 결과 (target_agent, confidence, alternatives)
        """
        if not self._ready or not self._homotopy_classes:
            raise RuntimeError(
                "HomotopyRouter가 초기화되지 않았습니다. build()를 먼저 호출하세요."
            )

        # 1. 쿼리 임베딩 → QueryPoint 생성
        query_embedding = self._embedder.embed_query(query_text)
        query_point = QueryPoint(
            text=query_text,
            embedding=query_embedding.tolist(),
        )

        # 2. 모든 호모토피 클래스와 코사인 유사도 계산
        similarities: list[tuple[HomotopyClass, float]] = []
        for hc in self._homotopy_classes:
            centroid = hc.centroid_array()
            sim = Embedder.cosine_similarity(query_embedding, centroid)
            similarities.append((hc, sim))

        # 유사도 내림차순 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 3. Top-1 = 라우팅 대상
        best_class, best_sim = similarities[0]
        routing_probs = self._softmax_probabilities(
            query_embedding, temperature=SOFTMAX_TEMPERATURE
        )

        # 4. confidence = Top-1과 Top-2의 마진 기반
        confidence = self._compute_confidence(similarities, best_class, best_sim)

        # 5. 호모토피 클래스 할당
        query_point.homotopy_class_id = best_class.class_id

        # 6. 대안 Agent 목록 (Top-2~4)
        alternatives = [
            RoutingCandidate(
                agent_id=hc.agent_id,
                agent_name=hc.agent_name,
                similarity=float(sim),
                homotopy_class_id=hc.class_id,
            )
            for hc, sim in similarities[1:4]
        ]

        # 7. 모호성 판단
        is_ambiguous, ambiguity_reason = self._check_ambiguity(
            similarities, confidence
        )

        result = RoutingResult(
            query_text=query_text,
            target_agent_id=best_class.agent_id,
            target_agent_name=best_class.agent_name,
            homotopy_class_id=best_class.class_id,
            confidence=confidence,
            top_similarity=float(best_sim),
            routing_probabilities=routing_probs,
            alternatives=alternatives,
            is_ambiguous=is_ambiguous,
            ambiguity_reason=ambiguity_reason,
        )

        logger.info(
            f"라우팅 완료: '{query_text[:40]}' "
            f"→ {best_class.agent_name} (신뢰도: {confidence:.2f})"
        )
        return result

    def route_soft(
        self, query_text: str, temperature: float = SOFTMAX_TEMPERATURE
    ) -> dict[str, float]:
        """
        확률적 soft routing.
        P(U_k|x) = softmax(sim(x, c_k) / tau)
        """
        if not self._ready or not self._homotopy_classes:
            raise RuntimeError(
                "HomotopyRouter가 초기화되지 않았습니다. build()를 먼저 호출하세요."
            )
        query_embedding = self._embedder.embed_query(query_text)
        return self._softmax_probabilities(query_embedding, temperature=temperature)

    def classify(self, query_embedding: np.ndarray) -> HomotopyClass | None:
        """
        임베딩 벡터로 호모토피 클래스를 직접 분류합니다.

        route()가 텍스트 → 임베딩 → 분류를 모두 하는 반면,
        이 메서드는 이미 임베딩된 벡터에서 분류만 수행합니다.
        """
        if not self._homotopy_classes:
            return None

        best_class = None
        best_sim = -1.0

        for hc in self._homotopy_classes:
            centroid = hc.centroid_array()
            sim = Embedder.cosine_similarity(query_embedding, centroid)
            if sim > best_sim:
                best_sim = sim
                best_class = hc

        return best_class

    # ── 내부 계산 ─────────────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        similarities: list[tuple[HomotopyClass, float]],
        best_class: HomotopyClass,
        best_sim: float,
    ) -> float:
        """
        라우팅 신뢰도 계산.

        confidence = (top1_sim - top2_sim) / top1_sim (마진 기반)
        마진이 클수록 top1이 압도적으로 적합 → 높은 신뢰도
        """
        if len(similarities) < 2:
            return float(np.clip(best_sim, 0.0, 1.0)) if similarities else 0.0

        top1_sim = similarities[0][1]
        top2_sim = similarities[1][1]

        if top1_sim <= 0:
            return 0.0

        # 마진 기반 신뢰도
        margin = (top1_sim - top2_sim) / (top1_sim + 1e-8)
        margin_score = margin * np.clip(top1_sim, 0.0, 1.0)

        # 적응형 반경 기반 보정: 클래스 반경 안쪽일수록 신뢰도 강화
        radius = max(best_class.radius, 1e-6)
        distance = 1.0 - np.clip(best_sim, -1.0, 1.0)
        radius_score = float(np.clip(1.0 - (distance / radius), 0.0, 1.0))

        confidence = 0.7 * margin_score + 0.3 * radius_score
        return float(np.clip(confidence, 0.0, 1.0))

    def _softmax_probabilities(
        self, query_embedding: np.ndarray, temperature: float = SOFTMAX_TEMPERATURE
    ) -> dict[str, float]:
        temp = max(1e-4, temperature)
        sims = []
        for hc in self._homotopy_classes:
            sim = Embedder.cosine_similarity(query_embedding, hc.centroid_array())
            sims.append((hc.agent_id, sim))

        logits = np.array([sim / temp for _, sim in sims], dtype=float)
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits)
        probs_sum = probs.sum()
        if probs_sum <= 0:
            return {aid: 0.0 for aid, _ in sims}
        probs = probs / probs_sum
        return {aid: float(p) for (aid, _), p in zip(sims, probs)}

    @staticmethod
    def _adaptive_radius(
        centroid_emb: np.ndarray,
        task_embeddings: list[np.ndarray],
        margin: float = RADIUS_MARGIN,
        fallback: float = 0.3,
    ) -> float:
        if not task_embeddings:
            return float(np.clip(fallback, 0.05, 1.0))

        distances = [
            1.0 - Embedder.cosine_similarity(centroid_emb, emb)
            for emb in task_embeddings
        ]
        mean_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))
        radius = mean_dist + std_dist + margin
        return float(np.clip(radius, 0.05, 1.0))

    def _check_ambiguity(
        self,
        similarities: list[tuple[HomotopyClass, float]],
        confidence: float,
    ) -> tuple[bool, str]:
        """모호성 여부와 원인 반환"""
        if confidence < MIN_CONFIDENCE:
            return True, f"쿼리의 최대 유사도({similarities[0][1]:.2f})가 너무 낮습니다."

        if len(similarities) >= 2:
            top1_sim = similarities[0][1]
            top2_sim = similarities[1][1]
            if (top1_sim - top2_sim) < AMBIGUITY_THRESHOLD:
                return True, (
                    f"상위 두 Agent의 유사도 차이가 작습니다 "
                    f"({similarities[0][0].agent_name}: {top1_sim:.2f} vs "
                    f"{similarities[1][0].agent_name}: {top2_sim:.2f})"
                )

        return False, ""

    @property
    def homotopy_classes(self) -> list[HomotopyClass]:
        return self._homotopy_classes

    @property
    def is_ready(self) -> bool:
        return self._ready
