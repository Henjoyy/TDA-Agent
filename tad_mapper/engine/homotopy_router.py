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
import re
from collections import defaultdict

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
MEMBER_SOFTMAX_TEMPERATURE = 0.35
TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]{2,}")
TOKEN_NORMALIZATION = {
    "리포트": "보고서",
    "레포트": "보고서",
    "통계": "분석",
    "모니터링": "감시",
    "forecast": "예측",
    "report": "보고서",
    "search": "검색",
    "query": "조회",
    "작성": "생성",
    "작성해줘": "생성",
    "만들어줘": "생성",
    "알려줘": "조회",
    "추천해줘": "추천",
    "발주": "주문",
}


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
        self._class_members: dict[str, list[DiscoveredAgent]] = {}
        self._agent_centroids: dict[str, np.ndarray] = {}
        self._class_tokens: dict[str, set[str]] = {}
        self._agent_tokens: dict[str, set[str]] = {}
        self._member_task_tokens: dict[str, list[set[str]]] = {}
        self._class_hub_agent_id: dict[str, str] = {}
        self._class_task_total: dict[str, int] = {}
        self._member_class_id: dict[str, str] = {}
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
        self._class_members = {}
        self._agent_centroids = {}
        self._class_tokens = {}
        self._agent_tokens = {}
        self._member_task_tokens = {}
        self._class_hub_agent_id = {}
        self._class_task_total = {}
        self._member_class_id = {}

        region_by_agent = {}
        if manifold is not None:
            region_by_agent = {r.agent_id: r for r in manifold.regions}

        # 1) 에이전트별 중심 임베딩 계산
        for agent in agents:
            agent_name = agent.suggested_name or agent.agent_id
            agent_role = agent.suggested_role or "태스크 처리"
            region = region_by_agent.get(agent.agent_id)

            if region and region.centroid_embedding:
                centroid_emb = np.array(region.centroid_embedding)
            else:
                centroid_emb = self._embedder.embed_agent_profile(
                    agent_name, agent_role, agent.task_names
                )
            self._agent_centroids[agent.agent_id] = centroid_emb
            capabilities_text = " ".join(agent.suggested_capabilities[:8])
            agent_text = (
                f"{agent_name} {agent_role} "
                f"{' '.join(agent.task_names)} "
                f"{capabilities_text} {agent.tool_prefix}"
            )
            self._agent_tokens[agent.agent_id] = self._tokenize(agent_text)
            self._member_task_tokens[agent.agent_id] = [
                self._tokenize(task_name) for task_name in agent.task_names if task_name
            ]

        # 2) 운영 split과 무관한 semantic group 구성
        grouped: dict[str, list[DiscoveredAgent]] = defaultdict(list)
        for agent in agents:
            group_id = agent.routing_group_id or agent.agent_id
            grouped[group_id].append(agent)

        for group_id, members in grouped.items():
            primary = members[0]
            representative_agent_id = primary.agent_id
            representative_name = primary.suggested_name or primary.agent_id
            representative_role = primary.suggested_role or "태스크 처리"
            hub_member = max(members, key=lambda m: len(m.task_ids))
            self._class_hub_agent_id[f"class_{group_id}"] = hub_member.agent_id
            self._class_task_total[f"class_{group_id}"] = max(
                1, sum(len(m.task_ids) for m in members)
            )

            member_centroids = np.array(
                [self._agent_centroids[m.agent_id] for m in members]
            )
            group_centroid = member_centroids.mean(axis=0)

            group_task_embeddings: list[np.ndarray] = []
            fallback_radius = 0.3
            if manifold is not None:
                region_radii = []
                for m in members:
                    region = region_by_agent.get(m.agent_id)
                    if not region:
                        continue
                    group_task_embeddings.extend(
                        np.array(v) for v in region.task_embeddings
                    )
                    if region.radius > 0:
                        region_radii.append(region.radius)
                if region_radii:
                    fallback_radius = float(np.mean(region_radii))

            radius = self._adaptive_radius(
                group_centroid,
                group_task_embeddings,
                margin=RADIUS_MARGIN,
                fallback=fallback_radius,
            )

            all_task_names: list[str] = []
            for member in members:
                for task_name in member.task_names:
                    if task_name not in all_task_names:
                        all_task_names.append(task_name)
            representative_text = (
                f"{representative_name}: {representative_role}. "
                f"처리 태스크: {', '.join(all_task_names)}"
            )

            homotopy_class = HomotopyClass(
                class_id=f"class_{group_id}",
                agent_id=representative_agent_id,
                agent_name=representative_name,
                representative_text=representative_text,
                centroid_embedding=group_centroid.tolist(),
                radius=radius,
            )
            self._homotopy_classes.append(homotopy_class)
            self._class_members[homotopy_class.class_id] = members
            self._class_tokens[homotopy_class.class_id] = self._tokenize(
                representative_text
            )
            for member in members:
                self._member_class_id[member.agent_id] = homotopy_class.class_id

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

        # 1. 쿼리 임베딩 + 토큰화
        query_embedding = self._embedder.embed_query(query_text)
        query_tokens = self._tokenize(query_text)
        query_point = QueryPoint(
            text=query_text,
            embedding=query_embedding.tolist(),
        )

        # 2. 모든 호모토피 클래스와 hybrid 유사도 계산
        # tuple: (class, hybrid_sim, embed_sim, lexical_sim)
        similarities: list[tuple[HomotopyClass, float, float, float]] = []
        for hc in self._homotopy_classes:
            embed_sim = self._embedding_similarity(query_embedding, hc.centroid_array())
            lexical_sim = self._lexical_similarity(
                query_tokens, self._class_tokens.get(hc.class_id, set())
            )
            sim = self._combine_similarity(embed_sim, lexical_sim)
            similarities.append((hc, sim, embed_sim, lexical_sim))

        # 유사도 내림차순 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 3. Top-1 = 라우팅 대상
        best_class, best_sim, best_embed_sim, _ = similarities[0]
        target_agent = self._select_target_agent(
            best_class, query_embedding, query_tokens
        )
        routing_probs = self._softmax_probabilities(
            query_embedding,
            query_tokens,
            temperature=SOFTMAX_TEMPERATURE,
        )

        # 4. confidence = Top-1과 Top-2의 마진 기반
        confidence = self._compute_confidence(
            similarities, best_class, best_sim, best_embed_sim
        )

        # 5. 호모토피 클래스 할당
        query_point.homotopy_class_id = best_class.class_id

        # 6. 대안 Agent 목록 (Top-2~4)
        alternatives = []
        for hc, sim, _, _ in similarities[1:4]:
            alt_agent = self._select_target_agent(hc, query_embedding, query_tokens)
            alternatives.append(
                RoutingCandidate(
                    agent_id=alt_agent.agent_id,
                    agent_name=alt_agent.suggested_name or alt_agent.agent_id,
                    similarity=float(sim),
                    homotopy_class_id=hc.class_id,
                )
            )

        # 7. 모호성 판단
        is_ambiguous, ambiguity_reason = self._check_ambiguity(
            similarities, confidence
        )

        result = RoutingResult(
            query_text=query_text,
            target_agent_id=target_agent.agent_id,
            target_agent_name=target_agent.suggested_name or target_agent.agent_id,
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
            f"→ {target_agent.suggested_name or target_agent.agent_id} "
            f"(신뢰도: {confidence:.2f})"
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
        query_tokens = self._tokenize(query_text)
        return self._softmax_probabilities(
            query_embedding, query_tokens, temperature=temperature
        )

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
            sim = self._embedding_similarity(query_embedding, centroid)
            if sim > best_sim:
                best_sim = sim
                best_class = hc

        return best_class

    # ── 내부 계산 ─────────────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        similarities: list[tuple[HomotopyClass, float, float, float]],
        best_class: HomotopyClass,
        best_sim: float,
        best_embed_sim: float,
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
        alpha = self._embedding_reliability_weight()
        effective_sim = (
            alpha * np.clip(best_embed_sim, 0.0, 1.0)
            + (1.0 - alpha) * np.clip(best_sim, 0.0, 1.0)
        )
        radius = max(best_class.radius, 1e-6)
        distance = 1.0 - np.clip(effective_sim, 0.0, 1.0)
        radius_score = float(np.clip(1.0 - (distance / radius), 0.0, 1.0))

        confidence = 0.6 * margin_score + 0.4 * radius_score
        return float(np.clip(confidence, 0.0, 1.0))

    def _softmax_probabilities(
        self,
        query_embedding: np.ndarray,
        query_tokens: set[str],
        temperature: float = SOFTMAX_TEMPERATURE,
    ) -> dict[str, float]:
        # 클래스 확률을 계산 후, 클래스 내부 멤버에 재분배
        temp = max(1e-4, temperature)
        class_scores: list[tuple[HomotopyClass, float]] = []
        for hc in self._homotopy_classes:
            embed_sim = self._embedding_similarity(query_embedding, hc.centroid_array())
            lexical_sim = self._lexical_similarity(
                query_tokens, self._class_tokens.get(hc.class_id, set())
            )
            score = self._combine_similarity(embed_sim, lexical_sim)
            class_scores.append((hc, score))

        if not class_scores:
            return {}

        class_logits = np.array([score / temp for _, score in class_scores], dtype=float)
        class_logits -= class_logits.max()
        class_probs = np.exp(class_logits)
        class_sum = class_probs.sum()
        if class_sum <= 0:
            return {}
        class_probs = class_probs / class_sum

        agent_probs: dict[str, float] = {}
        for (hc, _), class_prob in zip(class_scores, class_probs):
            members = self._class_members.get(hc.class_id) or []
            if not members:
                agent_probs[hc.agent_id] = agent_probs.get(hc.agent_id, 0.0) + float(class_prob)
                continue
            if len(members) == 1:
                aid = members[0].agent_id
                agent_probs[aid] = agent_probs.get(aid, 0.0) + float(class_prob)
                continue

            member_scores = np.array(
                [
                    self._member_similarity(
                        member, query_embedding, query_tokens
                    )
                    for member in members
                ],
                dtype=float,
            )
            local_temp = max(1e-4, MEMBER_SOFTMAX_TEMPERATURE)
            local_logits = member_scores / local_temp
            local_logits -= local_logits.max()
            local_probs = np.exp(local_logits)
            local_sum = local_probs.sum()
            if local_sum <= 0:
                local_probs = np.ones(len(members), dtype=float) / len(members)
            else:
                local_probs = local_probs / local_sum
            for member, p in zip(members, local_probs):
                aid = member.agent_id
                agent_probs[aid] = agent_probs.get(aid, 0.0) + float(class_prob) * float(p)

        total = sum(agent_probs.values())
        if total <= 0:
            return {aid: 0.0 for aid in agent_probs}
        return {aid: float(prob / total) for aid, prob in agent_probs.items()}

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
        similarities: list[tuple[HomotopyClass, float, float, float]],
        confidence: float,
    ) -> tuple[bool, str]:
        """모호성 여부와 원인 반환"""
        top1_hybrid = similarities[0][1]
        top1_lexical = similarities[0][3]
        top2_hybrid = similarities[1][1] if len(similarities) >= 2 else 0.0
        alpha = self._embedding_reliability_weight()
        dynamic_min_conf = MIN_CONFIDENCE * alpha + 0.05 * (1.0 - alpha)

        if confidence < dynamic_min_conf:
            # lexical 증거가 강하면 보수적 모호성 판단을 완화
            if top1_lexical >= 0.45 and top1_hybrid >= 0.3:
                return False, ""
            # 임베딩 fallback 환경에서는 class 간 마진이 충분하면 수용
            if alpha <= 0.55 and (top1_hybrid - top2_hybrid) >= 0.08 and top1_hybrid >= 0.08:
                return False, ""
            return True, f"쿼리의 최대 유사도({similarities[0][1]:.2f})가 너무 낮습니다."

        if len(similarities) >= 2:
            top1_sim = similarities[0][1]
            top2_sim = similarities[1][1]
            threshold = AMBIGUITY_THRESHOLD * alpha + 0.03 * (1.0 - alpha)
            if (top1_sim - top2_sim) < threshold:
                if top1_lexical >= 0.45:
                    return False, ""
                return True, (
                    f"상위 두 Agent의 유사도 차이가 작습니다 "
                    f"({similarities[0][0].agent_name}: {top1_sim:.2f} vs "
                    f"{similarities[1][0].agent_name}: {top2_sim:.2f})"
                )

        return False, ""

    def _select_target_agent(
        self,
        homotopy_class: HomotopyClass,
        query_embedding: np.ndarray,
        query_tokens: set[str],
    ) -> DiscoveredAgent:
        members = self._class_members.get(homotopy_class.class_id) or []
        if not members:
            return DiscoveredAgent(
                agent_id=homotopy_class.agent_id,
                cluster_id=-1,
                task_ids=[],
                task_names=[],
                centroid=np.array(homotopy_class.centroid_embedding),
                suggested_name=homotopy_class.agent_name,
            )
        if len(members) == 1:
            return members[0]

        scored = [
            (
                member,
                self._member_similarity(member, query_embedding, query_tokens),
            )
            for member in members
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        if len(scored) >= 2:
            top_member, top_score = scored[0]
            second_score = scored[1][1]
            score_margin = top_score - second_score
            # 증거가 약하면 class 내 허브(주요 태스크 보유)로 수렴시켜 분산 오답을 줄임
            if top_score < 0.1 and score_margin < 0.03:
                hub_id = self._class_hub_agent_id.get(homotopy_class.class_id, "")
                if hub_id:
                    hub = next((m for m in members if m.agent_id == hub_id), None)
                    if hub is not None:
                        hub_score = next(
                            (score for m, score in scored if m.agent_id == hub_id),
                            top_score,
                        )
                        if hub_score + 0.02 >= top_score:
                            return hub
            return top_member
        return scored[0][0]

    def _member_similarity(
        self,
        member: DiscoveredAgent,
        query_embedding: np.ndarray,
        query_tokens: set[str],
    ) -> float:
        centroid = self._agent_centroids.get(member.agent_id)
        if centroid is None:
            name = member.suggested_name or member.agent_id
            role = member.suggested_role or "태스크 처리"
            centroid = self._embedder.embed_agent_profile(name, role, member.task_names)
            self._agent_centroids[member.agent_id] = centroid
        embed_sim = self._embedding_similarity(query_embedding, centroid)
        profile_lexical = self._lexical_similarity(
            query_tokens, self._agent_tokens.get(member.agent_id, set())
        )
        task_lexical = self._max_task_similarity(
            query_tokens,
            self._member_task_tokens.get(member.agent_id, []),
        )
        lexical_sim = float(np.clip(0.35 * profile_lexical + 0.65 * task_lexical, 0.0, 1.0))
        score = self._combine_similarity(embed_sim, lexical_sim)

        # fallback 환경에서는 member 크기 prior를 약하게 반영해 랜덤 분산을 줄임
        alpha = self._embedding_reliability_weight()
        if alpha <= 0.55:
            score = 0.88 * score + 0.12 * self._member_size_prior(member)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _embedding_similarity(a: np.ndarray, b: np.ndarray) -> float:
        # 코사인 유사도 음수는 정보가 거의 없으므로 0으로 절단
        raw = Embedder.cosine_similarity(a, b)
        return float(np.clip(raw, 0.0, 1.0))

    def _combine_similarity(self, embed_sim: float, lexical_sim: float) -> float:
        alpha = self._embedding_reliability_weight()
        score = alpha * embed_sim + (1.0 - alpha) * lexical_sim
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _lexical_similarity(a_tokens: set[str], b_tokens: set[str]) -> float:
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens & b_tokens)
        if inter <= 0:
            return 0.0
        union = len(a_tokens | b_tokens)
        if union <= 0:
            return 0.0
        # query coverage를 우선하여 large agent token-size 편향 완화
        recall = inter / max(1, len(a_tokens))
        jaccard = inter / union
        score = 0.8 * recall + 0.2 * jaccard
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        raw_tokens = {t.lower() for t in TOKEN_PATTERN.findall(text or "")}
        tokens = {HomotopyRouter._normalize_token(token) for token in raw_tokens}
        return {t for t in tokens if len(t) >= 2}

    @staticmethod
    def _normalize_token(token: str) -> str:
        if token in TOKEN_NORMALIZATION:
            return TOKEN_NORMALIZATION[token]
        suffixes = ("해주세요", "해줘", "하세요", "하기", "해", "줘")
        normalized = token
        for suffix in suffixes:
            if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
                normalized = normalized[: -len(suffix)]
                break
        return TOKEN_NORMALIZATION.get(normalized, normalized)

    def _max_task_similarity(
        self,
        query_tokens: set[str],
        member_task_tokens: list[set[str]],
    ) -> float:
        if not query_tokens or not member_task_tokens:
            return 0.0
        best = 0.0
        for task_tokens in member_task_tokens:
            score = self._lexical_similarity(query_tokens, task_tokens)
            if score > best:
                best = score
        return best

    def _member_size_prior(self, member: DiscoveredAgent) -> float:
        class_id = self._member_class_id.get(member.agent_id, "")
        if class_id:
            total = float(self._class_task_total.get(class_id, 1))
            ratio = len(member.task_ids) / total
            # prior는 약하게만 사용: [0.2, 1.0] 구간으로 압축
            return float(np.clip(0.2 + 0.8 * ratio, 0.0, 1.0))
        return 0.2

    def _embedding_reliability_weight(self) -> float:
        # 임베딩 fallback이 많을수록 lexical 비중을 올립니다.
        fallback_ratio = 0.0
        try:
            health = self._embedder.get_health()
            fallback_ratio = float(getattr(health, "fallback_ratio", 0.0))
        except Exception:
            return 0.8
        if fallback_ratio >= 0.8:
            return 0.35
        if fallback_ratio >= 0.4:
            return 0.55
        return 0.8

    @property
    def homotopy_classes(self) -> list[HomotopyClass]:
        return self._homotopy_classes

    @property
    def is_ready(self) -> bool:
        return self._ready
