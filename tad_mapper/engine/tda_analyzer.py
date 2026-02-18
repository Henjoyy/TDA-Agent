"""
TDA Analyzer - HDBSCAN 기반 Mapper 알고리즘
AI 적용 기회들을 위상 공간에 매핑하고 클러스터를 자동 발견합니다.

클러스터링 전략:
- 1단계: HDBSCAN으로 자연스러운 클러스터 자동 발견 (k 불필요)
- 2단계: 노이즈 태스크 → K-Means로 가장 가까운 클러스터에 배정
- 3단계: refine_clusters()로 God Agent 방지 (MAX_TASKS_PER_AGENT 초과 시 분할)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import hdbscan
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

from tad_mapper.engine.feature_extractor import TopologicalFeature

logger = logging.getLogger(__name__)

# Agent당 최대 태스크 수 - 이 값을 초과하면 자동 분할
MAX_TASKS_PER_AGENT = 15


@dataclass
class MapperNode:
    """Mapper 그래프의 노드 (태스크 군집)"""
    node_id: int
    task_ids: list[str]
    task_names: list[str]
    centroid: np.ndarray          # 특징 공간에서의 중심점
    cover_interval: int           # 속한 커버 구간 인덱스


@dataclass
class MapperGraph:
    """Mapper 알고리즘 결과 그래프"""
    nodes: list[MapperNode]
    edges: list[tuple[int, int]]  # 공유 멤버가 있는 노드 쌍
    task_to_nodes: dict[str, list[int]]  # task_id → 속한 노드 ID 목록
    feature_matrix: np.ndarray
    projected_2d: np.ndarray      # PCA 2D 투영 (시각화용)


@dataclass
class DiscoveredAgent:
    """데이터 기반으로 자동 발견된 Agent 유형"""
    agent_id: str                 # "agent_0", "agent_1", ...
    cluster_id: int
    task_ids: list[str]
    task_names: list[str]
    centroid: np.ndarray
    # Gemini가 나중에 채워줄 필드
    suggested_name: str = ""
    suggested_role: str = ""
    suggested_capabilities: list[str] = field(default_factory=list)
    tool_prefix: str = ""


class TDAAnalyzer:
    """
    scikit-learn 기반 Mapper 알고리즘 구현.

    파이프라인:
    1. 특징 행렬 정규화
    2. PCA로 2D 필터 함수 생성
    3. 1D 커버(구간 분할 + 오버랩)
    4. 각 구간 내 DBSCAN 클러스터링
    5. 공유 멤버 기반 엣지 연결
    """

    def __init__(
        self,
        n_intervals: int = 10,
        overlap_frac: float = 0.3,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 2,
    ) -> None:
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

    def run_mapper(self, features: list[TopologicalFeature]) -> MapperGraph:
        """Mapper 알고리즘 실행"""
        if len(features) < 2:
            raise ValueError("Mapper 알고리즘에는 최소 2개 이상의 태스크가 필요합니다.")

        task_ids = [f.task_id for f in features]
        task_names = [f.task_name for f in features]
        matrix = np.stack([f.vector for f in features])

        # 1. 정규화
        scaler = StandardScaler()
        matrix_scaled = scaler.fit_transform(matrix)

        # 2. PCA 2D 투영 (필터 함수)
        n_components = min(2, matrix_scaled.shape[1], matrix_scaled.shape[0])
        pca = PCA(n_components=n_components)
        projected = pca.fit_transform(matrix_scaled)
        # 1D 필터값 (첫 번째 주성분)
        filter_values = projected[:, 0]

        # 3. 커버 생성 (오버랩 포함 구간 분할)
        f_min, f_max = filter_values.min(), filter_values.max()
        interval_len = (f_max - f_min) / self.n_intervals
        overlap_len = interval_len * self.overlap_frac

        nodes: list[MapperNode] = []
        node_id = 0
        task_to_nodes: dict[str, list[int]] = {tid: [] for tid in task_ids}

        for i in range(self.n_intervals):
            lo = f_min + i * interval_len - overlap_len
            hi = f_min + (i + 1) * interval_len + overlap_len

            # 구간 내 태스크 인덱스
            mask = (filter_values >= lo) & (filter_values <= hi)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            # 4. 구간 내 DBSCAN 클러스터링
            sub_matrix = matrix_scaled[indices]
            if len(indices) < self.dbscan_min_samples:
                # 샘플 수 부족 → 단일 노드
                clusters = np.zeros(len(indices), dtype=int)
            else:
                eps = max(self.dbscan_eps, 0.1)
                db = DBSCAN(eps=eps, min_samples=min(self.dbscan_min_samples, len(indices)))
                clusters = db.fit_predict(sub_matrix)

            # 클러스터별 노드 생성
            for cluster_label in set(clusters):
                if cluster_label == -1:
                    # 노이즈 포인트도 개별 노드로 처리
                    noise_indices = indices[clusters == -1]
                    for ni in noise_indices:
                        node = MapperNode(
                            node_id=node_id,
                            task_ids=[task_ids[ni]],
                            task_names=[task_names[ni]],
                            centroid=matrix_scaled[ni],
                            cover_interval=i,
                        )
                        nodes.append(node)
                        task_to_nodes[task_ids[ni]].append(node_id)
                        node_id += 1
                    continue

                cluster_mask = clusters == cluster_label
                cluster_indices = indices[cluster_mask]
                node = MapperNode(
                    node_id=node_id,
                    task_ids=[task_ids[j] for j in cluster_indices],
                    task_names=[task_names[j] for j in cluster_indices],
                    centroid=matrix_scaled[cluster_indices].mean(axis=0),
                    cover_interval=i,
                )
                nodes.append(node)
                for j in cluster_indices:
                    task_to_nodes[task_ids[j]].append(node_id)
                node_id += 1

        # 5. 엣지 생성 (공유 태스크가 있는 노드 쌍)
        edges: list[tuple[int, int]] = []
        for tid, nids in task_to_nodes.items():
            for a in range(len(nids)):
                for b in range(a + 1, len(nids)):
                    edge = (min(nids[a], nids[b]), max(nids[a], nids[b]))
                    if edge not in edges:
                        edges.append(edge)

        logger.info(f"Mapper 완료: {len(nodes)}개 노드, {len(edges)}개 엣지")

        return MapperGraph(
            nodes=nodes,
            edges=edges,
            task_to_nodes=task_to_nodes,
            feature_matrix=matrix,
            projected_2d=projected if projected.shape[1] >= 2
                         else np.column_stack([projected, np.zeros(len(projected))]),
        )

    def discover_agents(
        self,
        features: list[TopologicalFeature],
        n_agents: int | None = None,
    ) -> list[DiscoveredAgent]:
        """
        HDBSCAN 기반 Agent 자동 발견.

        K-Means 대비 장점:
        - k를 미리 지정할 필요 없음 (자동 결정)
        - 비구형(non-spherical) 클러스터 지원
        - 노이즈 태스크 자동 분리 후 K-Means로 재배정
        - 의미적으로 유사한 태스크들이 자연스럽게 묶임

        n_agents가 지정되면 HDBSCAN 결과를 무시하고 KMeans로 강제 분할.
        """
        if len(features) < 2:
            return [DiscoveredAgent(
                agent_id="agent_0",
                cluster_id=0,
                task_ids=[f.task_id for f in features],
                task_names=[f.task_name for f in features],
                centroid=features[0].vector if features else np.zeros(6),
            )]

        matrix = np.stack([f.vector for f in features])
        scaler = StandardScaler()
        matrix_scaled = scaler.fit_transform(matrix)

        # n_agents가 명시적으로 지정된 경우 → KMeans 강제 사용
        if n_agents is not None:
            n_agents = min(n_agents, len(features))
            logger.info(f"Agent 자동 발견: KMeans {n_agents}개 클러스터 (사용자 지정)")
            kmeans = KMeans(n_clusters=n_agents, random_state=42, n_init=10)
            labels = kmeans.fit_predict(matrix_scaled)
            return self._build_agents_from_labels(features, matrix_scaled, labels)

        # ── HDBSCAN 자동 클러스터링 ──────────────────────────────────────────
        # min_cluster_size: 에이전트가 되려면 최소 몇 개 태스크가 필요한지
        # 태스크 수에 따라 동적으로 조정 (소규모: 2, 대규모: 3~5)
        n = len(features)
        min_cluster_size = max(2, min(5, n // 6))
        min_samples = max(1, min_cluster_size // 2)

        logger.info(
            f"HDBSCAN 클러스터링 시작: {n}개 태스크, "
            f"min_cluster_size={min_cluster_size}, min_samples={min_samples}"
        )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",  # Excess of Mass: 안정적인 클러스터 선택
            prediction_data=True,
        )
        labels = clusterer.fit_predict(matrix_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        logger.info(
            f"HDBSCAN 결과: {n_clusters}개 클러스터, {n_noise}개 노이즈 태스크"
        )

        # 클러스터가 1개 이하면 KMeans 폴백
        if n_clusters <= 1:
            fallback_k = self._find_optimal_k(matrix_scaled)
            logger.warning(
                f"HDBSCAN 클러스터 수 부족 ({n_clusters}개). "
                f"KMeans 폴백 (k={fallback_k})"
            )
            kmeans = KMeans(n_clusters=fallback_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(matrix_scaled)
            return self._build_agents_from_labels(features, matrix_scaled, labels)

        # ── 노이즈 태스크 재배정 ─────────────────────────────────────────────
        # HDBSCAN이 -1(노이즈)로 분류한 태스크를 가장 가까운 클러스터에 배정
        if n_noise > 0:
            labels = self._assign_noise_to_nearest(
                matrix_scaled, labels, n_clusters
            )
            logger.info(f"  → 노이즈 {n_noise}개 태스크를 인접 클러스터에 재배정 완료")

        return self._build_agents_from_labels(features, matrix_scaled, labels)

    @staticmethod
    def _assign_noise_to_nearest(
        matrix: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
    ) -> np.ndarray:
        """
        HDBSCAN 노이즈 포인트(-1)를 가장 가까운 클러스터 centroid에 배정합니다.
        """
        labels = labels.copy()
        noise_mask = labels == -1
        if not noise_mask.any():
            return labels

        # 각 클러스터의 centroid 계산
        centroids = np.array([
            matrix[labels == k].mean(axis=0)
            for k in range(n_clusters)
        ])

        # 노이즈 포인트 → 가장 가까운 centroid 배정
        noise_points = matrix[noise_mask]
        nearest_clusters, _ = pairwise_distances_argmin_min(noise_points, centroids)
        labels[noise_mask] = nearest_clusters
        return labels

    @staticmethod
    def _build_agents_from_labels(
        features: list[TopologicalFeature],
        matrix_scaled: np.ndarray,
        labels: np.ndarray,
    ) -> list[DiscoveredAgent]:
        """레이블 배열로부터 DiscoveredAgent 목록을 생성합니다."""
        unique_labels = sorted(set(labels))
        agents: list[DiscoveredAgent] = []
        for idx, k in enumerate(unique_labels):
            mask = labels == k
            cluster_features = [f for f, m in zip(features, mask) if m]
            centroid = matrix_scaled[mask].mean(axis=0)
            agents.append(DiscoveredAgent(
                agent_id=f"agent_{idx}",
                cluster_id=idx,
                task_ids=[f.task_id for f in cluster_features],
                task_names=[f.task_name for f in cluster_features],
                centroid=centroid,
            ))
        logger.info(
            f"Agent 구성 완료: {len(agents)}개 "
            f"(태스크 분포: {[len(a.task_ids) for a in agents]})"
        )
        return agents

    def refine_clusters(
        self,
        agents: list[DiscoveredAgent],
        features: list[TopologicalFeature],
        max_tasks: int = MAX_TASKS_PER_AGENT,
    ) -> list[DiscoveredAgent]:
        """
        God Agent 방지: 태스크가 너무 많은 Agent를 자동 분할합니다.

        max_tasks를 초과하는 Agent에 대해 재귀적으로 K-Means 서브클러스터링을
        적용하여 균형 잡힌 에이전트 분포를 만듭니다.

        Args:
            agents: 발견된 Agent 목록
            features: 원본 태스크 특징 벡터 목록
            max_tasks: Agent당 최대 허용 태스크 수

        Returns:
            균형 잡힌 Agent 목록 (분할된 Agent 포함)
        """
        feature_map = {f.task_id: f for f in features}
        refined: list[DiscoveredAgent] = []

        for agent in agents:
            if len(agent.task_ids) <= max_tasks:
                refined.append(agent)
            else:
                logger.warning(
                    f"God Agent 탐지: '{agent.suggested_name or agent.agent_id}' "
                    f"({len(agent.task_ids)}개 태스크 > 최대 {max_tasks}개). 분할 시작."
                )
                split = self._split_large_agent(agent, feature_map, max_tasks)
                refined.extend(split)
                logger.info(
                    f"  → {len(split)}개 서브 에이전트로 분할 완료: "
                    f"{[a.agent_id for a in split]}"
                )

        if len(refined) != len(agents):
            logger.info(
                f"클러스터 정제 완료: {len(agents)}개 → {len(refined)}개 Agent "
                f"(태스크 상한: {max_tasks}개/Agent)"
            )

        return refined

    def _split_large_agent(
        self,
        agent: DiscoveredAgent,
        feature_map: dict[str, "TopologicalFeature"],
        max_tasks: int,
    ) -> list[DiscoveredAgent]:
        """
        단일 Agent를 재귀적으로 분할합니다.
        분할 수 = ceil(task_count / max_tasks)
        """
        agent_features = [
            feature_map[tid]
            for tid in agent.task_ids
            if tid in feature_map
        ]

        if len(agent_features) < 2:
            return [agent]

        # 필요한 서브클러스터 수 계산
        import math
        n_split = min(
            math.ceil(len(agent.task_ids) / max_tasks),
            len(agent_features),
        )
        n_split = max(n_split, 2)  # 최소 2분할

        matrix = np.stack([f.vector for f in agent_features])
        from sklearn.preprocessing import StandardScaler
        matrix_scaled = StandardScaler().fit_transform(matrix)

        kmeans = KMeans(n_clusters=n_split, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix_scaled)

        split_agents: list[DiscoveredAgent] = []
        suffix_labels = ["A", "B", "C", "D", "E", "F"]
        base_name = agent.suggested_name or agent.agent_id
        base_id = agent.agent_id

        for k in range(n_split):
            mask = labels == k
            sub_features = [f for f, m in zip(agent_features, mask) if m]
            if not sub_features:
                continue

            suffix = suffix_labels[k] if k < len(suffix_labels) else str(k)
            sub_centroid = matrix_scaled[mask].mean(axis=0)

            sub_agent = DiscoveredAgent(
                agent_id=f"{base_id}_sub{k}",
                cluster_id=agent.cluster_id * 10 + k,
                task_ids=[f.task_id for f in sub_features],
                task_names=[f.task_name for f in sub_features],
                centroid=sub_centroid,
                # 이름은 AgentNamer가 나중에 재명명함
                # 임시 이름은 기존 이름 + 접미사
                suggested_name=f"{base_name} {suffix}",
                suggested_role=agent.suggested_role,
                suggested_capabilities=list(agent.suggested_capabilities),
            )
            split_agents.append(sub_agent)

        return split_agents if split_agents else [agent]

    @staticmethod
    def _find_optimal_k(matrix: np.ndarray, max_k: int = 8) -> int:
        """엘보우 방법으로 최적 클러스터 수 결정"""
        n = len(matrix)
        max_k = min(max_k, n - 1, 8)
        if max_k < 2:
            return 1

        inertias: list[float] = []
        k_range = range(2, max_k + 1)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(matrix)
            inertias.append(km.inertia_)

        # 엘보우 포인트: 기울기 변화가 가장 큰 지점
        if len(inertias) < 2:
            return 2

        diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        elbow_idx = int(np.argmax(diffs))
        optimal_k = list(k_range)[elbow_idx]
        logger.info(f"엘보우 방법 최적 k={optimal_k} (범위: 2~{max_k})")
        return optimal_k
