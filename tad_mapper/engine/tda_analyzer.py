"""
TDA Analyzer - scikit-learn 기반 Mapper 알고리즘
AI 적용 기회들을 위상 공간에 매핑하고 클러스터를 자동 발견합니다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tad_mapper.engine.feature_extractor import TopologicalFeature

logger = logging.getLogger(__name__)


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
        데이터 기반 Agent 자동 발견.

        n_agents가 None이면 엘보우 방법으로 최적 클러스터 수 자동 결정.
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

        # 최적 클러스터 수 결정
        if n_agents is None:
            n_agents = self._find_optimal_k(matrix_scaled)

        n_agents = min(n_agents, len(features))
        logger.info(f"Agent 자동 발견: {n_agents}개 클러스터로 분류")

        kmeans = KMeans(n_clusters=n_agents, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix_scaled)

        agents: list[DiscoveredAgent] = []
        for k in range(n_agents):
            mask = labels == k
            cluster_features = [f for f, m in zip(features, mask) if m]
            centroid = matrix_scaled[mask].mean(axis=0) if mask.any() else np.zeros(matrix_scaled.shape[1])

            agents.append(DiscoveredAgent(
                agent_id=f"agent_{k}",
                cluster_id=k,
                task_ids=[f.task_id for f in cluster_features],
                task_names=[f.task_name for f in cluster_features],
                centroid=centroid,
            ))

        return agents

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
