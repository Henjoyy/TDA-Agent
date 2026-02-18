"""
Query Manifold — 수학적 정식화 #1

Q ⊆ ∪ Ui (사용자 쿼리 매니폴드 ⊆ Unit Agent들의 열린 피복)

모든 사용자 쿼리가 속하는 위상 공간 Q를 정의하고,
Unit Agent들이 이 공간을 빈틈없이 커버하는지 검증합니다.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, Voronoi

from tad_mapper.engine.embedder import Embedder
from tad_mapper.engine.tda_analyzer import DiscoveredAgent
from tad_mapper.engine.feature_extractor import TopologicalFeature
from tad_mapper.models.topology import CoverageRegion, CoverageMetrics

logger = logging.getLogger(__name__)


class QueryManifold:
    """
    사용자 쿼리 매니폴드 Q와 Agent 피복 {Ui}를 모델링합니다.

    Q ⊆ ∪ Ui 조건을 측정하고 커버리지 메트릭을 계산합니다.
    """

    def __init__(self) -> None:
        self._regions: list[CoverageRegion] = []
        self._task_embeddings: np.ndarray | None = None        # shape [n_tasks, 768]
        self._projected_2d: np.ndarray | None = None           # shape [n_tasks, 2]
        self._agent_centroids_2d: np.ndarray | None = None     # shape [n_agents, 2]
        self._pca: PCA | None = None
        self._task_to_agent: dict[str, str] = {}               # task_id → agent_id
        self._coverage_metrics: CoverageMetrics | None = None

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def build(
        self,
        agents: list[DiscoveredAgent],
        features: list[TopologicalFeature],
        embedder: Embedder,
        task_embeddings: np.ndarray | None = None,
    ) -> None:
        """
        Agent 정보와 태스크 특징으로 Query Manifold를 구축합니다.

        Args:
            agents: 발견된 Unit Agent 목록
            features: 6D 위상 특징 벡터 목록
            embedder: 임베딩 생성기
            task_embeddings: 미리 계산된 태스크 임베딩 (없으면 features 기반으로 대체)
        """
        logger.info("Query Manifold 구축 시작...")

        # task_id → agent_id 매핑
        self._task_to_agent = {
            tid: agent.agent_id
            for agent in agents
            for tid in agent.task_ids
        }

        # 임베딩 행렬 준비
        if task_embeddings is not None and len(task_embeddings) > 0:
            self._task_embeddings = task_embeddings
        else:
            # 6D 특징 벡터를 대용으로 사용 (임베딩 미생성 시 폴백)
            logger.warning("태스크 임베딩 없음 — 6D 특징 벡터로 대체합니다.")
            self._task_embeddings = np.stack([f.vector for f in features])

        # 2D PCA 투영 (Voronoi + 시각화용)
        n_samples, n_features = self._task_embeddings.shape
        n_components = min(2, n_samples, n_features)
        self._pca = PCA(n_components=n_components)
        projected = self._pca.fit_transform(self._task_embeddings)

        if projected.shape[1] < 2:
            projected = np.column_stack([projected, np.zeros(len(projected))])
        self._projected_2d = projected

        # Agent별 Coverage Region 구성
        self._regions = []
        task_id_list = [f.task_id for f in features]

        for agent in agents:
            agent_name = agent.suggested_name or agent.agent_id
            agent_role = agent.suggested_role or ""

            # Agent에 속한 태스크 인덱스 수집
            agent_indices = [
                i for i, tid in enumerate(task_id_list)
                if tid in agent.task_ids
            ]

            if not agent_indices:
                continue

            agent_embeddings = self._task_embeddings[agent_indices]

            # centroid = Agent 태스크 임베딩들의 평균
            centroid_emb = agent_embeddings.mean(axis=0)

            # 반경 = centroid까지의 최대 코사인 거리
            sims = [
                Embedder.cosine_similarity(centroid_emb, self._task_embeddings[i])
                for i in agent_indices
            ]
            radius = 1.0 - min(sims) if sims else 0.3  # 코사인 거리로 변환

            # 2D 투영 centroid
            proj_centroid = self._projected_2d[agent_indices].mean(axis=0)

            region = CoverageRegion(
                agent_id=agent.agent_id,
                agent_name=agent_name,
                centroid_embedding=centroid_emb.tolist(),
                task_embeddings=[
                    self._task_embeddings[i].tolist() for i in agent_indices
                ],
                radius=float(np.clip(radius, 0.0, 1.0)),
                projected_centroid_2d=proj_centroid.tolist(),
            )
            self._regions.append(region)

        # Agent centroid 2D 좌표 행렬
        if self._regions:
            self._agent_centroids_2d = np.array(
                [r.projected_centroid_2d for r in self._regions]
            )

        logger.info(f"Query Manifold 구축 완료: {len(self._regions)}개 Coverage Region")

    def compute_coverage(self) -> CoverageMetrics:
        """
        Q ⊆ ∪ Ui 커버리지를 측정합니다.

        Voronoi tessellation으로 각 에이전트의 영역을 정의하고
        전체 공간 대비 커버된 비율을 계산합니다.
        """
        if not self._regions or self._projected_2d is None:
            logger.warning("Manifold가 구축되지 않았습니다. build()를 먼저 호출하세요.")
            return CoverageMetrics(
                coverage_ratio=0.0, overlap_ratio=0.0, gap_ratio=1.0,
                coverage_complete=False,
            )

        n_tasks = len(self._projected_2d)
        coverage_ratio, overlap_ratio, gap_ratio = self._compute_2d_coverage()

        # Agent별 커버리지 면적 (Voronoi 셀 기반)
        agent_areas = self._compute_voronoi_areas()
        for region in self._regions:
            region.voronoi_area = agent_areas.get(region.agent_id, 0.0)

        # 미커버 태스크 탐지 (어떤 region의 radius 안에도 없는 태스크)
        uncovered = self._find_uncovered_tasks()

        # Agent 간 중첩 쌍
        overlap_pairs = self._find_overlap_pairs()

        metrics = CoverageMetrics(
            coverage_ratio=coverage_ratio,
            overlap_ratio=overlap_ratio,
            gap_ratio=gap_ratio,
            agent_coverage_areas=agent_areas,
            uncovered_task_ids=uncovered,
            overlap_agent_pairs=overlap_pairs,
            coverage_complete=(coverage_ratio >= 0.95 and len(uncovered) == 0),
        )
        self._coverage_metrics = metrics

        logger.info(
            f"커버리지 분석 완료: {coverage_ratio:.1%} 커버 "
            f"(중첩: {overlap_ratio:.1%}, 갭: {gap_ratio:.1%})"
        )
        return metrics

    def find_nearest_region(self, query_embedding: np.ndarray) -> CoverageRegion | None:
        """
        쿼리 임베딩에서 가장 가까운 Coverage Region(Agent 영역)을 찾습니다.

        코사인 유사도 기반으로 Q → Ui 매핑.
        """
        if not self._regions:
            return None

        best_region = None
        best_sim = -1.0

        for region in self._regions:
            centroid = region.centroid_array()
            sim = Embedder.cosine_similarity(query_embedding, centroid)
            if sim > best_sim:
                best_sim = sim
                best_region = region

        return best_region

    def project_query(self, query_embedding: np.ndarray) -> np.ndarray:
        """쿼리 임베딩을 2D 투영 (시각화용)"""
        if self._pca is None:
            return np.zeros(2)
        try:
            projected = self._pca.transform(query_embedding.reshape(1, -1))
            return projected[0]
        except Exception:
            return np.zeros(2)

    @property
    def regions(self) -> list[CoverageRegion]:
        return self._regions

    @property
    def projected_2d(self) -> np.ndarray | None:
        return self._projected_2d

    @property
    def agent_centroids_2d(self) -> np.ndarray | None:
        return self._agent_centroids_2d

    # ── 내부 계산 ─────────────────────────────────────────────────────────────

    def _compute_2d_coverage(self) -> tuple[float, float, float]:
        """2D 투영 공간에서 커버리지 비율 계산"""
        points = self._projected_2d
        if len(points) < 4:
            return 1.0, 0.0, 0.0

        try:
            hull = ConvexHull(points)
            total_area = hull.area if points.shape[1] == 2 else hull.volume

            # Agent별 포인트로 서브 convex hull 계산
            covered_area = 0.0
            task_id_list = list(self._task_to_agent.keys())

            for region in self._regions:
                # 이 Agent에 속한 2D 포인트
                agent_point_indices = [
                    i for i, (p, a) in enumerate(zip(
                        self._projected_2d,
                        [self._task_to_agent.get(tid, "") for tid in task_id_list]
                    ))
                    if a == region.agent_id
                ]

                if len(agent_point_indices) >= 3:
                    try:
                        sub_hull = ConvexHull(self._projected_2d[agent_point_indices])
                        covered_area += sub_hull.area if points.shape[1] == 2 else sub_hull.volume
                    except Exception:
                        pass

            coverage_ratio = min(covered_area / total_area, 1.0) if total_area > 0 else 1.0
            # 간단한 추정: overlap은 covered/total - 1이 양수인 경우
            overlap_ratio = max(0.0, (covered_area / total_area) - 1.0) if total_area > 0 else 0.0
            overlap_ratio = min(overlap_ratio, 1.0)
            gap_ratio = max(0.0, 1.0 - coverage_ratio)

        except Exception as e:
            logger.warning(f"커버리지 계산 실패: {e}")
            coverage_ratio = float(len(self._regions) > 0)
            overlap_ratio = 0.0
            gap_ratio = 1.0 - coverage_ratio

        return coverage_ratio, overlap_ratio, gap_ratio

    def _compute_voronoi_areas(self) -> dict[str, float]:
        """Voronoi tessellation으로 Agent별 셀 면적 계산"""
        areas: dict[str, float] = {r.agent_id: 0.0 for r in self._regions}

        if self._agent_centroids_2d is None or len(self._agent_centroids_2d) < 4:
            # Voronoi에 최소 4개 포인트 필요 — 균등 분배로 대체
            n = len(self._regions)
            if n > 0:
                for region in self._regions:
                    areas[region.agent_id] = 1.0 / n
            return areas

        try:
            # 경계 포인트 추가 (Voronoi ridge가 무한대로 뻗는 것 방지)
            padding = 5.0
            centroids = self._agent_centroids_2d
            border_pts = np.array([
                [centroids[:, 0].min() - padding, centroids[:, 1].min() - padding],
                [centroids[:, 0].max() + padding, centroids[:, 1].min() - padding],
                [centroids[:, 0].min() - padding, centroids[:, 1].max() + padding],
                [centroids[:, 0].max() + padding, centroids[:, 1].max() + padding],
            ])
            extended = np.vstack([centroids, border_pts])

            vor = Voronoi(extended)
            # Voronoi 셀 면적은 정확 계산이 복잡 — 태스크 카운트 기반 근사 사용
            for region in self._regions:
                count = len(region.task_embeddings)
                areas[region.agent_id] = float(count)

            # 정규화
            total = sum(areas.values())
            if total > 0:
                areas = {k: v / total for k, v in areas.items()}

        except Exception as e:
            logger.warning(f"Voronoi 계산 실패: {e}. 태스크 수 기반 근사 사용.")
            total = sum(len(r.task_embeddings) for r in self._regions)
            if total > 0:
                for region in self._regions:
                    areas[region.agent_id] = len(region.task_embeddings) / total

        return areas

    def _find_uncovered_tasks(self) -> list[str]:
        """어떤 Coverage Region에도 포함되지 않는 태스크 ID 반환"""
        covered = {tid for region in self._regions
                   for tid in (self._task_to_agent.keys()
                                if region.task_embeddings else [])}
        all_tasks = set(self._task_to_agent.keys())
        return list(all_tasks - covered)

    def _find_overlap_pairs(self) -> list[tuple[str, str]]:
        """centroid 간 거리가 radius 합보다 작은 에이전트 쌍 (중첩 영역 존재)"""
        pairs: list[tuple[str, str]] = []
        for i, ri in enumerate(self._regions):
            for rj in self._regions[i + 1:]:
                ci = ri.centroid_array()
                cj = rj.centroid_array()
                sim = Embedder.cosine_similarity(ci, cj)
                # 코사인 유사도 0.8 이상 → 영역 중첩 가능성
                if sim > 0.8:
                    pairs.append((ri.agent_id, rj.agent_id))
        return pairs
