"""
QueryManifold 단위 테스트 (임베딩 API 없이 Mock 사용)

테스트 항목:
1. build() 후 regions 생성
2. compute_coverage() 반환값 범위
3. find_nearest_region() 정확성
4. coverage_complete 조건
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from tad_mapper.engine.embedder import Embedder
from tad_mapper.engine.feature_extractor import TopologicalFeature
from tad_mapper.engine.query_manifold import QueryManifold
from tad_mapper.engine.tda_analyzer import DiscoveredAgent
from tad_mapper.models.topology import CoverageMetrics, CoverageRegion


# ── 픽스처 ───────────────────────────────────────────────────────────────────

def make_features(n: int = 6) -> list[TopologicalFeature]:
    """n개의 더미 6D 특징 벡터"""
    np.random.seed(42)
    feats = []
    for i in range(n):
        vec = np.clip(np.random.rand(6), 0.0, 1.0)
        feats.append(TopologicalFeature(
            task_id=f"t{i}",
            task_name=f"태스크 {i}",
            data_type=float(vec[0]),
            reasoning_depth=float(vec[1]),
            automation_potential=float(vec[2]),
            interaction_type=float(vec[3]),
            output_complexity=float(vec[4]),
            domain_specificity=float(vec[5]),
        ))
    return feats


def make_agents(n_agents: int = 3) -> list[DiscoveredAgent]:
    np.random.seed(42)
    agents = []
    tasks_per_agent = 2
    for i in range(n_agents):
        task_ids = [f"t{j}" for j in range(i * tasks_per_agent, (i + 1) * tasks_per_agent)]
        agents.append(DiscoveredAgent(
            agent_id=f"agent_{i}",
            cluster_id=i,
            task_ids=task_ids,
            task_names=[f"태스크 {j}" for j in range(i * tasks_per_agent, (i + 1) * tasks_per_agent)],
            centroid=np.random.rand(6),
            suggested_name=f"에이전트 {i}",
            suggested_role=f"역할 {i}",
        ))
    return agents


def make_mock_embedder() -> Embedder:
    """API 없이 결정적 임베딩 반환"""
    embedder = MagicMock(spec=Embedder)
    embedder.cosine_similarity = Embedder.cosine_similarity

    # embed_agent_profile: 단순 랜덤 벡터 (재현 가능)
    call_count = [0]
    def embed_agent_profile(name, role, tasks):
        np.random.seed(call_count[0])
        call_count[0] += 1
        return np.random.rand(6)

    embedder.embed_agent_profile.side_effect = embed_agent_profile
    return embedder


# ── 테스트 ────────────────────────────────────────────────────────────────────

class TestQueryManifoldBuild:
    def test_build_creates_regions(self):
        """build() 후 agents 수만큼 Coverage Region 생성"""
        manifold = QueryManifold()
        features = make_features(6)
        agents = make_agents(3)
        embedder = make_mock_embedder()

        # task_embeddings로 6D 특징 벡터 사용 (임베딩 API 불필요)
        task_embeddings = np.stack([f.vector for f in features])
        manifold.build(agents, features, embedder, task_embeddings)

        assert len(manifold.regions) == 3

    def test_build_without_embeddings(self):
        """task_embeddings=None이면 6D 특징 벡터로 대체"""
        manifold = QueryManifold()
        features = make_features(6)
        agents = make_agents(3)
        embedder = make_mock_embedder()

        manifold.build(agents, features, embedder, task_embeddings=None)
        # 6D 벡터로 폴백하면 region 수는 줄어들 수 있음
        assert len(manifold.regions) > 0

    def test_projected_2d_shape(self):
        """projected_2d shape가 [n_tasks, 2]"""
        manifold = QueryManifold()
        features = make_features(6)
        agents = make_agents(3)
        embedder = make_mock_embedder()
        task_embeddings = np.stack([f.vector for f in features])

        manifold.build(agents, features, embedder, task_embeddings)
        proj = manifold.projected_2d
        assert proj is not None
        assert proj.shape == (6, 2)


class TestCoverageMetrics:
    def setup_method(self):
        """각 테스트 전 Manifold 구축"""
        np.random.seed(0)
        self.manifold = QueryManifold()
        self.features = make_features(6)
        self.agents = make_agents(3)
        embedder = make_mock_embedder()
        task_embeddings = np.stack([f.vector for f in self.features])
        self.manifold.build(self.agents, self.features, embedder, task_embeddings)

    def test_coverage_ratio_range(self):
        """coverage_ratio ∈ [0, 1]"""
        metrics = self.manifold.compute_coverage()
        assert 0.0 <= metrics.coverage_ratio <= 1.0

    def test_overlap_ratio_range(self):
        """overlap_ratio ∈ [0, 1]"""
        metrics = self.manifold.compute_coverage()
        assert 0.0 <= metrics.overlap_ratio <= 1.0

    def test_gap_ratio_range(self):
        """gap_ratio ∈ [0, 1]"""
        metrics = self.manifold.compute_coverage()
        assert 0.0 <= metrics.gap_ratio <= 1.0

    def test_metrics_returns_coverage_metrics(self):
        """반환 타입이 CoverageMetrics"""
        metrics = self.manifold.compute_coverage()
        assert isinstance(metrics, CoverageMetrics)

    def test_agent_coverage_areas_keys(self):
        """agent_coverage_areas에 모든 에이전트 ID가 포함"""
        metrics = self.manifold.compute_coverage()
        for agent in self.agents:
            assert agent.agent_id in metrics.agent_coverage_areas

    def test_empty_manifold_returns_zero(self):
        """구축 전 compute_coverage() → coverage=0"""
        empty_manifold = QueryManifold()
        metrics = empty_manifold.compute_coverage()
        assert metrics.coverage_ratio == 0.0

    def test_uncovered_tasks_detected_when_region_radius_too_small(self):
        """region 반경이 매우 작으면 uncovered_task_ids가 계산되어야 함"""
        for region in self.manifold.regions:
            region.radius = 0.0

        metrics = self.manifold.compute_coverage()
        assert len(metrics.uncovered_task_ids) > 0
        assert metrics.coverage_complete is False
        assert metrics.gap_ratio > 0.0


class TestFindNearestRegion:
    def setup_method(self):
        np.random.seed(0)
        self.manifold = QueryManifold()
        features = make_features(6)
        agents = make_agents(3)
        embedder = make_mock_embedder()
        task_embeddings = np.stack([f.vector for f in features])
        self.manifold.build(agents, features, embedder, task_embeddings)

    def test_returns_coverage_region(self):
        """find_nearest_region() 반환값이 CoverageRegion"""
        query_emb = np.random.rand(6)
        region = self.manifold.find_nearest_region(query_emb)
        assert region is None or isinstance(region, CoverageRegion)

    def test_nearest_region_is_actually_nearest(self):
        """반환된 Region의 centroid가 쿼리와 가장 유사"""
        # region 0의 centroid를 쿼리로 사용 → region 0이 반환되어야 함
        if not self.manifold.regions:
            pytest.skip("regions 없음")

        region_0 = self.manifold.regions[0]
        centroid_0 = np.array(region_0.centroid_embedding)

        found = self.manifold.find_nearest_region(centroid_0)
        assert found is not None
        assert found.agent_id == region_0.agent_id

    def test_empty_manifold_returns_none(self):
        """빈 Manifold → None"""
        empty = QueryManifold()
        result = empty.find_nearest_region(np.zeros(6))
        assert result is None
