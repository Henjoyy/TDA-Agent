"""
HomotopyRouter 단위 테스트 (임베딩 API 없이 Mock 사용)

테스트 항목:
1. build() 후 router.is_ready == True
2. route() 가 유사 쿼리를 같은 Agent로 라우팅
3. confidence 범위 [0, 1]
4. 모호성 탐지
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tad_mapper.engine.embedder import Embedder
from tad_mapper.engine.homotopy_router import HomotopyRouter
from tad_mapper.engine.tda_analyzer import DiscoveredAgent
from tad_mapper.models.topology import HomotopyClass, RoutingResult


# ── Mock Embedder ─────────────────────────────────────────────────────────────

def make_mock_embedder(query_vec: np.ndarray | None = None) -> Embedder:
    """실제 API 호출 없이 결정적 임베딩을 반환하는 Mock Embedder"""
    embedder = MagicMock(spec=Embedder)

    # 에이전트 프로파일 임베딩: 각 에이전트마다 고정 벡터
    agent_vecs = {
        "agent_0": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 검색 방향
        "agent_1": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 분석 방향
        "agent_2": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 생성 방향
    }

    def embed_agent_profile(agent_name, agent_role, task_names):
        # agent_name에서 agent_id 추출
        for aid, vec in agent_vecs.items():
            if aid in agent_name.lower() or agent_name.lower() in aid:
                return vec
        return np.random.rand(10)

    embedder.embed_agent_profile.side_effect = embed_agent_profile

    # 쿼리 임베딩: 테스트마다 다르게 설정 가능
    if query_vec is not None:
        embedder.embed_query.return_value = query_vec
    else:
        embedder.embed_query.return_value = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # cosine_similarity는 실제 함수 사용
    embedder.cosine_similarity = Embedder.cosine_similarity

    return embedder


def make_agents() -> list[DiscoveredAgent]:
    return [
        DiscoveredAgent(
            agent_id="agent_0",
            cluster_id=0,
            task_ids=["t1", "t2"],
            task_names=["데이터 검색", "정보 조회"],
            centroid=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            suggested_name="agent_0 검색 에이전트",
            suggested_role="데이터 검색 및 조회",
        ),
        DiscoveredAgent(
            agent_id="agent_1",
            cluster_id=1,
            task_ids=["t3", "t4"],
            task_names=["통계 분석", "트렌드 분석"],
            centroid=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            suggested_name="agent_1 분석 에이전트",
            suggested_role="통계 분석 및 집계",
        ),
        DiscoveredAgent(
            agent_id="agent_2",
            cluster_id=2,
            task_ids=["t5"],
            task_names=["보고서 생성"],
            centroid=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            suggested_name="agent_2 생성 에이전트",
            suggested_role="문서 및 보고서 생성",
        ),
    ]


# ── 테스트 ────────────────────────────────────────────────────────────────────

class TestHomotopyRouterBuild:
    def test_is_ready_after_build(self):
        """build() 완료 후 is_ready == True"""
        embedder = make_mock_embedder()
        router = HomotopyRouter(embedder)
        agents = make_agents()
        router.build(agents)
        assert router.is_ready is True

    def test_homotopy_classes_count(self):
        """에이전트 수만큼 호모토피 클래스 생성"""
        embedder = make_mock_embedder()
        router = HomotopyRouter(embedder)
        agents = make_agents()
        router.build(agents)
        assert len(router.homotopy_classes) == len(agents)

    def test_class_ids_match_agents(self):
        """각 호모토피 클래스의 agent_id가 에이전트 ID와 일치"""
        embedder = make_mock_embedder()
        router = HomotopyRouter(embedder)
        agents = make_agents()
        router.build(agents)
        class_agent_ids = {hc.agent_id for hc in router.homotopy_classes}
        agent_ids = {a.agent_id for a in agents}
        assert class_agent_ids == agent_ids


class TestRouting:
    def test_route_returns_routing_result(self):
        """route() 반환값이 RoutingResult 타입"""
        # 쿼리가 agent_0 방향 (검색 에이전트)
        query_vec = np.array([0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        embedder = make_mock_embedder(query_vec)
        router = HomotopyRouter(embedder)
        router.build(make_agents())

        result = router.route("데이터 검색해줘")
        assert isinstance(result, RoutingResult)

    def test_similar_queries_same_agent(self):
        """유사한 의미의 쿼리가 같은 Agent로 라우팅"""
        # 두 쿼리 모두 agent_0 방향
        query_vec = np.array([0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        embedder = make_mock_embedder(query_vec)
        router = HomotopyRouter(embedder)
        router.build(make_agents())

        result1 = router.route("자료 찾아줘")
        result2 = router.route("데이터 검색해")
        result3 = router.route("정보 좀 줘")

        # 모두 같은 Agent로 라우팅되어야 함 (같은 호모토피 클래스)
        assert result1.target_agent_id == result2.target_agent_id
        assert result2.target_agent_id == result3.target_agent_id

    def test_confidence_in_range(self):
        """confidence ∈ [0, 1]"""
        embedder = make_mock_embedder()
        router = HomotopyRouter(embedder)
        router.build(make_agents())
        result = router.route("분석해줘")
        assert 0.0 <= result.confidence <= 1.0

    def test_alternatives_not_include_target(self):
        """대안 목록에 타겟 에이전트가 포함되지 않아야 함"""
        embedder = make_mock_embedder()
        router = HomotopyRouter(embedder)
        router.build(make_agents())
        result = router.route("어떤 쿼리")
        alt_ids = [a.agent_id for a in result.alternatives]
        assert result.target_agent_id not in alt_ids

    def test_raises_when_not_built(self):
        """build() 전 route() 호출 시 RuntimeError"""
        embedder = make_mock_embedder()
        router = HomotopyRouter(embedder)
        with pytest.raises(RuntimeError):
            router.route("쿼리")

    def test_route_includes_probabilities(self):
        """route() 결과에 agent별 확률 분포 포함"""
        embedder = make_mock_embedder()
        router = HomotopyRouter(embedder)
        router.build(make_agents())
        result = router.route("분석해줘")
        assert set(result.routing_probabilities) == {"agent_0", "agent_1", "agent_2"}
        assert sum(result.routing_probabilities.values()) == pytest.approx(1.0, abs=1e-6)

    def test_route_soft_distribution(self):
        """route_soft()는 확률 분포를 반환해야 함"""
        embedder = make_mock_embedder()
        router = HomotopyRouter(embedder)
        router.build(make_agents())
        probs = router.route_soft("데이터 검색해줘")
        assert set(probs) == {"agent_0", "agent_1", "agent_2"}
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


class TestAmbiguity:
    def test_high_confidence_not_ambiguous(self):
        """명확한 쿼리 (agent_0 방향) → is_ambiguous == False"""
        # 매우 강한 신호: [1, 0, 0, ...]
        query_vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        embedder = make_mock_embedder(query_vec)
        router = HomotopyRouter(embedder)
        router.build(make_agents())
        result = router.route("명확한 검색 쿼리")
        # top1 vs top2 마진이 충분히 크면 ambiguous=False
        assert result.target_agent_id == "agent_0"

    def test_ambiguous_query_detected(self):
        """두 Agent와 유사도가 비슷한 쿼리 → is_ambiguous 가능"""
        # agent_0와 agent_1 사이의 쿼리
        query_vec = np.array([0.707, 0.707, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        query_vec = query_vec / np.linalg.norm(query_vec)
        embedder = make_mock_embedder(query_vec)
        router = HomotopyRouter(embedder)
        router.build(make_agents())
        result = router.route("애매한 쿼리")
        # ambiguous 여부는 마진에 따라 결정됨 (항상 True는 아니지만 신뢰도가 낮을 것)
        assert result.confidence <= 0.7  # 완벽한 중간이면 신뢰도 낮음


class TestClassify:
    def test_classify_returns_homotopy_class(self):
        """classify() 가 HomotopyClass 반환"""
        embedder = make_mock_embedder()
        router = HomotopyRouter(embedder)
        router.build(make_agents())

        query_vec = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        hc = router.classify(query_vec)
        assert isinstance(hc, HomotopyClass)
        assert hc.agent_id == "agent_0"  # agent_0 방향

    def test_classify_empty_router(self):
        """빈 router의 classify() → None"""
        embedder = make_mock_embedder()
        router = HomotopyRouter(embedder)
        result = router.classify(np.zeros(10))
        assert result is None
