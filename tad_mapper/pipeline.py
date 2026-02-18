"""
TAD-Mapper 메인 파이프라인 오케스트레이터

수학적 정식화 고도화:
- Step 9:  태스크 임베딩 생성 (Embedder 활성화)
- Step 10: Query Manifold 구축 + 커버리지 분석 (Q ⊆ ∪Ui)
- Step 11: Homotopy Router 초기화 (Φ: Q → {Uk})
- Step 7.5: MCP Tool 균형 분석 + Agile 재분배

런타임 기능:
- route_query(query)     : 실시간 쿼리 → Agent 라우팅
- compose_tools(q, aid)  : 쿼리 + Agent → Tool 합성 계획
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config.settings import TDA_N_INTERVALS, TDA_OVERLAP_FRAC, OUTPUT_DIR, validate_config
from tad_mapper.engine.embedder import Embedder
from tad_mapper.engine.feature_extractor import FeatureExtractor, TopologicalFeature
from tad_mapper.engine.homotopy_router import HomotopyRouter
from tad_mapper.engine.query_manifold import QueryManifold
from tad_mapper.engine.tda_analyzer import DiscoveredAgent, MapperGraph, TDAAnalyzer
from tad_mapper.engine.tool_balancer import ToolBalancer
from tad_mapper.engine.tool_composer import ToolComposer
from tad_mapper.engine.visualizer import TDAVisualizer
from tad_mapper.input.parser import JourneyParser
from tad_mapper.mapper.agent_namer import AgentNamer
from tad_mapper.mapper.hole_detector import HoleDetector
from tad_mapper.models.agent import HoleWarning, OverlapWarning
from tad_mapper.models.journey import UserJourney
from tad_mapper.models.mcp_tool import MCPToolSchema
from tad_mapper.models.topology import (
    BalanceReport,
    CompositionPlan,
    CoverageMetrics,
    RoutingResult,
)
from tad_mapper.output.mcp_generator import MCPGenerator
from tad_mapper.output.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """파이프라인 전체 실행 결과"""
    journey: UserJourney
    features: list[TopologicalFeature]
    mapper_graph: MapperGraph
    agents: list[DiscoveredAgent]
    mcp_tools: list[MCPToolSchema]
    holes: list[HoleWarning]
    overlaps: list[OverlapWarning]
    report_markdown: str
    report_json: dict
    # ── 수학적 정식화 추가 필드 ──────────────────────────────────
    task_embeddings: object = None               # np.ndarray | None
    coverage_metrics: CoverageMetrics | None = None
    balance_report: BalanceReport | None = None

    def summary(self) -> str:
        lines = [
            f"여정: {self.journey.title}",
            f"태스크: {len(self.journey.steps)}개",
            f"발견된 Agent: {len(self.agents)}개",
            f"MCP Tool: {len(self.mcp_tools)}개",
            f"경고(Hole): {len(self.holes)}개",
            f"중복(Overlap): {len(self.overlaps)}개",
        ]
        if self.coverage_metrics:
            lines.append(
                f"커버리지: {self.coverage_metrics.coverage_ratio:.1%} "
                f"({'완전 피복 ✅' if self.coverage_metrics.coverage_complete else '미완 ⚠️'})"
            )
        if self.balance_report:
            lines.append(f"Tool 균형: {self.balance_report.summary}")
        return "\n".join(lines)


class TADMapperPipeline:
    """
    TAD-Mapper 전체 파이프라인

    실행 순서:
    1.   입력 파싱 (JSON/CSV → UserJourney)
    2.   특징 추출 (Gemini LLM → 6D 벡터)
    3.   TDA Mapper 분석 (위상 그래프 생성)
    4.   Agent 자동 발견 (KMeans 클러스터링)
    5.   Agent 명명 (Gemini LLM → 이름/역할)
    6.   Hole/Overlap 탐지
    7.   MCP Tool 스키마 생성
    7.5  Tool 균형 분석 + Agile 재분배 [신규]
    8.   리포트 생성
    9.   태스크 임베딩 생성 [신규]
    10.  Query Manifold 구축 + 커버리지 분석 [신규]
    11.  Homotopy Router 초기화 [신규]
    """

    def __init__(
        self,
        n_agents: int | None = None,
        max_tools_per_agent: int = 7,
    ) -> None:
        validate_config()
        self.n_agents = n_agents
        self.max_tools_per_agent = max_tools_per_agent

        self._parser = JourneyParser()
        self._extractor = FeatureExtractor()
        self._embedder = Embedder()
        self._tda = TDAAnalyzer(
            n_intervals=TDA_N_INTERVALS,
            overlap_frac=TDA_OVERLAP_FRAC,
        )
        self._namer = AgentNamer()
        self._hole_detector = HoleDetector()
        self._mcp_gen = MCPGenerator()
        self._balancer = ToolBalancer(max_tools_per_agent=max_tools_per_agent)
        self._report_gen = ReportGenerator()
        self._visualizer = TDAVisualizer()
        self._composer = ToolComposer()

        # 런타임 라우팅용 (run() 완료 후 초기화)
        self._router: HomotopyRouter | None = None
        self._manifold: QueryManifold | None = None
        self._result: PipelineResult | None = None

    # ── 메인 파이프라인 ──────────────────────────────────────────────────────

    def run(self, input_path: str | Path) -> PipelineResult:
        """파이프라인 전체 실행"""
        logger.info(f"=== TAD-Mapper 파이프라인 시작: {input_path} ===")

        # 1. 입력 파싱
        logger.info("[1/11] 입력 파싱 중...")
        journey = self._parser.parse(input_path)
        logger.info(f"  → {len(journey.steps)}개 태스크 로드 완료")

        # 2. 특징 추출
        logger.info("[2/11] Gemini LLM으로 특징 추출 중...")
        features = self._extractor.extract(journey.steps)

        # 3. TDA Mapper 분석
        logger.info("[3/11] TDA Mapper 알고리즘 실행 중...")
        mapper_graph = self._tda.run_mapper(features)
        logger.info(f"  → {len(mapper_graph.nodes)}개 노드, {len(mapper_graph.edges)}개 엣지")

        # 4. Agent 자동 발견
        logger.info("[4/11] Agent 클러스터 자동 발견 중...")
        agents = self._tda.discover_agents(features, n_agents=self.n_agents)
        logger.info(f"  → {len(agents)}개 Agent 클러스터 발견")

        # 4.5. God Agent 방지: 태스크 과부하 Agent 자동 분할 [신규]
        logger.info("[4.5/11] God Agent 방지 - 클러스터 균형 정제 중...")
        agents = self._tda.refine_clusters(agents, features)
        logger.info(f"  → 정제 후 {len(agents)}개 Agent")

        # 5. Agent 명명
        logger.info("[5/11] Gemini LLM으로 Agent 이름/역할 부여 중...")
        agents = self._namer.name_agents(agents)


        # 6. Hole/Overlap 탐지
        logger.info("[6/11] Hole/Overlap 탐지 중...")
        holes = self._hole_detector.detect_holes(journey, agents, mapper_graph)
        overlaps = self._hole_detector.detect_overlaps(agents, mapper_graph)
        logger.info(f"  → Hole: {len(holes)}개, Overlap: {len(overlaps)}개")

        # 7. MCP Tool 스키마 생성
        logger.info("[7/11] MCP Tool 스키마 생성 중...")
        mcp_tools = self._mcp_gen.generate(journey, agents)
        logger.info(f"  → {len(mcp_tools)}개 MCP Tool 스키마 생성")

        # 7.5. Tool 균형 분석 + Agile 재분배 [신규]
        logger.info("[7.5/11] Tool 균형 분석 및 Agile 재분배 검토 중...")
        balance_report = self._balancer.analyze(agents, mcp_tools)
        if balance_report.overloaded_agents:
            logger.info(
                f"  → 오버로드 Agent 탐지: {balance_report.overloaded_agents}. "
                "Agile 재분배 시작..."
            )
            agents, mcp_tools, balance_report = self._balancer.rebalance(
                agents, features, mcp_tools
            )
            logger.info(
                f"  → 재분배 완료: {len(agents)}개 Agent, "
                f"Gini: {balance_report.gini_coefficient:.3f}"
            )
        else:
            logger.info(f"  → {balance_report.summary}")

        # 8. 리포트 생성 및 저장
        logger.info("[8/11] 리포트 생성 중...")
        report_md = self._report_gen.generate_markdown(
            journey.title, agents, mcp_tools, holes, overlaps, len(journey.steps)
        )
        report_json = self._report_gen.generate_json(
            journey.title, agents, mcp_tools, holes, overlaps
        )

        out_dir = OUTPUT_DIR / journey.id
        out_dir.mkdir(parents=True, exist_ok=True)
        self._report_gen.save(report_md, out_dir / "report.md")
        (out_dir / "result.json").write_text(
            json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 시각화
        fig_mapper = self._visualizer.create_mapper_graph(mapper_graph, agents, journey.title)
        self._visualizer.save_html(fig_mapper, out_dir / "mapper_graph.html")
        fig_radar = self._visualizer.create_feature_radar(agents, features)
        self._visualizer.save_html(fig_radar, out_dir / "feature_radar.html")

        # 9. 태스크 임베딩 생성 [신규]
        logger.info("[9/11] 태스크 임베딩 생성 중 (Query Manifold용)...")
        task_embeddings = None
        try:
            task_embeddings = self._embedder.embed_tasks(journey.steps)
            logger.info(f"  → 임베딩 완료: shape={task_embeddings.shape}")
        except Exception as e:
            logger.warning(f"  → 임베딩 생성 실패: {e}. 6D 벡터로 대체됩니다.")

        # 10. Query Manifold 구축 + 커버리지 분석 [신규]
        logger.info("[10/11] Query Manifold 구축 중 (Q ⊆ ∪Ui)...")
        self._manifold = QueryManifold()
        coverage_metrics = None
        try:
            self._manifold.build(agents, features, self._embedder, task_embeddings)
            coverage_metrics = self._manifold.compute_coverage()
            logger.info(
                f"  → 커버리지: {coverage_metrics.coverage_ratio:.1%} "
                f"({'완전 피복' if coverage_metrics.coverage_complete else '미완'})"
            )
            # Manifold 시각화
            try:
                fig_manifold = self._visualizer.create_query_manifold(self._manifold)
                self._visualizer.save_html(fig_manifold, out_dir / "query_manifold.html")
            except Exception as ve:
                logger.warning(f"  Manifold 시각화 실패: {ve}")
        except Exception as e:
            logger.warning(f"  → Query Manifold 구축 실패: {e}")

        # 11. Homotopy Router 초기화 [신규]
        logger.info("[11/11] Homotopy Router 초기화 중 (Φ: Q → {Uk})...")
        self._router = HomotopyRouter(self._embedder)
        try:
            self._router.build(agents, manifold=self._manifold)
            logger.info(
                f"  → Router 준비 완료: "
                f"{len(self._router.homotopy_classes)}개 호모토피 클래스"
            )
        except Exception as e:
            logger.warning(f"  → Router 초기화 실패: {e}")
            self._router = None

        logger.info(f"=== 완료! 결과 저장: {out_dir} ===")

        self._result = PipelineResult(
            journey=journey,
            features=features,
            mapper_graph=mapper_graph,
            agents=agents,
            mcp_tools=mcp_tools,
            holes=holes,
            overlaps=overlaps,
            report_markdown=report_md,
            report_json=report_json,
            task_embeddings=task_embeddings,
            coverage_metrics=coverage_metrics,
            balance_report=balance_report,
        )
        logger.info("\n%s", self._result.summary())
        return self._result

    # ── 런타임 라우팅 API ────────────────────────────────────────────────────

    def route_query(self, query_text: str) -> RoutingResult:
        """
        Φ(x) = U_k — 사용자 쿼리를 적절한 Agent로 라우팅합니다.

        run() 완료 후 호출 가능합니다.
        """
        if self._router is None or not self._router.is_ready:
            raise RuntimeError(
                "HomotopyRouter가 초기화되지 않았습니다. "
                "먼저 run()을 실행하세요."
            )
        return self._router.route(query_text)

    def compose_tools(self, query_text: str, agent_id: str) -> CompositionPlan:
        """
        y = (t_π(m) ∘ ... ∘ t_π(1))(x) — Tool 합성 계획을 생성합니다.
        """
        if self._result is None:
            raise RuntimeError("먼저 run()을 실행하세요.")

        agent = next(
            (a for a in self._result.agents if a.agent_id == agent_id), None
        )
        if agent is None:
            raise ValueError(f"Agent '{agent_id}'를 찾을 수 없습니다.")

        agent_tools = [
            t for t in self._result.mcp_tools
            if t.annotations and t.annotations.assigned_agent == agent_id
        ]

        plan = self._composer.compose(query_text, agent, agent_tools)
        plan.agent_id = agent_id
        return plan

    def route_and_compose(
        self, query_text: str
    ) -> tuple[RoutingResult, CompositionPlan]:
        """라우팅 + 합성을 한번에 실행합니다."""
        routing = self.route_query(query_text)
        composition = self.compose_tools(query_text, routing.target_agent_id)
        return routing, composition

    @property
    def router(self) -> HomotopyRouter | None:
        return self._router

    @property
    def manifold(self) -> QueryManifold | None:
        return self._manifold
