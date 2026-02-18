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
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config.settings import (
    HIERARCHY_MAX_ORCHESTRATORS,
    HIERARCHY_MIN_ORCHESTRATOR_PROB,
    HIERARCHY_SIMPLE_THRESHOLD,
    OUTPUT_DIR,
    ROUTER_DISABLE_ON_FALLBACK_RATIO,
    ROUTER_MAX_FALLBACK_RATIO,
    ROUTER_MIN_EMBED_CALLS,
    TDA_N_INTERVALS,
    TDA_OVERLAP_FRAC,
    validate_config,
)
from tad_mapper.engine.embedder import Embedder
from tad_mapper.engine.feature_extractor import FeatureExtractor, TopologicalFeature
from tad_mapper.engine.homotopy_router import HomotopyRouter
from tad_mapper.engine.hierarchy_planner import HierarchyPlanner
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
    HierarchicalExecutionPlan,
    HierarchicalExecutionStep,
    HierarchicalRoutingPlan,
    HierarchyBlueprint,
    RoutingResult,
)
from tad_mapper.output.mcp_generator import MCPGenerator
from tad_mapper.output.report_generator import ReportGenerator

logger = logging.getLogger(__name__)
_KEYWORD_PATTERN = re.compile(r"[0-9A-Za-z가-힣]{2,}")
_STOPWORDS = {
    "execute", "task", "agent", "tool", "query", "data",
    "처리", "작업", "요청", "기능", "태스크", "에이전트",
}


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
    embedding_health: dict = field(default_factory=dict)
    router_block_reason: str = ""
    hierarchy_blueprint: HierarchyBlueprint | None = None

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
        if self.embedding_health:
            lines.append(
                f"임베딩 상태: model={self.embedding_health.get('active_model')} "
                f"| fallback={self.embedding_health.get('fallback_calls', 0)}/"
                f"{self.embedding_health.get('total_calls', 0)} "
                f"({self.embedding_health.get('fallback_ratio', 0.0):.1%})"
            )
        if self.router_block_reason:
            lines.append(f"Router 상태: 비활성화 ({self.router_block_reason})")
        if self.hierarchy_blueprint:
            orches = len(self.hierarchy_blueprint.orchestrator_to_units)
            lines.append(
                f"계층 구조: Master 1, Orchestrator {orches}개, Unit {len(self.agents)}개"
            )
        return "\n".join(lines)


class TADMapperPipeline:
    """
    TAD-Mapper 전체 파이프라인

    실행 순서:
    1.   입력 파싱 (JSON/CSV → UserJourney)
    2.   특징 추출 (Gemini LLM → 10D 벡터)
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
        self._hierarchy_planner = HierarchyPlanner(
            simple_threshold=HIERARCHY_SIMPLE_THRESHOLD,
            max_orchestrators=HIERARCHY_MAX_ORCHESTRATORS,
            min_orchestrator_prob=HIERARCHY_MIN_ORCHESTRATOR_PROB,
        )

        # 런타임 라우팅용 (run() 완료 후 초기화)
        self._router: HomotopyRouter | None = None
        self._manifold: QueryManifold | None = None
        self._hierarchy: HierarchyBlueprint | None = None
        self._result: PipelineResult | None = None
        self._router_block_reason: str = ""

    # ── 메인 파이프라인 ──────────────────────────────────────────────────────

    def run(self, input_path: str | Path) -> PipelineResult:
        """파이프라인 전체 실행"""
        logger.info(f"=== TAD-Mapper 파이프라인 시작: {input_path} ===")
        self._router_block_reason = ""
        self._embedder.reset_health_stats()

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

        # 라우팅 품질 보강: MCP Tool 의미 정보를 Agent에 주입
        self._enrich_agents_for_routing(agents, mcp_tools)
        self._hierarchy = self._hierarchy_planner.build(agents)

        # 8. 리포트 생성 및 저장
        logger.info("[8/11] 리포트 생성 중...")
        report_md = self._report_gen.generate_markdown(
            journey.title, agents, mcp_tools, holes, overlaps, len(journey.steps)
        )
        report_json = self._report_gen.generate_json(
            journey.title, agents, mcp_tools, holes, overlaps, features=features
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
            logger.warning(f"  → 임베딩 생성 실패: {e}. 10D 벡터로 대체됩니다.")
        embedding_health = self._embedder.get_health_dict()
        logger.info("  → 임베딩 건강도: %s", embedding_health)

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
        self._router_block_reason = self._evaluate_router_block_reason(
            embedding_health
        )
        if self._router_block_reason:
            logger.error("  → Router 비활성화: %s", self._router_block_reason)
            self._router = None
        else:
            self._router = HomotopyRouter(self._embedder)
            try:
                self._router.build(agents, manifold=self._manifold)
                logger.info(
                    f"  → Router 준비 완료: "
                    f"{len(self._router.homotopy_classes)}개 호모토피 클래스"
                )
            except Exception as e:
                self._router_block_reason = f"Router 초기화 실패: {e}"
                logger.warning(f"  → {self._router_block_reason}")
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
            embedding_health=embedding_health,
            router_block_reason=self._router_block_reason,
            hierarchy_blueprint=self._hierarchy,
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

    def plan_hierarchy(self, query_text: str) -> HierarchicalRoutingPlan:
        """
        Master → (Orchestrator) → Unit 계층 라우팅 계획을 생성합니다.
        """
        if self._result is None:
            raise RuntimeError("먼저 run()을 실행하세요.")
        if self._hierarchy is None:
            raise RuntimeError("계층 구조가 초기화되지 않았습니다.")
        routing = self.route_query(query_text)
        return self._hierarchy_planner.plan(query_text, routing)

    def route_hierarchy_and_compose(
        self, query_text: str
    ) -> HierarchicalExecutionPlan:
        """
        계층 라우팅 + 서브태스크별 Unit Tool 합성을 한번에 실행합니다.

        - path_type=master_unit: 동일 Unit에 대해 query 단위 합성 1회
        - path_type=master_orchestrator_unit: assignment별 subtask 합성
        """
        hierarchical = self.plan_hierarchy(query_text)
        execution_steps: list[HierarchicalExecutionStep] = []

        if hierarchical.path_type == "master_unit":
            if not hierarchical.selected_unit_ids:
                raise RuntimeError("계층 계획에 선택된 Unit이 없습니다.")
            unit_id = hierarchical.selected_unit_ids[0]
            composition = self.compose_tools(query_text, unit_id)
            execution_steps.append(
                HierarchicalExecutionStep(
                    subtask_id="subtask_1",
                    subtask_text=query_text,
                    source_subtask_ids=["subtask_1"],
                    source_subtask_texts=[query_text],
                    unit_agent_id=unit_id,
                    composition_plan=composition,
                )
            )
        else:
            merged_assignments = self._merge_assignments_for_execution(
                hierarchical.assignments
            )
            for merged in merged_assignments:
                composition = self.compose_tools(
                    merged["merged_text"], merged["unit_agent_id"]
                )
                execution_steps.append(
                    HierarchicalExecutionStep(
                        subtask_id=merged["merged_subtask_id"],
                        subtask_text=merged["merged_text"],
                        source_subtask_ids=merged["source_ids"],
                        source_subtask_texts=merged["source_texts"],
                        orchestrator_id=merged["orchestrator_id"],
                        unit_agent_id=merged["unit_agent_id"],
                        composition_plan=composition,
                    )
                )

        return HierarchicalExecutionPlan(
            query_text=query_text,
            hierarchical_routing=hierarchical,
            execution_steps=execution_steps,
        )

    @staticmethod
    def _merge_assignments_for_execution(assignments: list) -> list[dict]:
        """
        동일 Orchestrator/Unit으로 향하는 서브태스크를 하나로 병합합니다.

        병합 목적:
        - 동일 Unit에 대한 compose 호출 횟수 축소
        - 공유 Tool 기반 실행 최적화
        """
        grouped: dict[tuple[str, str], list] = defaultdict(list)
        for assignment in assignments:
            key = (assignment.orchestrator_id, assignment.unit_agent_id)
            grouped[key].append(assignment)

        merged: list[dict] = []
        for (orch_id, unit_id), rows in grouped.items():
            source_ids = [r.subtask_id for r in rows]
            source_texts = [r.subtask_text for r in rows]
            if len(rows) == 1:
                merged.append(
                    {
                        "merged_subtask_id": rows[0].subtask_id,
                        "merged_text": rows[0].subtask_text,
                        "source_ids": source_ids,
                        "source_texts": source_texts,
                        "orchestrator_id": orch_id,
                        "unit_agent_id": unit_id,
                    }
                )
                continue

            # 동일 Unit 작업을 한 번에 처리할 수 있도록 합성 입력을 통합
            merged_text = " 그리고 ".join(source_texts)
            merged_id = "+".join(source_ids)
            merged.append(
                {
                    "merged_subtask_id": merged_id,
                    "merged_text": merged_text,
                    "source_ids": source_ids,
                    "source_texts": source_texts,
                    "orchestrator_id": orch_id,
                    "unit_agent_id": unit_id,
                }
            )

        # 결정적 순서 보장
        merged.sort(key=lambda x: (x["orchestrator_id"], x["unit_agent_id"], x["merged_subtask_id"]))
        return merged

    @staticmethod
    def _enrich_agents_for_routing(
        agents: list[DiscoveredAgent],
        mcp_tools: list[MCPToolSchema],
    ) -> None:
        """
        Tool 메타데이터로 Agent 의미 정보를 보강합니다.
        - capabilities에 도메인 키워드 추가
        - generic한 이름/역할은 키워드 기반으로 최소 보정
        """
        texts_by_agent: dict[str, list[str]] = defaultdict(list)
        for tool in mcp_tools:
            ann = tool.annotations
            if ann is None:
                continue
            aid = ann.assigned_agent
            if not aid:
                continue
            texts_by_agent[aid].append(tool.name.replace("_", " "))
            if tool.description:
                texts_by_agent[aid].append(tool.description)

        for agent in agents:
            texts = texts_by_agent.get(agent.agent_id, [])
            if not texts:
                continue
            tokens: list[str] = []
            for text in texts:
                for token in _KEYWORD_PATTERN.findall(text.lower()):
                    if token in _STOPWORDS or len(token) < 2:
                        continue
                    tokens.append(token)
            if not tokens:
                continue
            top_keywords = [k for k, _ in Counter(tokens).most_common(8)]

            # capabilities 확장
            merged_caps = list(dict.fromkeys([
                *(agent.suggested_capabilities or []),
                *top_keywords,
            ]))
            agent.suggested_capabilities = merged_caps[:10]

            # 이름/역할이 generic할 때만 최소 보정
            if not agent.suggested_name or "태스크 그룹" in agent.suggested_name:
                name_keywords = top_keywords[:2]
                if name_keywords:
                    agent.suggested_name = " ".join(name_keywords) + " 에이전트"
            if not agent.suggested_role or "태스크 그룹" in agent.suggested_role:
                role_keywords = ", ".join(top_keywords[:4])
                if role_keywords:
                    agent.suggested_role = f"{role_keywords} 관련 요청 처리"

    @property
    def router(self) -> HomotopyRouter | None:
        return self._router

    @property
    def manifold(self) -> QueryManifold | None:
        return self._manifold

    @property
    def hierarchy(self) -> HierarchyBlueprint | None:
        return self._hierarchy

    @property
    def router_block_reason(self) -> str:
        return self._router_block_reason

    @staticmethod
    def _evaluate_router_block_reason(embedding_health: dict) -> str:
        model_available = bool(embedding_health.get("model_available", True))
        if not model_available:
            model_name = embedding_health.get("active_model", "")
            reason = embedding_health.get("model_validation_error") or "unknown"
            return f"임베딩 모델 사용 불가({model_name}): {reason}"

        total_calls = int(embedding_health.get("total_calls", 0))
        fallback_ratio = float(embedding_health.get("fallback_ratio", 0.0))
        if (
            total_calls >= ROUTER_MIN_EMBED_CALLS
            and fallback_ratio > ROUTER_MAX_FALLBACK_RATIO
        ):
            message = (
                f"임베딩 fallback 비율 과다({fallback_ratio:.1%}) "
                f"- 임계값 {ROUTER_MAX_FALLBACK_RATIO:.1%} 초과"
            )
            if ROUTER_DISABLE_ON_FALLBACK_RATIO:
                return message
            logger.warning(
                "%s. 라우터는 hybrid(lexical 보정) 모드로 계속 동작합니다.",
                message,
            )
            return ""
        return ""
