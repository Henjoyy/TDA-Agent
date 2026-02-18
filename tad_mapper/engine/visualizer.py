"""
TDA 시각화 모듈 - Plotly 기반 인터랙티브 그래프

신규:
- create_query_manifold() : Query Manifold + Agent Coverage Region 시각화
- create_composition_flow(): Tool 합성 흐름도 (Sankey diagram)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tad_mapper.engine.tda_analyzer import DiscoveredAgent, MapperGraph
from tad_mapper.engine.feature_extractor import TopologicalFeature

if TYPE_CHECKING:
    from tad_mapper.engine.query_manifold import QueryManifold
    from tad_mapper.models.topology import CompositionPlan, RoutingResult

logger = logging.getLogger(__name__)

# Agent별 색상 팔레트
AGENT_COLORS = [
    "#6366f1", "#f59e0b", "#10b981", "#ef4444",
    "#8b5cf6", "#06b6d4", "#f97316", "#84cc16",
]


class TDAVisualizer:
    """Plotly 기반 TDA 결과 시각화"""

    def create_mapper_graph(
        self,
        graph: MapperGraph,
        agents: list[DiscoveredAgent],
        title: str = "TAD-Mapper: 위상 그래프",
    ) -> go.Figure:
        """Mapper 그래프 시각화 (노드 = 클러스터, 엣지 = 공유 태스크)"""

        # 노드 위치: PCA 2D 투영 기반
        node_positions: dict[int, tuple[float, float]] = {}
        for node in graph.nodes:
            # 간단히 centroid 기반 위치 사용
            x = float(node.centroid[0]) if len(node.centroid) > 0 else 0.0
            y = float(node.centroid[1]) if len(node.centroid) > 1 else 0.0
            node_positions[node.node_id] = (x, y)

        # 태스크 ID → Agent 색상 매핑
        task_agent_color: dict[str, str] = {}
        task_agent_name: dict[str, str] = {}
        for i, agent in enumerate(agents):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            name = agent.suggested_name or f"Agent {i}"
            for tid in agent.task_ids:
                task_agent_color[tid] = color
                task_agent_name[tid] = name

        fig = go.Figure()

        # 엣지 그리기
        for (n1, n2) in graph.edges:
            if n1 in node_positions and n2 in node_positions:
                x0, y0 = node_positions[n1]
                x1, y1 = node_positions[n2]
                fig.add_trace(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode="lines",
                    line=dict(color="#94a3b8", width=1.5),
                    hoverinfo="none",
                    showlegend=False,
                ))

        # 노드 그리기 (Agent별 색상)
        for node in graph.nodes:
            if node.node_id not in node_positions:
                continue
            x, y = node_positions[node.node_id]
            # 노드의 대표 색상 (첫 번째 태스크의 Agent 색상)
            color = task_agent_color.get(node.task_ids[0], "#94a3b8") if node.task_ids else "#94a3b8"
            agent_name = task_agent_name.get(node.task_ids[0], "미분류") if node.task_ids else "미분류"
            hover_text = f"<b>{agent_name}</b><br>" + "<br>".join(
                f"• {name}" for name in node.task_names
            )
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=20 + len(node.task_ids) * 5, color=color, opacity=0.85,
                            line=dict(color="white", width=2)),
                text=[str(len(node.task_ids))],
                textposition="middle center",
                textfont=dict(color="white", size=11, family="Arial Black"),
                hovertext=hover_text,
                hoverinfo="text",
                showlegend=False,
            ))

        # Agent 범례
        for i, agent in enumerate(agents):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            name = agent.suggested_name or f"Agent {i}"
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=color),
                name=f"{name} ({len(agent.task_ids)}개 태스크)",
                showlegend=True,
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color="#1e293b")),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="white",
            legend=dict(orientation="v", x=1.02, y=1),
            margin=dict(l=20, r=160, t=60, b=20),
            height=500,
        )
        return fig

    def create_feature_radar(
        self,
        agents: list[DiscoveredAgent],
        features: list[TopologicalFeature],
    ) -> go.Figure:
        """Agent별 평균 특징 벡터 레이더 차트"""
        feature_names = [
            "데이터 유형", "추론 깊이", "자동화 가능성",
            "상호작용", "출력 복잡도", "도메인 특화",
            "시간 민감도", "데이터 규모", "보안 수준", "상태 의존성",
        ]

        fig = go.Figure()
        feature_map = {f.task_id: f for f in features}

        for i, agent in enumerate(agents):
            agent_features = [feature_map[tid] for tid in agent.task_ids if tid in feature_map]
            if not agent_features:
                continue
            avg_vec = np.mean([f.vector for f in agent_features], axis=0)
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            name = agent.suggested_name or f"Agent {i}"
            dim_names = feature_names[: len(avg_vec)]

            fig.add_trace(go.Scatterpolar(
                r=list(avg_vec) + [avg_vec[0]],
                theta=dim_names + [dim_names[0]],
                fill="toself",
                fillcolor=color,
                opacity=0.3,
                line=dict(color=color, width=2),
                name=name,
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Agent별 특징 프로파일",
            height=450,
            paper_bgcolor="white",
        )
        return fig

    def create_query_manifold(
        self,
        manifold: "QueryManifold",
        routing_result: "RoutingResult | None" = None,
    ) -> go.Figure:
        """
        Query Manifold 시각화 (수학적 정식화 #1: Q ⊆ ∪Ui)

        - 태스크 임베딩을 2D 산점도로 표시 (Agent별 색상)
        - Agent centroid를 큰 마커로 표시
        - 라우팅 결과가 있으면 쿼리 포인트도 표시
        """
        fig = go.Figure()

        projected = manifold.projected_2d
        if projected is None or len(projected) == 0:
            fig.add_annotation(text="임베딩 데이터 없음", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
            return fig

        regions = manifold.regions
        task_region_map: dict[int, str] = {}  # 포인트 인덱스 → agent_id

        # 포인트 → 가장 가까운 region 매핑
        for idx, pt in enumerate(projected):
            best_aid = regions[0].agent_id if regions else "unknown"
            best_dist = float("inf")
            for region in regions:
                cp = np.array(region.projected_centroid_2d) if region.projected_centroid_2d else np.zeros(2)
                dist = np.linalg.norm(pt - cp)
                if dist < best_dist:
                    best_dist = dist
                    best_aid = region.agent_id
            task_region_map[idx] = best_aid

        # Agent별 태스크 포인트 그리기
        for i, region in enumerate(regions):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            indices = [idx for idx, aid in task_region_map.items() if aid == region.agent_id]
            if not indices:
                continue
            pts = projected[indices]
            fig.add_trace(go.Scatter(
                x=pts[:, 0], y=pts[:, 1],
                mode="markers",
                marker=dict(size=10, color=color, opacity=0.7,
                            line=dict(color="white", width=1)),
                name=region.agent_name or region.agent_id,
                hovertext=[f"{region.agent_name}<br>태스크 포인트" for _ in indices],
                hoverinfo="text",
            ))

        # Agent centroid 마커
        for i, region in enumerate(regions):
            if not region.projected_centroid_2d:
                continue
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            cx, cy = region.projected_centroid_2d[0], region.projected_centroid_2d[1]
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers+text",
                marker=dict(size=24, color=color, symbol="diamond",
                            line=dict(color="white", width=2)),
                text=[region.agent_name or region.agent_id],
                textposition="top center",
                textfont=dict(size=11, color="#1e293b"),
                hovertext=(
                    f"<b>{region.agent_name}</b><br>"
                    f"Coverage radius: {region.radius:.3f}<br>"
                    f"Voronoi area: {region.voronoi_area:.3f}"
                ),
                hoverinfo="text",
                showlegend=False,
            ))

        # 라우팅 결과 쿼리 포인트
        if routing_result is not None:
            fig.add_trace(go.Scatter(
                x=[0], y=[0],  # 중심에 표시 (실제 투영 좌표 없음 시)
                mode="markers+text",
                marker=dict(size=18, color="#dc2626", symbol="star",
                            line=dict(color="white", width=2)),
                text=["쿼리"],
                textposition="top center",
                hovertext=(
                    f"<b>쿼리:</b> {routing_result.query_text[:50]}<br>"
                    f"→ {routing_result.target_agent_name} "
                    f"(신뢰도: {routing_result.confidence:.2f})"
                ),
                hoverinfo="text",
                name="사용자 쿼리",
            ))

        fig.update_layout(
            title=dict(
                text="Query Manifold — Q ⊆ ∪ Ui (에이전트 피복 시각화)",
                font=dict(size=16, color="#1e293b")
            ),
            xaxis=dict(title="PCA 1", showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(title="PCA 2", showgrid=True, gridcolor="#f1f5f9"),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="white",
            legend=dict(orientation="v", x=1.02, y=1),
            margin=dict(l=40, r=160, t=60, b=40),
            height=520,
        )
        return fig

    def create_composition_flow(self, plan: "CompositionPlan") -> go.Figure:
        """
        Tool 합성 흐름도 (수학적 정식화 #3)

        y = (t_π(m) ∘ ... ∘ t_π(1))(x) 를 Sankey diagram으로 시각화
        """
        steps = sorted(plan.steps, key=lambda s: s.order)

        if not steps:
            fig = go.Figure()
            fig.add_annotation(text="합성 단계 없음", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
            return fig

        # 노드 목록: query + 각 Tool
        node_labels = ["사용자 쿼리"] + [s.tool_name for s in steps] + ["최종 결과"]
        node_colors = ["#64748b"] + [
            AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(len(steps))
        ] + ["#10b981"]

        # 링크 생성
        node_idx = {label: i for i, label in enumerate(node_labels)}
        sources, targets, values, labels = [], [], [], []

        for step in steps:
            # input_from → tool
            src_label = "사용자 쿼리" if step.input_from == "query" else step.input_from
            src = node_idx.get(src_label, 0)
            tgt = node_idx.get(step.tool_name, 1)
            sources.append(src)
            targets.append(tgt)
            values.append(1)
            labels.append(step.rationale[:30] if step.rationale else "")

            # tool → output_to (마지막 단계면 "최종 결과"로)
            if step.output_to == "final":
                sources.append(tgt)
                targets.append(node_idx["최종 결과"])
                values.append(1)
                labels.append("")

        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                label=labels,
            ),
        ))

        fig.update_layout(
            title=dict(
                text=(
                    f"Tool 합성 흐름도 — {plan.agent_name}<br>"
                    f"<sub>y = (t_π({len(steps)}) ∘ ... ∘ t_π(1))(x)</sub>"
                ),
                font=dict(size=14, color="#1e293b"),
            ),
            height=400,
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=80, b=20),
        )
        return fig

    def save_html(self, fig: go.Figure, path: str | Path) -> None:
        """Plotly 그래프를 HTML 파일로 저장"""
        fig.write_html(str(path), include_plotlyjs="cdn")
        logger.info(f"시각화 저장: {path}")

    def to_json(self, fig: go.Figure) -> str:
        """Plotly 그래프를 JSON 문자열로 변환 (웹 전송용)"""
        return fig.to_json()
