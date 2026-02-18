"""
ToolComposer 내부 그래프/정렬 로직 단위 테스트
"""
from __future__ import annotations

from tad_mapper.engine.tool_composer import ToolComposer
from tad_mapper.models.mcp_tool import (
    MCPInputSchema,
    MCPPropertySchema,
    MCPToolAnnotations,
    MCPToolSchema,
)


def make_tool(name: str, required: list[str], props: list[str]) -> MCPToolSchema:
    return MCPToolSchema(
        name=name,
        description=f"{name} description",
        inputSchema=MCPInputSchema(
            properties={
                key: MCPPropertySchema(type="string", description=key) for key in props
            },
            required=required,
        ),
    )


def make_annotated_tool(name: str, task_name: str) -> MCPToolSchema:
    return MCPToolSchema(
        name=name,
        description=f"{task_name} 처리",
        inputSchema=MCPInputSchema(
            properties={"query": MCPPropertySchema(type="string", description="입력")},
            required=["query"],
        ),
        annotations=MCPToolAnnotations(
            assigned_agent="agent_0",
            source_task_id=task_name,
            source_task_name=task_name,
            source_task_ids=[task_name],
            source_task_names=[task_name],
            confidence=0.9,
        ),
    )


def test_dependency_graph_and_topological_sort_are_consistent():
    composer = ToolComposer.__new__(ToolComposer)
    tools = [
        make_tool("fetch_data", required=["query"], props=["query"]),
        make_tool("analyze_data", required=["raw_query"], props=["raw_query", "records"]),
        make_tool("generate_report", required=["records"], props=["records", "report"]),
    ]

    dep_graph = composer._build_dependency_graph(tools)
    assert "analyze_data" in dep_graph["generate_report"]

    order = composer._topological_sort(tools, dep_graph)
    assert order.index("analyze_data") < order.index("generate_report")


def test_select_candidate_tools_prefers_relevant_source_tasks():
    composer = ToolComposer.__new__(ToolComposer)
    tools = [
        make_annotated_tool("analyze_trade_risk", "무역 리스크 분석"),
        make_annotated_tool("send_fx_alert", "환율 알림 전송"),
        make_annotated_tool("generate_contract", "계약서 생성"),
    ]

    selected = composer._select_candidate_tools(
        "환율 알림 보내줘",
        tools,
        max_candidates=2,
    )
    names = [t.name for t in selected]
    assert "send_fx_alert" in names
    assert "generate_contract" not in names
