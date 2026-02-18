"""
ToolComposer 내부 그래프/정렬 로직 단위 테스트
"""
from __future__ import annotations

from tad_mapper.engine.tool_composer import ToolComposer
from tad_mapper.models.mcp_tool import MCPInputSchema, MCPPropertySchema, MCPToolSchema


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
