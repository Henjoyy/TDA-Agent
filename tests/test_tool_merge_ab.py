from __future__ import annotations

from tad_mapper.eval.tool_merge_ab import _routing_consistency, _tool_stats
from tad_mapper.models.mcp_tool import (
    MCPInputSchema,
    MCPPropertySchema,
    MCPToolAnnotations,
    MCPToolSchema,
)


class _R:
    def __init__(self, tools):
        self.mcp_tools = tools
        self.journey = type("J", (), {"steps": [1, 2, 3]})()
        self.agents = [1, 2]
        self.balance_report = type("B", (), {"gini_coefficient": 0.2})()


def _tool(name: str, task_ids: list[str]) -> MCPToolSchema:
    return MCPToolSchema(
        name=name,
        description=name,
        inputSchema=MCPInputSchema(
            properties={"query": MCPPropertySchema(type="string", description="q")},
            required=["query"],
        ),
        annotations=MCPToolAnnotations(
            assigned_agent="a1",
            source_task_id=task_ids[0],
            source_task_name=task_ids[0],
            source_task_ids=task_ids,
            source_task_names=task_ids,
            confidence=0.9,
        ),
    )


def test_tool_stats_handles_shared_tools():
    result = _R([
        _tool("t1", ["task1", "task2"]),
        _tool("t2", ["task3"]),
    ])
    stats = _tool_stats(result)
    assert stats["tool_count"] == 2
    assert stats["shared_tool_count"] == 1
    assert stats["max_tasks_per_tool"] == 2
    assert stats["avg_tasks_per_tool"] == 1.5


def test_routing_consistency_computation():
    base = [
        {"target_agent_id": "a1", "confidence": 0.4},
        {"target_agent_id": "a2", "confidence": 0.7},
    ]
    merged = [
        {"target_agent_id": "a1", "confidence": 0.6},
        {"target_agent_id": "a3", "confidence": 0.5},
    ]
    c = _routing_consistency(
        base,
        merged,
        {"a1": "g1", "a2": "g2"},
        {"a1": "g1", "a3": "g2"},
    )
    assert c["n_queries"] == 2
    assert c["exact_agent_consistency"] == 0.5
    assert c["group_consistency"] == 1.0
