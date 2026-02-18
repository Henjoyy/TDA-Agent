"""
AgentNamer 안정성 테스트
"""
from __future__ import annotations

import numpy as np

from tad_mapper.engine.tda_analyzer import DiscoveredAgent
from tad_mapper.mapper.agent_namer import AgentNamer


def make_agent() -> DiscoveredAgent:
    return DiscoveredAgent(
        agent_id="agent_0",
        cluster_id=0,
        task_ids=["t1", "t2"],
        task_names=["데이터 수집", "데이터 분석"],
        centroid=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    )


def test_apply_naming_does_not_mutate_agent_id():
    namer = AgentNamer.__new__(AgentNamer)
    agent = make_agent()
    original_id = agent.agent_id

    namer._apply_naming(
        agent,
        {
            "name_ko": "데이터 분석 에이전트",
            "role": "데이터 분석을 수행",
            "capabilities": ["analysis"],
            "mcp_tool_prefix": "data_analysis",
        },
    )

    assert agent.agent_id == original_id
    assert agent.tool_prefix == "data_analysis"


def test_keyword_fallback_sets_tool_prefix():
    namer = AgentNamer.__new__(AgentNamer)
    agent = make_agent()

    namer._apply_keyword_fallback(agent)

    assert agent.tool_prefix != ""
