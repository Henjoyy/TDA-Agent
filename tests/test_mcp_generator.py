"""
MCPGenerator 안정성 테스트
"""
from __future__ import annotations

import numpy as np

from tad_mapper.engine.tda_analyzer import DiscoveredAgent
from tad_mapper.models.journey import TaskStep, UserJourney
from tad_mapper.output.mcp_generator import MCPGenerator


def make_journey(n_tasks: int = 5) -> UserJourney:
    steps = [
        TaskStep(
            id=f"t{i}",
            name=f"태스크 {i}",
            description=f"설명 {i}",
            input_data=["query"],
            output_data=["result"],
        )
        for i in range(n_tasks)
    ]
    return UserJourney(id="j1", title="J1", steps=steps)


def make_agent(task_ids: list[str]) -> DiscoveredAgent:
    return DiscoveredAgent(
        agent_id="agent_0",
        cluster_id=0,
        task_ids=task_ids,
        task_names=[f"태스크 {tid}" for tid in task_ids],
        centroid=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        suggested_name="테스트 에이전트",
        suggested_role="테스트",
    )


def test_generate_splits_into_chunks():
    journey = make_journey(5)
    agent = make_agent([s.id for s in journey.steps])

    generator = MCPGenerator.__new__(MCPGenerator)
    generator._batch_size = 2
    generator._timeout_ms = 1_000
    generator._max_retries = 0

    calls: list[list[str]] = []

    def fake_generate_batch(tasks, task_to_agent):
        calls.append([t.id for t in tasks])
        return {}

    generator._generate_batch = fake_generate_batch

    tools = generator.generate(journey, [agent])

    assert len(calls) == 3
    assert [len(c) for c in calls] == [2, 2, 1]
    assert len(tools) == len(journey.steps)


def test_generate_batch_retry_is_bounded(monkeypatch):
    journey = make_journey(1)
    agent = make_agent([s.id for s in journey.steps])
    task_to_agent = {journey.steps[0].id: agent}

    class DummyModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, *args, **kwargs):
            self.calls += 1
            raise TimeoutError("timeout")

    class DummyClient:
        def __init__(self):
            self.models = DummyModels()

    monkeypatch.setattr("tad_mapper.output.mcp_generator.time.sleep", lambda _: None)

    generator = MCPGenerator.__new__(MCPGenerator)
    generator._client = DummyClient()
    generator._batch_size = 2
    generator._timeout_ms = 1_000
    generator._max_retries = 2

    result = generator._generate_batch(journey.steps, task_to_agent)

    assert result == {}
    assert generator._client.models.calls == 3
