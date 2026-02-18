"""
MCPGenerator 안정성 테스트
"""
from __future__ import annotations

import numpy as np
from concurrent.futures import Future

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
    generator._merge_enabled = False

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
    generator._merge_enabled = False

    result = generator._generate_batch(journey.steps, task_to_agent)

    assert result == {}
    assert generator._client.models.calls == 3


def test_timeout_can_be_disabled(monkeypatch):
    captured: dict[str, object] = {}

    class DummyClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.models = None

    monkeypatch.setenv("TAD_MCP_TIMEOUT_MS", "0")
    monkeypatch.setattr("tad_mapper.output.mcp_generator.genai.Client", DummyClient)

    generator = MCPGenerator()

    assert generator._timeout_ms is None
    assert "http_options" not in captured


def test_generate_parallel_chunks_when_workers_gt_one(monkeypatch):
    journey = make_journey(4)
    agent = make_agent([s.id for s in journey.steps])

    generator = MCPGenerator.__new__(MCPGenerator)
    generator._batch_size = 1
    generator._timeout_ms = None
    generator._max_retries = 0
    generator._max_workers = 4
    generator._merge_enabled = False

    executor_used = {"max_workers": 0}

    class FakeExecutor:
        def __init__(self, max_workers):
            executor_used["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            fut = Future()
            fut.set_result(fn(*args, **kwargs))
            return fut

    def fake_generate_batch(tasks, task_to_agent):
        task = tasks[0]
        return {
            task.id: {
                "name": f"tool_{task.id}",
                "description": "desc",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "q"},
                    },
                    "required": ["query"],
                },
            }
        }

    generator._generate_batch = fake_generate_batch
    monkeypatch.setattr("tad_mapper.output.mcp_generator.ThreadPoolExecutor", FakeExecutor)
    tools = generator.generate(journey, [agent])

    assert len(tools) == len(journey.steps)
    assert executor_used["max_workers"] == len(journey.steps)


def test_generate_merges_similar_tools_into_shared_tool():
    journey = UserJourney(
        id="j2",
        title="J2",
        steps=[
            TaskStep(
                id="t1",
                name="무역 리스크 분석",
                description="무역 리스크를 분석해줘",
                input_data=["query"],
                output_data=["risk_report"],
            ),
            TaskStep(
                id="t2",
                name="무역 위험도 평가",
                description="무역 위험도를 평가해줘",
                input_data=["query"],
                output_data=["risk_score"],
            ),
            TaskStep(
                id="t3",
                name="알림 전송",
                description="결과를 알림으로 전송",
                input_data=["message"],
                output_data=["status"],
            ),
        ],
    )
    agent = make_agent([s.id for s in journey.steps])

    generator = MCPGenerator.__new__(MCPGenerator)
    generator._batch_size = 10
    generator._timeout_ms = None
    generator._max_retries = 0
    generator._max_workers = 1
    generator._merge_enabled = True
    generator._merge_min_similarity = 0.45
    generator._max_tasks_per_tool = 4

    def fake_generate_batch(tasks, task_to_agent):
        out = {}
        for task in tasks:
            if task.id in {"t1", "t2"}:
                out[task.id] = {
                    "name": "analyze_trade_risk",
                    "description": "무역 리스크를 분석합니다",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "분석 요청"},
                            "region": {"type": "string", "description": "지역"},
                        },
                        "required": ["query"],
                    },
                }
            else:
                out[task.id] = {
                    "name": "send_alert",
                    "description": "알림을 전송합니다",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "메시지"},
                        },
                        "required": ["message"],
                    },
                }
        return out

    generator._generate_batch = fake_generate_batch
    tools = generator.generate(journey, [agent])

    assert len(tools) == 2
    merged = next(t for t in tools if t.name.startswith("analyze_trade_risk"))
    assert merged.annotations is not None
    assert set(merged.annotations.all_task_ids()) == {"t1", "t2"}
    assert merged.annotations.assigned_agent == "agent_0"


def test_generate_task_dedup_reuses_representative_llm_schema():
    journey = UserJourney(
        id="j3",
        title="J3",
        steps=[
            TaskStep(
                id="t1",
                name="환율 조회",
                description="환율 데이터를 조회해줘",
                input_data=["currency_pair"],
                output_data=["rate"],
            ),
            TaskStep(
                id="t2",
                name="환율 조회",
                description="환율 데이터를 조회해줘",
                input_data=["currency_pair"],
                output_data=["rate"],
            ),
            TaskStep(
                id="t3",
                name="리스크 분석",
                description="거래 리스크를 분석해줘",
                input_data=["portfolio"],
                output_data=["risk_report"],
            ),
            TaskStep(
                id="t4",
                name="리스크 분석",
                description="거래 리스크를 분석해줘",
                input_data=["portfolio"],
                output_data=["risk_report"],
            ),
        ],
    )
    agent = make_agent([s.id for s in journey.steps])

    generator = MCPGenerator.__new__(MCPGenerator)
    generator._batch_size = 10
    generator._timeout_ms = None
    generator._max_retries = 0
    generator._max_workers = 1
    generator._merge_enabled = False
    generator._dedup_enabled = True
    generator._dedup_min_tasks = 1
    generator._dedup_min_similarity = 0.7

    llm_calls: list[list[str]] = []

    def fake_generate_batch(tasks, task_to_agent):
        llm_calls.append([t.id for t in tasks])
        out = {}
        for task in tasks:
            out[task.id] = {
                "name": f"tool_{task.id}",
                "description": "desc",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "q"},
                    },
                    "required": ["query"],
                },
            }
        return out

    generator._generate_batch = fake_generate_batch
    tools = generator.generate(journey, [agent])

    assert len(llm_calls) == 1
    assert len(llm_calls[0]) == 2  # 대표 태스크만 LLM 호출
    assert len(tools) == 4
    assert set(t.name for t in tools) == {"tool_t1", "tool_t3"}


def test_generate_task_dedup_uses_structural_compatibility():
    journey = UserJourney(
        id="j4",
        title="J4",
        steps=[
            TaskStep(
                id="t1",
                name="월간 매출 리포트 생성",
                description="ERP 매출 데이터를 기준으로 월간 리포트를 생성합니다.",
                input_data=["ERP 매출원장", "기준월"],
                output_data=["월간매출리포트", "핵심지표"],
                tags=["재무", "리포트", "정형데이터"],
            ),
            TaskStep(
                id="t2",
                name="주간 매출 리포트 생성",
                description="ERP 매출 데이터를 기준으로 주간 리포트를 생성합니다.",
                input_data=["ERP 매출원장", "기준주차"],
                output_data=["주간매출리포트", "팀별실적"],
                tags=["재무", "리포트", "정형데이터"],
            ),
        ],
    )
    agent = make_agent([s.id for s in journey.steps])

    generator = MCPGenerator.__new__(MCPGenerator)
    generator._batch_size = 10
    generator._timeout_ms = None
    generator._max_retries = 0
    generator._max_workers = 1
    generator._merge_enabled = False
    generator._dedup_enabled = True
    generator._dedup_min_tasks = 1
    generator._dedup_min_similarity = 0.72
    generator._dedup_loose_similarity = 0.45
    generator._dedup_min_tag_overlap = 0.5

    llm_calls: list[list[str]] = []

    def fake_generate_batch(tasks, task_to_agent):
        llm_calls.append([t.id for t in tasks])
        return {
            tasks[0].id: {
                "name": "generate_sales_report",
                "description": "매출 리포트를 생성합니다",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "q"},
                    },
                    "required": ["query"],
                },
            }
        }

    generator._generate_batch = fake_generate_batch
    tools = generator.generate(journey, [agent])

    assert len(llm_calls) == 1
    assert len(llm_calls[0]) == 1  # 구조 유사성으로 대표 1개만 호출
    assert len(tools) == 2
    assert set(t.name for t in tools) == {"generate_sales_report"}


def test_generate_task_dedup_uses_template_match():
    journey = UserJourney(
        id="j5",
        title="J5",
        steps=[
            TaskStep(
                id="t1",
                name="견적 승인 라우팅",
                description="견적 금액과 정책을 기준으로 승인 경로를 지정합니다.",
                input_data=["견적서", "승인규정"],
                output_data=["승인요청", "승인경로"],
                tags=["영업", "승인", "워크플로우"],
            ),
            TaskStep(
                id="t2",
                name="발주 승인 라우팅",
                description="발주 금액과 정책을 기준으로 승인 경로를 지정합니다.",
                input_data=["발주서", "승인정책"],
                output_data=["승인요청", "승인경로"],
                tags=["구매", "승인", "워크플로우"],
            ),
        ],
    )
    agent = make_agent([s.id for s in journey.steps])

    generator = MCPGenerator.__new__(MCPGenerator)
    generator._batch_size = 10
    generator._timeout_ms = None
    generator._max_retries = 0
    generator._max_workers = 1
    generator._merge_enabled = False
    generator._dedup_enabled = True
    generator._dedup_min_tasks = 1
    generator._dedup_min_similarity = 0.72
    generator._dedup_loose_similarity = 0.45
    generator._dedup_min_tag_overlap = 0.5
    generator._dedup_template_similarity = 0.32

    llm_calls: list[list[str]] = []

    def fake_generate_batch(tasks, task_to_agent):
        llm_calls.append([t.id for t in tasks])
        return {
            tasks[0].id: {
                "name": "route_approval_request",
                "description": "승인 라우팅을 처리합니다",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "q"},
                    },
                    "required": ["query"],
                },
            }
        }

    generator._generate_batch = fake_generate_batch
    tools = generator.generate(journey, [agent])

    assert len(llm_calls) == 1
    assert len(llm_calls[0]) == 1  # 템플릿 매칭으로 대표 1개만 호출
    assert len(tools) == 2
    assert set(t.name for t in tools) == {"route_approval_request"}
