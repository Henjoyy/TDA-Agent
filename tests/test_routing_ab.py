"""routing_ab evaluator unit tests."""
from __future__ import annotations

import json

from tad_mapper.eval.routing_ab import (
    EvalQuery,
    _load_queries,
    _parse_float_list,
    _safe_ratio,
    _threshold_leaderboard,
)


def test_safe_ratio_handles_none_values() -> None:
    assert _safe_ratio([True, False, None, True]) == 2 / 3
    assert _safe_ratio([None, None]) is None


def test_parse_float_list_uses_default_on_empty() -> None:
    assert _parse_float_list("", [0.1, 0.2]) == [0.1, 0.2]


def test_parse_float_list_deduplicates_and_sorts() -> None:
    assert _parse_float_list("0.3, 0.1,0.3", [0.5]) == [0.1, 0.3]


def test_load_queries_parses_optional_fields(tmp_path) -> None:
    path = tmp_path / "queries.json"
    payload = [
        {
            "query": "재고를 조회해줘",
            "label": "inventory",
            "expected_task_id": "task_001",
            "expected_agent_id": "agent_0",
        },
        {
            "query": "보고서 생성",
        },
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    queries = _load_queries(path)

    assert queries == [
        EvalQuery(
            query="재고를 조회해줘",
            label="inventory",
            expected_task_id="task_001",
            expected_agent_id="agent_0",
        ),
        EvalQuery(
            query="보고서 생성",
            label="",
            expected_task_id=None,
            expected_agent_id=None,
        ),
    ]


def test_threshold_leaderboard_scores_rows() -> None:
    rows = [
        {
            "mode": "hybrid_guard",
            "error": "",
            "confidence": 0.8,
            "prob_gap": 0.3,
            "exact_hit": True,
            "group_hit": True,
            "top3_exact_hit": True,
            "top3_group_hit": True,
        },
        {
            "mode": "hybrid_guard",
            "error": "",
            "confidence": 0.2,
            "prob_gap": 0.01,
            "exact_hit": False,
            "group_hit": True,
            "top3_exact_hit": True,
            "top3_group_hit": True,
        },
    ]

    board = _threshold_leaderboard(rows, conf_thresholds=[0.1, 0.5], gap_thresholds=[0.0, 0.2])

    assert board
    top = sorted(board, key=lambda x: x["score"], reverse=True)[0]
    assert top["mode"] == "hybrid_guard"
    assert top["accepted_count"] >= 1
    assert 0.0 <= top["score"] <= 1.0
