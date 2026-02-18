"""generate_queries utility tests."""
from __future__ import annotations

from tad_mapper.eval.generate_queries import _queries_for_step
from tad_mapper.models.journey import TaskStep


def test_queries_for_step_respects_variant_limit() -> None:
    step = TaskStep(
        id="task_001",
        name="수출입 통계 분석",
        description="특정 기간 통계를 분석합니다.",
        input_data=["기간"],
        output_data=["리포트"],
        tags=["분석", "통계"],
    )

    rows = _queries_for_step(step, variants=2)

    assert len(rows) == 2
    assert all(r["expected_task_id"] == "task_001" for r in rows)
    assert all(r["label"] == "수출입 통계 분석" for r in rows)
