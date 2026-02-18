"""Generate routing evaluation queries from a journey file.

Usage:
  .venv/bin/python -m tad_mapper.eval.generate_queries \
      --input data/samples/ai_opportunities_30.csv \
      --output data/samples/routing_queries.generated.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tad_mapper.input.parser import JourneyParser
from tad_mapper.models.journey import TaskStep


def _queries_for_step(step: TaskStep, variants: int) -> list[dict]:
    templates = [
        f"{step.name} 작업을 처리해줘",
        f"{step.name} 해줘",
    ]
    if step.description:
        desc = step.description.strip()
        short_desc = desc if len(desc) <= 44 else desc[:41] + "..."
        templates.append(f"{short_desc} 관련 요청을 처리해줘")
    if step.tags:
        primary_tag = step.tags[0]
        templates.append(f"{primary_tag} 관점에서 {step.name} 진행해줘")

    rows: list[dict] = []
    used = set()
    for text in templates:
        q = text.strip()
        if not q or q in used:
            continue
        used.add(q)
        rows.append(
            {
                "query": q,
                "label": step.name,
                "expected_task_id": step.id,
            }
        )
        if len(rows) >= max(1, variants):
            break
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Routing 평가용 쿼리셋 생성")
    parser.add_argument("--input", required=True, help="Journey 파일(.json/.csv)")
    parser.add_argument(
        "--output",
        default="",
        help="출력 JSON 파일 경로 (미지정 시 data/samples/<stem>.queries.json)",
    )
    parser.add_argument("--variants", type=int, default=2, help="task당 생성 쿼리 수")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("data/samples") / f"{input_path.stem}.queries.json"

    parser = JourneyParser()
    journey = parser.parse(input_path)

    rows: list[dict] = []
    for step in journey.steps:
        rows.extend(_queries_for_step(step, variants=max(1, args.variants)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Routing Query Set 생성 완료]")
    print(f"- input : {input_path}")
    print(f"- output: {output_path}")
    print(f"- total : {len(rows)}")


if __name__ == "__main__":
    main()
