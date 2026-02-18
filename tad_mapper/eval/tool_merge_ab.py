"""Tool merge A/B evaluator.

Compare MCP shared-tool merge OFF vs ON on identical input.
Focus:
- tool count reduction
- shared-tool coverage
- routing consistency impact
"""
from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tad_mapper.pipeline import TADMapperPipeline


@dataclass
class EvalQuery:
    query: str
    label: str = ""
    expected_task_id: str | None = None


def _safe_avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _parse_float_list(raw: str, default: list[float]) -> list[float]:
    token = raw.strip()
    if not token:
        return default
    vals: list[float] = []
    for part in token.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return sorted(set(vals)) if vals else default


def _parse_int_list(raw: str, default: list[int]) -> list[int]:
    token = raw.strip()
    if not token:
        return default
    vals: list[int] = []
    for part in token.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return sorted(set(vals)) if vals else default


@contextmanager
def _temp_environ(overrides: dict[str, str]):
    original = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ[k] = v
        yield
    finally:
        for k, old in original.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _load_queries(path: Path) -> list[EvalQuery]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("queries JSON은 배열이어야 합니다.")
    queries: list[EvalQuery] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"queries[{idx}]는 객체여야 합니다.")
        q = str(item.get("query", "")).strip()
        if not q:
            raise ValueError(f"queries[{idx}].query가 비어 있습니다.")
        queries.append(
            EvalQuery(
                query=q,
                label=str(item.get("label", "")).strip(),
                expected_task_id=(
                    str(item["expected_task_id"]).strip()
                    if item.get("expected_task_id")
                    else None
                ),
            )
        )
    return queries


def _auto_queries(result: Any, variants: int = 2) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    for step in result.journey.steps:
        templates = [f"{step.name} 해줘", f"{step.name} 작업 처리해줘"]
        if step.description:
            desc = step.description.strip()
            short_desc = desc if len(desc) <= 42 else desc[:39] + "..."
            templates.append(f"{short_desc} 진행해줘")
        used = set()
        generated = 0
        for q in templates:
            if q in used:
                continue
            used.add(q)
            queries.append(EvalQuery(query=q, label=step.name, expected_task_id=step.id))
            generated += 1
            if generated >= max(1, variants):
                break
    return queries


def _tool_stats(result: Any) -> dict[str, Any]:
    tools = result.mcp_tools
    n_tools = len(tools)
    task_sizes: list[int] = []
    shared_count = 0
    for tool in tools:
        ann = tool.annotations
        if ann is None:
            task_sizes.append(0)
            continue
        n_tasks = len(ann.all_task_ids())
        task_sizes.append(n_tasks)
        if n_tasks > 1:
            shared_count += 1
    return {
        "task_count": len(result.journey.steps),
        "agent_count": len(result.agents),
        "tool_count": n_tools,
        "tool_per_task_ratio": (n_tools / len(result.journey.steps)) if result.journey.steps else None,
        "avg_tasks_per_tool": _safe_avg([float(v) for v in task_sizes]) if task_sizes else None,
        "max_tasks_per_tool": max(task_sizes) if task_sizes else 0,
        "shared_tool_count": shared_count,
        "shared_tool_ratio": (shared_count / n_tools) if n_tools > 0 else 0.0,
        "balance_gini": (
            float(result.balance_report.gini_coefficient)
            if result.balance_report is not None else None
        ),
    }


def _routing_snapshot(pipeline: TADMapperPipeline, queries: list[EvalQuery]) -> dict[str, Any]:
    ready = pipeline.router is not None and pipeline.router.is_ready
    if not ready:
        return {"ready": False, "rows": [], "error": pipeline.router_block_reason}

    rows = []
    for q in queries:
        try:
            routing = pipeline.route_query(q.query)
            rows.append(
                {
                    "query": q.query,
                    "target_agent_id": routing.target_agent_id,
                    "homotopy_class_id": routing.homotopy_class_id,
                    "confidence": float(routing.confidence),
                    "is_ambiguous": bool(routing.is_ambiguous),
                }
            )
        except Exception:
            rows.append(
                {
                    "query": q.query,
                    "target_agent_id": "",
                    "homotopy_class_id": "",
                    "confidence": 0.0,
                    "is_ambiguous": True,
                }
            )
    return {"ready": True, "rows": rows, "error": ""}


def _group_map(result: Any) -> dict[str, str]:
    return {
        a.agent_id: (a.routing_group_id or a.agent_id)
        for a in result.agents
    }


def _routing_consistency(
    base_rows: list[dict[str, Any]],
    merge_rows: list[dict[str, Any]],
    base_group_map: dict[str, str],
    merge_group_map: dict[str, str],
) -> dict[str, Any]:
    pairs = list(zip(base_rows, merge_rows))
    if not pairs:
        return {
            "n_queries": 0,
            "exact_agent_consistency": None,
            "group_consistency": None,
            "avg_confidence_delta": None,
            "merge_higher_conf_rate": None,
        }
    exact_hits: list[bool] = []
    group_hits: list[bool] = []
    conf_deltas: list[float] = []
    merge_higher: list[bool] = []
    for b, m in pairs:
        b_agent = b.get("target_agent_id", "")
        m_agent = m.get("target_agent_id", "")
        exact_hits.append(b_agent == m_agent and b_agent != "")

        b_group = base_group_map.get(b_agent, "")
        m_group = merge_group_map.get(m_agent, "")
        group_hits.append(b_group == m_group and b_group != "")

        b_conf = float(b.get("confidence", 0.0))
        m_conf = float(m.get("confidence", 0.0))
        conf_deltas.append(m_conf - b_conf)
        merge_higher.append(m_conf >= b_conf)

    return {
        "n_queries": len(pairs),
        "exact_agent_consistency": sum(exact_hits) / len(exact_hits),
        "group_consistency": sum(group_hits) / len(group_hits),
        "avg_confidence_delta": _safe_avg(conf_deltas),
        "merge_higher_conf_rate": sum(merge_higher) / len(merge_higher),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    no = payload["no_merge"]
    yes = payload["with_merge"]
    c = payload["routing_consistency"]
    variants = payload.get("variants", [])
    md = [
        "# Tool Merge A/B Report",
        "",
        f"- Timestamp: {payload['timestamp']}",
        f"- Input: `{payload['input']}`",
        f"- Queries: `{payload['queries']}`",
        "",
        "## Tool Metrics",
        "",
        "| Mode | Tasks | Tools | Tool/Task | Shared Tools | Avg Tasks/Tool | Max Tasks/Tool | Gini |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| no_merge | {no['task_count']} | {no['tool_count']} | {no['tool_per_task_ratio']} | "
            f"{no['shared_tool_count']} | {no['avg_tasks_per_tool']} | {no['max_tasks_per_tool']} | {no['balance_gini']} |"
        ),
        (
            f"| with_merge | {yes['task_count']} | {yes['tool_count']} | {yes['tool_per_task_ratio']} | "
            f"{yes['shared_tool_count']} | {yes['avg_tasks_per_tool']} | {yes['max_tasks_per_tool']} | {yes['balance_gini']} |"
        ),
        "",
        "## Routing Consistency",
        "",
        f"- n_queries: {c['n_queries']}",
        f"- exact_agent_consistency: {c['exact_agent_consistency']}",
        f"- group_consistency: {c['group_consistency']}",
        f"- avg_confidence_delta (merge - no_merge): {c['avg_confidence_delta']}",
        f"- merge_higher_conf_rate: {c['merge_higher_conf_rate']}",
    ]
    if variants:
        md.extend(
            [
                "",
                "## Merge Variant Leaderboard",
                "",
                "| Similarity | MaxTasks/Tool | Tool Reduction | Group Consistency | Score |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for v in variants:
            md.append(
                f"| {v['merge_similarity']} | {v['max_tasks_per_tool']} | "
                f"{v['tool_reduction_ratio']} | {v['group_consistency']} | {v['score']} |"
            )
    path.write_text("\n".join(md), encoding="utf-8")


def _merge_variant_score(
    no_tool_count: int,
    merge_tool_count: int,
    group_consistency: float | None,
) -> float:
    if no_tool_count <= 0:
        return 0.0
    reduction_ratio = max(0.0, (no_tool_count - merge_tool_count) / no_tool_count)
    consistency = float(group_consistency) if group_consistency is not None else 0.0
    if consistency < 0.90:
        return reduction_ratio * consistency * 0.5
    return reduction_ratio * consistency


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool merge A/B evaluator")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--queries", type=Path, default=None)
    parser.add_argument("--auto-query-variants", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("output/eval"))
    parser.add_argument("--n-agents", type=int, default=None)
    parser.add_argument("--max-tools-per-agent", type=int, default=7)
    parser.add_argument("--merge-similarity", type=float, default=0.55)
    parser.add_argument("--max-tasks-per-tool", type=int, default=4)
    parser.add_argument(
        "--merge-similarities",
        type=str,
        default="",
        help="병합 유사도 스윕(쉼표 구분). 예: 0.45,0.5,0.55",
    )
    parser.add_argument(
        "--max-tasks-options",
        type=str,
        default="",
        help="Tool당 최대 task 수 스윕(쉼표 구분). 예: 2,3,4",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with _temp_environ(
        {
            "TAD_MCP_ENABLE_TOOL_MERGE": "false",
            "TAD_MCP_MERGE_MIN_SIMILARITY": str(args.merge_similarity),
            "TAD_MCP_MAX_TASKS_PER_TOOL": str(args.max_tasks_per_tool),
        }
    ):
        pipeline_no = TADMapperPipeline(
            n_agents=args.n_agents,
            max_tools_per_agent=args.max_tools_per_agent,
        )
        result_no = pipeline_no.run(args.input)

    queries = (
        _load_queries(args.queries)
        if args.queries is not None
        else _auto_queries(result_no, variants=args.auto_query_variants)
    )
    no_snapshot = _routing_snapshot(pipeline_no, queries)
    no_stats = _tool_stats(result_no)

    merge_similarities = _parse_float_list(
        args.merge_similarities,
        default=[args.merge_similarity],
    )
    max_task_options = _parse_int_list(
        args.max_tasks_options,
        default=[args.max_tasks_per_tool],
    )

    variant_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_payload: dict[str, Any] | None = None

    for merge_similarity in merge_similarities:
        for max_tasks_per_tool in max_task_options:
            with _temp_environ(
                {
                    "TAD_MCP_ENABLE_TOOL_MERGE": "true",
                    "TAD_MCP_MERGE_MIN_SIMILARITY": str(merge_similarity),
                    "TAD_MCP_MAX_TASKS_PER_TOOL": str(max_tasks_per_tool),
                }
            ):
                pipeline_yes = TADMapperPipeline(
                    n_agents=args.n_agents,
                    max_tools_per_agent=args.max_tools_per_agent,
                )
                result_yes = pipeline_yes.run(args.input)

            yes_snapshot = _routing_snapshot(pipeline_yes, queries)
            yes_stats = _tool_stats(result_yes)
            consistency = _routing_consistency(
                no_snapshot["rows"],
                yes_snapshot["rows"],
                _group_map(result_no),
                _group_map(result_yes),
            )
            score = _merge_variant_score(
                no_stats["tool_count"],
                yes_stats["tool_count"],
                consistency.get("group_consistency"),
            )
            tool_reduction_ratio = (
                max(0.0, (no_stats["tool_count"] - yes_stats["tool_count"]) / no_stats["tool_count"])
                if no_stats["tool_count"] > 0 else 0.0
            )
            row = {
                "merge_similarity": merge_similarity,
                "max_tasks_per_tool": max_tasks_per_tool,
                "tool_count": yes_stats["tool_count"],
                "tool_reduction_ratio": round(tool_reduction_ratio, 6),
                "group_consistency": consistency.get("group_consistency"),
                "exact_agent_consistency": consistency.get("exact_agent_consistency"),
                "score": round(score, 6),
            }
            variant_rows.append(row)

            candidate_payload = {
                "merge_settings": {
                    "merge_enabled": True,
                    "merge_similarity": merge_similarity,
                    "max_tasks_per_tool": max_tasks_per_tool,
                },
                "with_merge": yes_stats,
                "routing_consistency": consistency,
                "router_ready": {
                    "no_merge": no_snapshot["ready"],
                    "with_merge": yes_snapshot["ready"],
                    "no_merge_error": no_snapshot["error"],
                    "with_merge_error": yes_snapshot["error"],
                },
            }
            if best is None or row["score"] > best["score"]:
                best = row
                best_payload = candidate_payload

    variant_rows.sort(key=lambda r: r["score"], reverse=True)
    if best_payload is None:
        raise RuntimeError("merge variant 평가 결과가 없습니다.")

    payload = {
        "timestamp": datetime.now().isoformat(),
        "input": str(args.input),
        "queries": str(args.queries) if args.queries else "auto",
        "query_count": len(queries),
        "no_merge": no_stats,
        "with_merge": best_payload["with_merge"],
        "merge_settings": best_payload["merge_settings"],
        "routing_consistency": best_payload["routing_consistency"],
        "router_ready": best_payload["router_ready"],
        "variants": variant_rows,
        "best_variant": best,
    }

    json_path = args.output_dir / f"tool_merge_ab_{timestamp}.json"
    md_path = args.output_dir / f"tool_merge_ab_{timestamp}.md"
    _write_report(json_path, payload)
    _write_markdown(md_path, payload)

    print("[Tool Merge A/B 완료]")
    print(f"- JSON: {json_path}")
    print(f"- MD  : {md_path}")
    print(
        f"- tools: {payload['no_merge']['tool_count']} -> {payload['with_merge']['tool_count']}"
    )
    print(
        f"- routing consistency (group): {payload['routing_consistency']['group_consistency']}"
    )
    if payload.get("best_variant"):
        b = payload["best_variant"]
        print(
            f"- best variant: sim={b['merge_similarity']}, "
            f"max_tasks={b['max_tasks_per_tool']}, score={b['score']}"
        )


if __name__ == "__main__":
    main()
