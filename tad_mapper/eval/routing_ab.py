"""A/B routing quality evaluator.

Compare strict router guard mode vs hybrid router mode on identical inputs,
and run confidence/prob-gap threshold sweeps for routing acceptance policy.

Usage:
  .venv/bin/python -m tad_mapper.eval.routing_ab \
      --input data/samples/ai_opportunities_30.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import tad_mapper.pipeline as pipeline_module
from tad_mapper.pipeline import TADMapperPipeline


@dataclass
class EvalQuery:
    query: str
    label: str = ""
    expected_task_id: str | None = None
    expected_agent_id: str | None = None


def _safe_ratio(values: list[bool | None]) -> float | None:
    decided = [v for v in values if v is not None]
    if not decided:
        return None
    return sum(1 for v in decided if v) / len(decided)


def _safe_avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _parse_float_list(raw: str, default: list[float]) -> list[float]:
    if not raw.strip():
        return default
    vals = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        return default
    return sorted(set(vals))


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
                expected_agent_id=(
                    str(item["expected_agent_id"]).strip()
                    if item.get("expected_agent_id")
                    else None
                ),
            )
        )
    return queries


def _auto_queries_from_result(result: Any, variants: int = 2) -> list[EvalQuery]:
    task_to_agent: dict[str, str] = {}
    for agent in result.agents:
        for task_id in agent.task_ids:
            task_to_agent[task_id] = agent.agent_id

    queries: list[EvalQuery] = []
    for step in result.journey.steps:
        templates = [
            f"{step.name} 작업을 처리해줘",
            f"{step.name} 해줘",
        ]
        if step.description:
            desc = step.description.strip()
            short_desc = desc if len(desc) <= 42 else desc[:39] + "..."
            templates.append(f"{short_desc} 관련 요청을 처리해줘")
        if step.tags:
            tag = step.tags[0]
            templates.append(f"{tag} 관점에서 {step.name} 진행해줘")

        used = set()
        generated = 0
        for q in templates:
            q = q.strip()
            if not q or q in used:
                continue
            used.add(q)
            queries.append(
                EvalQuery(
                    query=q,
                    label=step.name,
                    expected_task_id=step.id,
                    expected_agent_id=task_to_agent.get(step.id),
                )
            )
            generated += 1
            if generated >= max(1, variants):
                break

    return queries


def _normalize_queries(
    queries: list[EvalQuery],
    task_to_agent: dict[str, str],
) -> list[EvalQuery]:
    normalized: list[EvalQuery] = []
    for q in queries:
        expected_agent_id = q.expected_agent_id
        if not expected_agent_id and q.expected_task_id:
            expected_agent_id = task_to_agent.get(q.expected_task_id)
        normalized.append(
            EvalQuery(
                query=q.query,
                label=q.label,
                expected_task_id=q.expected_task_id,
                expected_agent_id=expected_agent_id,
            )
        )
    return normalized


def _evaluate_mode(
    mode_name: str,
    disable_on_fallback_ratio: bool,
    input_path: Path,
    queries: list[EvalQuery] | None,
    n_agents: int | None,
    max_tools_per_agent: int,
    auto_query_variants: int,
) -> dict[str, Any]:
    original_flag = pipeline_module.ROUTER_DISABLE_ON_FALLBACK_RATIO
    pipeline_module.ROUTER_DISABLE_ON_FALLBACK_RATIO = disable_on_fallback_ratio
    try:
        pipeline = TADMapperPipeline(
            n_agents=n_agents,
            max_tools_per_agent=max_tools_per_agent,
        )
        result = pipeline.run(input_path)
    finally:
        pipeline_module.ROUTER_DISABLE_ON_FALLBACK_RATIO = original_flag

    task_to_agent: dict[str, str] = {}
    agent_to_group: dict[str, str] = {}
    for agent in result.agents:
        for task_id in agent.task_ids:
            task_to_agent[task_id] = agent.agent_id
        agent_to_group[agent.agent_id] = agent.routing_group_id or agent.agent_id

    eval_queries = queries or _auto_queries_from_result(
        result, variants=auto_query_variants
    )
    eval_queries = _normalize_queries(eval_queries, task_to_agent)

    router_ready = pipeline.router is not None and pipeline.router.is_ready

    records: list[dict[str, Any]] = []
    exact_hits: list[bool | None] = []
    group_hits: list[bool | None] = []
    top3_exact_hits: list[bool | None] = []
    top3_group_hits: list[bool | None] = []
    ambiguous_flags: list[bool] = []
    confidences: list[float] = []
    gaps: list[float] = []
    route_errors = 0

    for q in eval_queries:
        record: dict[str, Any] = {
            "mode": mode_name,
            "query": q.query,
            "label": q.label,
            "expected_task_id": q.expected_task_id,
            "expected_agent_id": q.expected_agent_id,
            "predicted_agent_id": "",
            "predicted_group_id": "",
            "confidence": "",
            "is_ambiguous": "",
            "prob_gap": "",
            "top3_agents": "",
            "exact_hit": "",
            "group_hit": "",
            "top3_exact_hit": "",
            "top3_group_hit": "",
            "error": "",
        }

        if not router_ready:
            record["error"] = pipeline.router_block_reason or "router_not_ready"
            records.append(record)
            exact_hits.append(None)
            group_hits.append(None)
            top3_exact_hits.append(None)
            top3_group_hits.append(None)
            continue

        try:
            routing = pipeline.route_query(q.query)
            sorted_probs = sorted(
                routing.routing_probabilities.items(), key=lambda x: x[1], reverse=True
            )
            top3_ids = [aid for aid, _ in sorted_probs[:3]]
            top1_prob = float(sorted_probs[0][1]) if sorted_probs else 0.0
            top2_prob = float(sorted_probs[1][1]) if len(sorted_probs) > 1 else 0.0
            prob_gap = max(0.0, top1_prob - top2_prob)
            predicted_agent = routing.target_agent_id
            predicted_group = agent_to_group.get(predicted_agent, "")

            expected_agent = q.expected_agent_id
            expected_group = agent_to_group.get(expected_agent, "") if expected_agent else ""

            exact_hit = (predicted_agent == expected_agent) if expected_agent else None
            group_hit = (predicted_group == expected_group) if expected_group else None
            top3_exact_hit = (expected_agent in top3_ids) if expected_agent else None
            top3_group_hit = (
                any(agent_to_group.get(aid, "") == expected_group for aid in top3_ids)
                if expected_group
                else None
            )

            record.update(
                {
                    "predicted_agent_id": predicted_agent,
                    "predicted_group_id": predicted_group,
                    "confidence": round(float(routing.confidence), 6),
                    "is_ambiguous": bool(routing.is_ambiguous),
                    "prob_gap": round(prob_gap, 6),
                    "top3_agents": "|".join(top3_ids),
                    "exact_hit": exact_hit,
                    "group_hit": group_hit,
                    "top3_exact_hit": top3_exact_hit,
                    "top3_group_hit": top3_group_hit,
                }
            )

            exact_hits.append(exact_hit)
            group_hits.append(group_hit)
            top3_exact_hits.append(top3_exact_hit)
            top3_group_hits.append(top3_group_hit)
            ambiguous_flags.append(bool(routing.is_ambiguous))
            confidences.append(float(routing.confidence))
            gaps.append(prob_gap)

        except Exception as e:  # noqa: BLE001
            route_errors += 1
            record["error"] = str(e)
            exact_hits.append(None)
            group_hits.append(None)
            top3_exact_hits.append(None)
            top3_group_hits.append(None)

        records.append(record)

    summary = {
        "mode": mode_name,
        "router_ready": router_ready,
        "router_block_reason": pipeline.router_block_reason,
        "embedding_health": result.embedding_health,
        "n_agents": len(result.agents),
        "n_queries": len(eval_queries),
        "route_errors": route_errors,
        "avg_confidence": _safe_avg(confidences),
        "avg_prob_gap": _safe_avg(gaps),
        "ambiguous_rate": _safe_ratio(ambiguous_flags),
        "exact_agent_accuracy": _safe_ratio(exact_hits),
        "group_accuracy": _safe_ratio(group_hits),
        "top3_exact_hit_rate": _safe_ratio(top3_exact_hits),
        "top3_group_hit_rate": _safe_ratio(top3_group_hits),
    }

    return {
        "summary": summary,
        "records": records,
    }


def _threshold_leaderboard(
    rows: list[dict[str, Any]],
    conf_thresholds: list[float],
    gap_thresholds: list[float],
) -> list[dict[str, Any]]:
    leaderboard: list[dict[str, Any]] = []
    modes = sorted({str(r.get("mode", "")) for r in rows})

    for mode in modes:
        mode_rows = [r for r in rows if r.get("mode") == mode]
        valid_rows = [
            r for r in mode_rows
            if r.get("error") in {"", None}
            and isinstance(r.get("confidence"), (int, float))
            and isinstance(r.get("prob_gap"), (int, float))
        ]
        valid_count = len(valid_rows)

        for conf_thr in conf_thresholds:
            for gap_thr in gap_thresholds:
                accepted = [
                    r for r in valid_rows
                    if float(r["confidence"]) >= conf_thr
                    and float(r["prob_gap"]) >= gap_thr
                ]
                accepted_count = len(accepted)
                accepted_ratio = (accepted_count / valid_count) if valid_count > 0 else 0.0

                exact_vals = [r.get("exact_hit") for r in accepted]
                group_vals = [r.get("group_hit") for r in accepted]
                top3_exact_vals = [r.get("top3_exact_hit") for r in accepted]
                top3_group_vals = [r.get("top3_group_hit") for r in accepted]

                exact_acc = _safe_ratio(exact_vals)
                group_acc = _safe_ratio(group_vals)
                top3_exact = _safe_ratio(top3_exact_vals)
                top3_group = _safe_ratio(top3_group_vals)

                score = (group_acc or 0.0) * accepted_ratio
                leaderboard.append(
                    {
                        "mode": mode,
                        "confidence_threshold": conf_thr,
                        "gap_threshold": gap_thr,
                        "valid_count": valid_count,
                        "accepted_count": accepted_count,
                        "accepted_ratio": round(accepted_ratio, 6),
                        "accepted_exact_accuracy": exact_acc,
                        "accepted_group_accuracy": group_acc,
                        "accepted_top3_exact_hit_rate": top3_exact,
                        "accepted_top3_group_hit_rate": top3_group,
                        "score": round(score, 6),
                    }
                )

    leaderboard.sort(
        key=lambda x: (x["mode"], x["score"], x["accepted_group_accuracy"] or 0.0),
        reverse=True,
    )
    return leaderboard


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "query",
        "label",
        "expected_task_id",
        "expected_agent_id",
        "predicted_agent_id",
        "predicted_group_id",
        "confidence",
        "is_ambiguous",
        "prob_gap",
        "top3_agents",
        "exact_hit",
        "group_hit",
        "top3_exact_hit",
        "top3_group_hit",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(
    report: dict[str, Any],
    leaderboard: list[dict[str, Any]],
    path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Routing A/B Report")
    lines.append("")
    lines.append(f"- Timestamp: {report['timestamp']}")
    lines.append(f"- Input: `{report['input']}`")
    lines.append(f"- Queries: `{report['queries']}`")
    lines.append("")

    lines.append("## Mode Summary")
    lines.append("")
    lines.append("| Mode | Router Ready | Exact | Group | Top3 Exact | Top3 Group | Ambiguous | Avg Conf | Avg Gap |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in report["summaries"]:
        lines.append(
            "| {mode} | {ready} | {exact} | {group} | {top3_exact} | {top3_group} | {ambig} | {conf} | {gap} |".format(
                mode=s["mode"],
                ready=s["router_ready"],
                exact=s.get("exact_agent_accuracy"),
                group=s.get("group_accuracy"),
                top3_exact=s.get("top3_exact_hit_rate"),
                top3_group=s.get("top3_group_hit_rate"),
                ambig=s.get("ambiguous_rate"),
                conf=s.get("avg_confidence"),
                gap=s.get("avg_prob_gap"),
            )
        )
    lines.append("")

    lines.append("## Threshold Leaderboard")
    lines.append("")
    lines.append("| Mode | Conf Thr | Gap Thr | Accepted | Ratio | Exact(accepted) | Group(accepted) | Score |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in leaderboard:
        by_mode.setdefault(row["mode"], []).append(row)

    for mode, rows in by_mode.items():
        top_rows = sorted(rows, key=lambda x: x["score"], reverse=True)[:10]
        for r in top_rows:
            lines.append(
                "| {mode} | {ct} | {gt} | {ac}/{vc} | {ar} | {ea} | {ga} | {sc} |".format(
                    mode=mode,
                    ct=r["confidence_threshold"],
                    gt=r["gap_threshold"],
                    ac=r["accepted_count"],
                    vc=r["valid_count"],
                    ar=r["accepted_ratio"],
                    ea=r["accepted_exact_accuracy"],
                    ga=r["accepted_group_accuracy"],
                    sc=r["score"],
                )
            )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Routing 품질 A/B 평가")
    parser.add_argument("--input", required=True, help="Journey 입력 파일(.json/.csv)")
    parser.add_argument(
        "--queries",
        default="",
        help=(
            "선택: 쿼리셋 JSON 파일 경로. "
            "미지정 시 각 task 기반 자동 쿼리 생성"
        ),
    )
    parser.add_argument(
        "--auto-query-variants",
        type=int,
        default=2,
        help="자동 생성 시 task당 쿼리 수",
    )
    parser.add_argument("--output-dir", default="output/eval", help="리포트 출력 디렉터리")
    parser.add_argument("--n-agents", type=int, default=None, help="고정 Agent 수(옵션)")
    parser.add_argument("--max-tools-per-agent", type=int, default=7)
    parser.add_argument(
        "--conf-thresholds",
        default="0.15,0.25,0.35,0.45",
        help="confidence threshold 후보 리스트(콤마 구분)",
    )
    parser.add_argument(
        "--gap-thresholds",
        default="0.0,0.05,0.1,0.15",
        help="top1-top2 확률 gap threshold 후보 리스트(콤마 구분)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    query_path = Path(args.queries) if args.queries else None
    queries = _load_queries(query_path) if query_path else None

    conf_thresholds = _parse_float_list(
        args.conf_thresholds,
        default=[0.15, 0.25, 0.35, 0.45],
    )
    gap_thresholds = _parse_float_list(
        args.gap_thresholds,
        default=[0.0, 0.05, 0.1, 0.15],
    )

    modes = [
        ("strict_guard", True),
        ("hybrid_guard", False),
    ]

    mode_results: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for mode_name, disable_flag in modes:
        result = _evaluate_mode(
            mode_name=mode_name,
            disable_on_fallback_ratio=disable_flag,
            input_path=input_path,
            queries=queries,
            n_agents=args.n_agents,
            max_tools_per_agent=args.max_tools_per_agent,
            auto_query_variants=max(1, args.auto_query_variants),
        )
        mode_results.append(result["summary"])
        all_rows.extend(result["records"])

    threshold_leaderboard = _threshold_leaderboard(
        all_rows, conf_thresholds=conf_thresholds, gap_thresholds=gap_thresholds
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"routing_ab_{ts}.json"
    csv_path = out_dir / f"routing_ab_{ts}.csv"
    md_path = out_dir / f"routing_ab_{ts}.md"

    report = {
        "timestamp": datetime.now().isoformat(),
        "input": str(input_path),
        "queries": str(query_path) if query_path else f"auto_from_tasks(variants={max(1, args.auto_query_variants)})",
        "summaries": mode_results,
        "thresholds": {
            "confidence": conf_thresholds,
            "gap": gap_thresholds,
        },
        "threshold_leaderboard": threshold_leaderboard,
        "json_report": str(json_path),
        "csv_report": str(csv_path),
        "markdown_report": str(md_path),
    }

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(all_rows, csv_path)
    _write_markdown(report, threshold_leaderboard, md_path)

    print("[Routing A/B 완료]")
    print(f"- JSON: {json_path}")
    print(f"- CSV : {csv_path}")
    print(f"- MD  : {md_path}")
    for summary in mode_results:
        print(
            f"- {summary['mode']}: ready={summary['router_ready']}, "
            f"exact={summary['exact_agent_accuracy']}, group={summary['group_accuracy']}, "
            f"ambiguous={summary['ambiguous_rate']}, avg_conf={summary['avg_confidence']}"
        )


if __name__ == "__main__":
    main()
