"""
FastAPI 백엔드 - TAD-Mapper 웹 API

신규 엔드포인트:
- POST /api/route              : 쿼리 → Agent 라우팅 (Φ 함수)
- POST /api/compose            : 쿼리 + Agent → Tool 합성 계획
- GET  /api/coverage/{oid}     : 커버리지 메트릭 조회
- POST /api/route-and-compose  : 라우팅 + 합성 한번에
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config.settings import OUTPUT_DIR
from tad_mapper.pipeline import PipelineResult, TADMapperPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TAD-Mapper API",
    description=(
        "AI 적용 기회를 Unit Agent와 MCP Tool로 자동 변환하는 시스템. "
        "수학적 정식화: Q ⊆ ∪Ui, Φ: Q → {Uk}, y = (t_π(m) ∘ ... ∘ t_π(1))(x)"
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 (웹 UI)
WEB_DIR = Path(__file__).parent.parent / "web"
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

# ── 세션 상태 관리 ────────────────────────────────────────────────────────────
# output_id → (TADMapperPipeline, PipelineResult)
_sessions: dict[str, tuple[TADMapperPipeline, PipelineResult]] = {}


# ── 요청/응답 모델 ─────────────────────────────────────────────────────────────

class RouteRequest(BaseModel):
    output_id: str
    query: str


class ComposeRequest(BaseModel):
    output_id: str
    query: str
    agent_id: str


class RouteAndComposeRequest(BaseModel):
    output_id: str
    query: str


# ── 기존 엔드포인트 ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """대시보드 메인 페이지"""
    index_path = WEB_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(..., description="User Journey JSON 또는 CSV 파일"),
    n_agents: int | None = Form(default=None, description="Agent 수 (None=자동 결정)"),
    max_tools_per_agent: int = Form(default=7, description="Agent당 최대 MCP Tool 수"),
):
    """
    User Journey 파일을 분석하여 Agent 구성과 MCP Tool 스키마를 반환합니다.

    수학적 정식화가 적용된 11단계 파이프라인 실행:
    - Q ⊆ ∪Ui 커버리지 분석
    - Φ 라우터 초기화
    - Tool 균형 분석 및 Agile 재분배
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".json", ".csv"}:
        raise HTTPException(status_code=400, detail="JSON 또는 CSV 파일만 지원합니다.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        pipeline = TADMapperPipeline(
            n_agents=n_agents,
            max_tools_per_agent=max_tools_per_agent,
        )
        result = pipeline.run(tmp_path)

        # 세션 저장 (라우팅/합성 API에서 재사용)
        output_id = result.journey.id
        _sessions[output_id] = (pipeline, result)

        out_dir = OUTPUT_DIR / output_id

        # 커버리지 요약
        coverage_summary = None
        if result.coverage_metrics:
            cm = result.coverage_metrics
            coverage_summary = {
                "coverage_ratio": round(cm.coverage_ratio, 4),
                "overlap_ratio": round(cm.overlap_ratio, 4),
                "gap_ratio": round(cm.gap_ratio, 4),
                "coverage_complete": cm.coverage_complete,
                "uncovered_task_count": len(cm.uncovered_task_ids),
                "overlap_agent_pair_count": len(cm.overlap_agent_pairs),
            }

        # 균형 요약
        balance_summary = None
        if result.balance_report:
            br = result.balance_report
            balance_summary = {
                "gini_coefficient": round(br.gini_coefficient, 4),
                "balance_score": round(br.balance_score, 4),
                "overloaded_agents": br.overloaded_agents,
                "underloaded_agents": br.underloaded_agents,
                "rebalanced": br.rebalanced,
                "rebalance_iterations": br.rebalance_iterations,
                "agent_tool_counts": br.agent_tool_counts,
                "summary": br.summary,
            }

        return {
            "status": "success",
            "summary": {
                "journey_title": result.journey.title,
                "total_tasks": len(result.journey.steps),
                "agent_count": len(result.agents),
                "mcp_tool_count": len(result.mcp_tools),
                "hole_count": len(result.holes),
                "overlap_count": len(result.overlaps),
            },
            "coverage": coverage_summary,
            "tool_balance": balance_summary,
            "router_ready": pipeline.router is not None and pipeline.router.is_ready,
            "result": result.report_json,
            "output_id": output_id,
            "files": {
                "report_md": f"/api/output/{output_id}/report.md",
                "result_json": f"/api/output/{output_id}/result.json",
                "mapper_graph": f"/api/output/{output_id}/mapper_graph.html"
                    if (out_dir / "mapper_graph.html").exists() else None,
                "feature_radar": f"/api/output/{output_id}/feature_radar.html"
                    if (out_dir / "feature_radar.html").exists() else None,
                "query_manifold": f"/api/output/{output_id}/query_manifold.html"
                    if (out_dir / "query_manifold.html").exists() else None,
            },
        }
    except Exception as e:
        logger.exception(f"분석 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


# ── 신규 엔드포인트: 라우팅 ───────────────────────────────────────────────────

@app.post("/api/route")
async def route_query(req: RouteRequest):
    """
    Φ(x) = U_k — 사용자 쿼리를 Unit Agent로 라우팅합니다.

    동일한 의도의 다양한 표현을 같은 Agent로 라우팅합니다.
    예) "자료 찾아줘", "데이터 검색해", "정보 줘" → 같은 Agent
    """
    session = _sessions.get(req.output_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"세션 '{req.output_id}'를 찾을 수 없습니다. "
                   "먼저 /api/analyze를 실행하세요."
        )

    pipeline, result = session
    if pipeline.router is None or not pipeline.router.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Homotopy Router가 준비되지 않았습니다. 분석을 다시 실행하세요."
        )

    try:
        routing = pipeline.route_query(req.query)
        return {
            "status": "success",
            "routing": routing.to_dict(),
            "math": {
                "formula": "Φ(x) = U_k ⟺ [x] = [x_k]",
                "homotopy_class": routing.homotopy_class_id,
                "description": (
                    f"쿼리 '{req.query}'는 호모토피 클래스 [{routing.homotopy_class_id}]에 속하며, "
                    f"'{routing.target_agent_name}'으로 라우팅됩니다."
                ),
            },
        }
    except Exception as e:
        logger.exception(f"라우팅 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 신규 엔드포인트: 합성 ─────────────────────────────────────────────────────

@app.post("/api/compose")
async def compose_tools(req: ComposeRequest):
    """
    y = (t_π(m) ∘ ... ∘ t_π(1))(x) — Tool 합성 계획을 생성합니다.

    Agent가 쿼리에 맞게 MCP Tool을 동적으로 조립합니다.
    """
    session = _sessions.get(req.output_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"세션 '{req.output_id}'를 찾을 수 없습니다."
        )

    pipeline, result = session

    try:
        plan = pipeline.compose_tools(req.query, req.agent_id)
        return {
            "status": "success",
            "composition_plan": plan.model_dump(),
            "math": {
                "formula": f"y = (t_π({plan.total_steps}) ∘ ... ∘ t_π(1))(x)",
                "steps": [
                    f"t_π({s.order}) = {s.tool_name}" for s in plan.steps
                ],
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"합성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 신규 엔드포인트: 커버리지 ─────────────────────────────────────────────────

@app.get("/api/coverage/{output_id}")
async def get_coverage(output_id: str):
    """
    Q ⊆ ∪Ui 커버리지 메트릭을 반환합니다.
    """
    session = _sessions.get(output_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"세션 '{output_id}'를 찾을 수 없습니다."
        )

    _, result = session
    if result.coverage_metrics is None:
        raise HTTPException(status_code=404, detail="커버리지 데이터가 없습니다.")

    cm = result.coverage_metrics
    return {
        "output_id": output_id,
        "coverage_metrics": {
            "coverage_ratio": cm.coverage_ratio,
            "overlap_ratio": cm.overlap_ratio,
            "gap_ratio": cm.gap_ratio,
            "coverage_complete": cm.coverage_complete,
            "agent_coverage_areas": cm.agent_coverage_areas,
            "uncovered_task_ids": cm.uncovered_task_ids,
            "overlap_agent_pairs": cm.overlap_agent_pairs,
        },
        "math": {
            "formula": "Q ⊆ ∪ Ui",
            "interpretation": (
                f"전체 쿼리 공간의 {cm.coverage_ratio:.1%}가 Agent들에 의해 커버됩니다. "
                f"{'완전 피복 조건(Q ⊆ ∪Ui) 충족 ✅' if cm.coverage_complete else '미완 ⚠️'}"
            ),
        },
    }


# ── 신규 엔드포인트: 라우팅 + 합성 한번에 ────────────────────────────────────

@app.post("/api/route-and-compose")
async def route_and_compose(req: RouteAndComposeRequest):
    """
    라우팅(Φ) + 합성(∘) 파이프라인을 한번에 실행합니다.

    1. Φ(x) = U_k (쿼리 → Agent 라우팅)
    2. y = (t_π(m) ∘ ... ∘ t_π(1))(x) (Tool 합성 계획)
    """
    session = _sessions.get(req.output_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"세션 '{req.output_id}'를 찾을 수 없습니다."
        )

    pipeline, result = session

    try:
        routing, composition = pipeline.route_and_compose(req.query)
        return {
            "status": "success",
            "query": req.query,
            "routing": routing.to_dict(),
            "composition_plan": composition.model_dump(),
            "math": {
                "step1": f"Φ('{req.query[:30]}') = {routing.target_agent_name}",
                "step2": f"y = (t_π({composition.total_steps}) ∘ ... ∘ t_π(1))(x)",
                "confidence": routing.confidence,
                "is_ambiguous": routing.is_ambiguous,
            },
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception(f"라우팅+합성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 기존 엔드포인트 유지 ──────────────────────────────────────────────────────

@app.get("/api/output/{output_id}/{filename}")
async def get_output_file(output_id: str, filename: str):
    """생성된 결과 파일 다운로드"""
    file_path = OUTPUT_DIR / output_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(str(file_path))


@app.get("/api/sample")
async def get_sample():
    """샘플 User Journey 파일 반환"""
    sample_path = Path(__file__).parent.parent / "data" / "samples" / "trade_journey.json"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="샘플 파일이 없습니다.")
    return FileResponse(
        str(sample_path), media_type="application/json",
        filename="trade_journey.json"
    )


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "0.2.0",
        "features": [
            "query_manifold",
            "homotopy_routing",
            "tool_composition",
            "agile_tool_balancing",
        ],
        "active_sessions": len(_sessions),
    }
