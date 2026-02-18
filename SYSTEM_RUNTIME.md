# TAD-Mapper System Runtime Guide

## 1. 시스템 목적
TAD-Mapper는 사용자 여정(Task 집합)을 입력받아 다음을 자동화합니다.
- Unit Agent 군집 발견 및 역할 구조화
- 쿼리 라우팅(하드/소프트 + 계층형)
- MCP Tool 스키마 생성, 병합, 합성 계획

## 2. 실행 파이프라인
입력(`JSON/CSV`) → Feature Extraction(10D) → TDA/Clustering → Agent Naming →
Hole/Overlap 탐지 → MCP Tool 생성 → Tool Balance(Rebalance) →
Embedding/Query Manifold → Homotopy Router 초기화 → Report/Visualization 출력

핵심 오케스트레이터:
- `tad_mapper/pipeline.py`

## 3. 계층 라우팅 구조
시스템은 두 경로를 지원합니다.
- `Master -> Unit` : 단순 요청
- `Master -> Orchestrator -> Unit` : 복합 요청(서브태스크 분해)

핵심 구현:
- `tad_mapper/engine/hierarchy_planner.py`
- `api/main.py` (`/api/route-hierarchy`, `/api/route-hierarchy-and-compose`)

## 4. MCP Tool 생성/최적화
기본 동작:
1. Task를 chunk로 분할
2. LLM 배치 호출로 Tool schema 생성
3. 실패 시 task 단위 fallback schema로 복구
4. 유사 Tool 병합(shared tool)

성능/안정화 포인트:
- timeout 비활성화 가능(`TAD_MCP_TIMEOUT_MS=0`)
- 병렬 chunk 처리(`TAD_MCP_MAX_WORKERS`)
- 재시도 제한(`TAD_MCP_RETRIES`)
- 사전 dedup(대표 태스크만 LLM 호출)

핵심 구현:
- `tad_mapper/output/mcp_generator.py`

## 5. Dedup/병합 전략
### 5.1 Task Dedup
- strict similarity 기준
- loose similarity + 구조 유사(태그/입출력/의도)
- template match(예: 승인 라우팅, 자동 분류처럼 도메인 명사만 다른 작업)

주요 설정:
- `TAD_MCP_ENABLE_TASK_DEDUP`
- `TAD_MCP_TASK_DEDUP_MIN_SIMILARITY`
- `TAD_MCP_TASK_DEDUP_LOOSE_SIMILARITY`
- `TAD_MCP_TASK_DEDUP_MIN_TAG_OVERLAP`
- `TAD_MCP_TASK_DEDUP_TEMPLATE_SIMILARITY`

### 5.2 Shared Tool Merge
- 동일 Agent 내부 유사 Tool을 하나로 병합
- `max_tasks_per_tool` 한도 내에서만 통합

주요 설정:
- `TAD_MCP_ENABLE_TOOL_MERGE`
- `TAD_MCP_MERGE_MIN_SIMILARITY`
- `TAD_MCP_MAX_TASKS_PER_TOOL`

## 6. 라우팅 품질 보호
외부 임베딩/LLM 장애 시 fallback 벡터로 전환됩니다.
- strict guard: fallback 비율 과다 시 Router 비활성화 가능
- hybrid guard: lexical 보정으로 서비스 연속성 유지

관련 설정:
- `ROUTER_MAX_FALLBACK_RATIO`
- `ROUTER_DISABLE_ON_FALLBACK_RATIO`
- `ROUTE_MIN_CONFIDENCE`
- `ROUTE_MIN_PROB_GAP`

## 7. 운영 시 권장 점검 순서
1. `pytest -q`로 회귀 확인
2. `tool_merge_ab`로 Tool 수 절감/라우팅 일관성 확인
3. `routing_ab`로 strict/hybrid 품질 비교
4. 품질 저하 시 임계값 조정

## 8. 이번 실무형 50 Task 샘플 기준 결과(요약)
- 입력: `data/samples/ai_opportunities_50_company.csv`
- Tool merge: `50 -> 35`
- dedup: `50 -> 47`(대표 태스크 기준)
- 라우팅 일관성(group): `1.0` (hybrid 기준)

해당 수치는 네트워크/임베딩 모델 가용성에 따라 달라질 수 있습니다.
