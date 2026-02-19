# TAD-Mapper 🗺️

**AI 적용 기회(User Journey)를 입력하면 위상수학적 알고리즘으로 Unit Agent와 MCP Tool을 자동 설계하는 시스템**

> **v0.2.5** — 수학적 엔진 고도화: 10D 특징 벡터, Weighted 클러스터링, Softmax 기반 확률 라우팅

---

#### Mathematical Formulation of TAD System

### 1. 사용자 의도 공간과 에이전트 커버리지 (The User Manifold & Agent Covering)
사용자의 모든 가능한 발화(Query)와 의도(Intent)를 하나의 거대한 위상 공간(Topological Space) $\mathcal{Q}$라고 정의합니다. 우리가 만든 6개의 Unit Agent는 이 공간을 빈틈없이 덮는 **'열린 피복(Open Cover)'**입니다.

$$
\mathcal{Q} \subseteq \bigcup_{i \in \text{Agents}} U_i
$$

* **$\mathcal{Q}$**: 사용자 쿼리 매니폴드 (User Query Manifold)
* **$U_i$**: $i$번째 Unit Agent가 커버할 수 있는 기능의 영역 (e.g., $U_{\text{Search}}, U_{\text{Stats}}, \dots$)
* **의미**: 사용자의 어떤 질문($q \in \mathcal{Q}$)이 들어와도, 최소한 하나의 Agent($U_i$)는 이를 처리할 수 있어야 함 (빈틈 없음).

---

### 2. 호모토피 라우팅 함수 (Homotopy Routing Function)
Master Agent가 사용자의 말을 알아듣고 적절한 Agent에게 넘겨주는 과정을 **'호모토피 클래스 분류(Classification)'**로 정의합니다. 사용자의 입력 $x$와 기준이 되는 의도 $x_0$가 서로 호모토픽($x \simeq x_0$)하다면, 같은 Agent로 라우팅됩니다.

$$
\Phi : \mathcal{Q} \to \{U_{\text{Search}}, U_{\text{Stats}}, \dots, U_{\text{Report}}\}
$$
$$
\Phi(x) = U_k \iff [x] = [x_k]
$$

* **$\Phi$**: Master Agent의 라우팅 함수
* **$[x]$**: 입력 $x$의 호모토피 클래스 (표현은 다르지만 본질적 의도가 같은 쿼리들의 집합)
* **의미**: 표현($x$)은 다르지만 본질적 의도($[x]$)가 같다면 동일한 에이전트로 매핑됨.

---

### 3. MCP 툴 실행과 합성 (MCP Tool Composition)
선택된 Agent($U_k$)가 내부적으로 여러 MCP Tool을 골러서 실행하는 것을 **'함수의 합성(Composition of Functions)'**으로 표현합니다. 

$$
y = (t_{\pi(m)} \circ \dots \circ t_{\pi(2)} \circ t_{\pi(1)})(x)
$$

* **$t \in \mathcal{T}_k$**: 개별 MCP Tool (Function)
* **$\circ$**: 함수의 합성 (실행 순서)
* **$\pi$**: 문맥에 따라 동적으로 결정된 실행 순서(Sequence)

---

### 4. 시스템의 강건성 (Robustness / Stability Condition)
입력 $x$에 작은 변형(노이즈, 오타 등) $\epsilon$이 가해져도, 라우팅 결과는 변하지 않아야 함을 증명하는 **위상학적 불변성(Topological Invariance)** 수식입니다.

$$
\text{If } \| x - x' \| < \delta, \text{ then } \Phi(x) = \Phi(x')
$$

* **$\| x - x' \|$**: 사용자 발화 간의 거리
* **$\delta$**: Agent가 허용하는 변형의 임계치 (Tolerance)

## 시스템 동작 문서

- 상세 동작/구성/장애대응 흐름: `/Users/hahyeonjong/TAD-Agent Mapping/SYSTEM_RUNTIME.md`

---

## 빠른 시작

### 1. 환경 설정
```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

### 2. 서버 실행
```bash
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

### 3. 브라우저 접속
```
http://localhost:8000
```

### 4. CLI로 직접 실행
```bash
source .venv/bin/activate
python -m tad_mapper.pipeline data/samples/trade_journey.json
```

### 5. 테스트 실행
```bash
.venv/bin/python -m pytest tests/ -v
```

---

## 파이프라인 구조 (v0.2.0)

```
User Journey (JSON/CSV)
        ↓
[1]  Feature Extraction    ← Gemini LLM → 10D 위상학적 벡터
        ↓
[2]  TDA Mapper            ← scikit-learn Mapper 알고리즘
        ↓
[3]  Agent Discovery       ← HDBSCAN (자동 클러스터 수 결정, 비구형 클러스터)
        ↓
[4]  Agent Naming          ← Gemini 3.0 Flash + Skills 가이드라인 → 이름/역할 자동 부여
        ↓
[5]  Hole Detection        ← 논리적 구멍 & 중복 탐지
        ↓
[6]  MCP Tool Generation   ← Gemini LLM → JSON 스키마 자동 생성
        ↓
[7]  Tool Balance ★        ← Gini 계수 기반 오버로드 탐지 + Agile 재분배
        ↓
[8]  Report + Dashboard    ← Markdown/JSON + 웹 시각화
        ↓
[9]  Task Embeddings ★     ← Gemini text-embedding-004 (768D)
        ↓
[10] Query Manifold ★      ← Q ⊆ ∪Ui 커버리지 분석 (임베딩 반경 기반 + Voronoi 면적 근사)
        ↓
[11] Homotopy Router ★     ← Φ 라우터 초기화 (코사인 유사도 기반)

★ v0.2.0 신규
```

---

## 신규 기능 (v0.2.0)

### 수학적 엔진 고도화 (v0.2.5)

- **10D 위상 특징 벡터**: 기존 6D에 `temporal_sensitivity`, `data_volume`, `security_level`, `state_dependency`를 추가해 미세한 태스크 차이를 반영합니다.
- **Weighted Euclidean 클러스터링**: `reasoning_depth`, `domain_specificity`를 1.5x 가중해 Agent 정체성을 더 강하게 반영합니다.
- **적응형 라우팅 반경**: Agent별 태스크 분산(평균 거리 + 표준편차 + margin)으로 반경을 동적으로 계산합니다.
- **확률 라우팅(Φ_soft)**: `route_soft()`와 `routing_probabilities`로 다중 Agent 후보 확률을 제공합니다.
- **Semantic Group 라우팅**: Tool 밸런싱으로 split된 Agent를 `routing_group_id`로 다시 묶어 의미 축을 보존합니다.
- **Hybrid 유사도**: 임베딩 유사도 + lexical 토큰 유사도를 결합해 fallback 환경에서도 라우팅 품질을 보정합니다.
- **Fallback-aware 멤버 선택**: 그룹 내부 라우팅에서 query coverage 중심 lexical 점수 + hub prior를 적용해 split agent 오탐을 줄입니다.
- **계층형 라우팅 계획**: `Master → (Orchestrator) → Unit` 경로를 지원하며, 단순 쿼리는 `Master → Unit`으로 자동 단축합니다.

### Query Manifold & 커버리지 분석

태스크 임베딩을 기반으로 각 Agent의 커버리지 반경을 계산하고, 태스크 단위 커버/중첩/갭을 측정합니다.

- `coverage_ratio` : 하나 이상의 Agent 영역에 포함된 태스크 비율
- `overlap_ratio`  : 두 개 이상 Agent 영역에 동시에 포함된 태스크 비율
- `gap_ratio`      : 어떤 Agent 영역에도 포함되지 않은 태스크 비율
- `coverage_complete` : Q ⊆ ∪Ui 조건 충족 여부

### 안정화 리팩토링 (v0.2.2)

- **Query Manifold 정확도 개선**: 커버리지/갭 계산을 임베딩 반경 기반으로 통일하고, uncovered task 검출 버그를 수정했습니다.
- **Tool 합성 정렬 수정**: 의존성 그래프의 방향성과 위상 정렬 진입차수 계산을 바로잡아 실행 순서 안정성을 개선했습니다.
- **Agent ID 안정화**: LLM 명명 단계에서 내부 `agent_id`를 변경하지 않도록 수정하고, `tool_prefix`를 별도 필드로 분리했습니다.
- **임베딩 실패 fallback 개선**: 랜덤 벡터 대신 텍스트 해시 기반 결정적 벡터를 사용해 재현성을 확보했습니다.
- **API 세션 관리 개선**: `/api/analyze` 결과 세션에 TTL(1시간) + LRU(최대 32개) 정책을 적용했습니다.

### MCP Tool 생성 무한 대기 방지 (v0.2.3)

- **LLM 요청 타임아웃**: MCP 스키마 생성 API 호출에 밀리초 단위 타임아웃을 적용했습니다.
- **배치 분할 처리**: 많은 태스크를 한 번에 생성하지 않고 chunk 단위로 분할해 지연/실패 확률을 낮췄습니다.
- **재시도 횟수 제한**: 실패 시 제한된 횟수만 재시도하고, 초과 시 즉시 fallback 스키마로 복구합니다.
- **프런트 분석 요청 타임아웃**: 웹 대시보드 분석 요청도 120초 제한을 적용해 무한 로딩 상태를 차단합니다.
- **공유 Tool 병합**: 유사한 태스크는 Agent 내부에서 하나의 공유 MCP Tool로 병합해 Tool 수를 줄입니다.

### 라우팅 신뢰성 강화 (v0.2.4)

- **임베딩 모델 자동 전환**: 기본 임베딩 모델이 미지원(404)일 때 후보 모델로 자동 전환합니다.
- **Router 보호 가드**: 임베딩 fallback 비율이 높을 때 기본은 hybrid 모드로 계속 라우팅하고, 필요 시 비활성화할 수 있습니다.
- **Low-confidence 차단**: 신뢰도/모호성 기준 미달 시 `/api/route`는 409를 반환해 오탐 라우팅을 막습니다.

### HDBSCAN 기반 Agent 발견 (v0.2.1)

기존 KMeans의 한계(구형 클러스터, k 지정 필요)를 극복하기 위해 HDBSCAN을 도입했습니다.
- **자동 클러스터 수 결정**: 데이터 밀도에 따라 최적의 Agent 수($|Agents|$)를 자동으로 찾습니다.
- **노이즈 처리**: 어떤 Agent에도 속하지 않는 이상치(Task)를 식별하고, 가장 가까운 Agent에 재배정합니다.
- **God Agent 방지**: 태스크가 너무 많은 Agent는 자동으로 분할(`refine_clusters`)하여 부하를 분산합니다.

### 호모토피 라우팅 (실시간 쿼리 라우팅)

분석 완료 후 새로운 사용자 쿼리를 실시간으로 적절한 Agent로 라우팅합니다.

```python
pipeline = TADMapperPipeline()
result = pipeline.run("data/samples/trade_journey.json")

# 실시간 라우팅
routing = pipeline.route_query("수출 통계 조회해줘")
print(routing.target_agent_name)   # 예: "무역통계 분석 에이전트"
print(routing.confidence)          # 예: 0.82
print(routing.is_ambiguous)        # False
print(routing.routing_probabilities)  # 예: {"agent_1": 0.71, "agent_0": 0.18, ...}
```

### MCP Tool 합성 계획

라우팅된 Agent의 MCP Tool을 어떤 순서로 조합할지 동적으로 결정합니다.

```python
# 라우팅 + 합성 한번에
routing, plan = pipeline.route_and_compose("수출 통계 조회해줘")
for step in plan.steps:
    print(f"t_π({step.order}) = {step.tool_name}()")
    # t_π(1) = get_export_stats()
    # t_π(2) = analyze_trade_trend()
    # t_π(3) = generate_report()
```

### Agile Tool 균형 분배

Agent당 MCP Tool이 너무 많으면 LLM 성능이 저하됩니다. 자동으로 감지하고 재분배합니다.

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `max_tools_per_agent` | `7` | Agent당 최대 MCP Tool 수 |
| Gini 계수 > 0.3 | 경고 | 불균형 감지 임계값 |
| 최대 반복 | `3회` | Agile 재분배 수렴 횟수 |

```python
pipeline = TADMapperPipeline(max_tools_per_agent=5)  # 임계값 조정
```

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/api/analyze` | Journey 파일 분석 (전체 파이프라인) |
| `POST` | `/api/route` | 쿼리 → Agent 라우팅 (Φ 함수) |
| `POST` | `/api/compose` | 쿼리 + Agent → Tool 합성 계획 |
| `GET`  | `/api/coverage/{output_id}` | 커버리지 메트릭 조회 |
| `POST` | `/api/route-and-compose` | 라우팅 + 합성 한번에 |
| `GET`  | `/api/output/{id}/{file}` | 결과 파일 다운로드 |
| `GET`  | `/api/sample` | 샘플 Journey 파일 |
| `GET`  | `/api/health` | 상태 확인 |

세션 정책:
- 분석 세션은 마지막 접근 시점 기준 1시간 유지됩니다.
- 서버는 최대 32개 세션만 유지하며, 초과 시 오래된 세션부터 제거됩니다.

### 라우팅 예시

```bash
# 분석 먼저 실행
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@data/samples/trade_journey.json"
# → output_id: "trade_journey"

# 쿼리 라우팅
curl -X POST http://localhost:8000/api/route \
  -H "Content-Type: application/json" \
  -d '{"output_id": "trade_journey", "query": "수출 통계 조회해줘"}'

# 라우팅 + 합성 한번에
curl -X POST http://localhost:8000/api/route-and-compose \
  -H "Content-Type: application/json" \
  -d '{"output_id": "trade_journey", "query": "수출 통계 분석해줘"}'
```

---

## 프로젝트 구조

```
TAD-Agent Mapping/
├── tad_mapper/
│   ├── engine/
│   │   ├── feature_extractor.py   # 10D 위상 특징 벡터 추출 (Gemini)
│   │   ├── tda_analyzer.py        # Mapper + HDBSCAN + Weighted 클러스터링
│   │   ├── embedder.py            # 텍스트 임베딩 (768D) ★확장
│   │   ├── query_manifold.py      # Query Manifold Q ⊆ ∪Ui (커버리지 계산 안정화)
│   │   ├── homotopy_router.py     # 호모토피 라우팅 함수 Φ ★신규
│   │   ├── tool_composer.py       # MCP Tool 합성 엔진 (위상 정렬 안정화)
│   │   ├── tool_balancer.py       # Agile Tool 균형 분배 ★신규
│   │   └── visualizer.py          # Plotly 시각화 ★확장
│   ├── mapper/
│   │   ├── agent_namer.py         # Agent 명명 (Gemini)
│   │   └── hole_detector.py       # 논리적 구멍 탐지
│   ├── models/
│   │   ├── journey.py             # UserJourney, TaskStep
│   │   ├── agent.py               # Agent 매핑 결과
│   │   ├── mcp_tool.py            # MCP Tool 스키마
│   │   └── topology.py            # 수학적 정식화 모델 ★신규
│   ├── output/
│   │   ├── mcp_generator.py       # MCP Tool 스키마 생성
│   │   └── report_generator.py    # 리포트 생성
│   ├── input/
│   │   └── parser.py              # JSON/CSV 파서
│   └── pipeline.py                # 메인 오케스트레이터 ★확장
├── api/
│   └── main.py                    # FastAPI 서버 ★확장
├── web/
│   ├── index.html                 # 웹 대시보드 ★확장
│   ├── index.js                   # 클라이언트 로직 ★확장
│   └── index.css                  # 스타일 ★확장
├── tests/
│   ├── test_tool_balancer.py      # ToolBalancer 테스트
│   ├── test_homotopy_router.py    # HomotopyRouter 테스트
│   ├── test_query_manifold.py     # QueryManifold 테스트
│   ├── test_tool_composer.py      # ToolComposer 정렬/그래프 테스트 ★신규
│   ├── test_mcp_generator.py      # MCPGenerator 타임아웃/배치 테스트 ★신규
│   ├── test_agent_namer.py        # AgentNamer ID 안정성 테스트 ★신규
│   ├── test_embedder.py           # Embedder fallback 재현성 테스트 ★신규
│   └── test_api_sessions.py       # API 세션 TTL/LRU 테스트 ★신규
├── config/
│   ├── settings.py                # 전역 설정
│   └── unit_agents.yaml           # Agent 템플릿
└── data/samples/
    ├── trade_journey.json          # 8-태스크 샘플
    └── ai_opportunities_30.csv     # 30-태스크 샘플
```

---

## 출력 파일

분석 완료 시 `output/{journey_id}/` 에 저장됩니다.

| 파일 | 설명 |
|------|------|
| `report.md` | 분석 결과 Markdown 리포트 |
| `result.json` | 구조화된 JSON 결과 (10D `feature_space.task_features` 포함) |
| `mapper_graph.html` | TDA Mapper 위상 그래프 (Plotly) |
| `feature_radar.html` | Agent 특징 프로파일 레이더 차트 |
| `query_manifold.html` | Query Manifold Q ⊆ ∪Ui 시각화 ★신규 |

---

## 입력 형식

### JSON
```json
{
  "id": "my_journey",
  "title": "서비스 이름",
  "domain": "무역",
  "steps": [
    {
      "id": "task_001",
      "name": "태스크 이름",
      "description": "상세 설명",
      "input_data": ["입력1"],
      "output_data": ["출력1"],
      "dependencies": []
    }
  ]
}
```

### CSV
```csv
id,name,description,actor,input_data,output_data,dependencies,tags
task_001,태스크명,설명,user,입력1;입력2,출력1,,태그1;태그2
```

---

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `GEMINI_API_KEY` | **필수** | Gemini API 키 (.gitignore 처리 필수) |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | LLM 모델 (Gemini 3.0 Flash) |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | 임베딩 모델 (768D) |
| `EMBEDDING_MODEL_CANDIDATES` | `gemini-embedding-001,models/text-embedding-004,text-embedding-004` | 임베딩 모델 자동 전환 후보 |
| `TDA_N_INTERVALS` | `10` | Mapper 구간 수 |
| `TDA_OVERLAP_FRAC` | `0.3` | 구간 오버랩 비율 |
| `TAD_MCP_TIMEOUT_MS` | `0` | MCP Tool 생성 LLM 요청 타임아웃(ms), `0`/음수=무제한 |
| `TAD_MCP_BATCH_SIZE` | `10` | MCP Tool 생성 chunk 크기 |
| `TAD_MCP_RETRIES` | `0` | MCP Tool 생성 실패 시 재시도 횟수 |
| `TAD_MCP_MAX_WORKERS` | `4` | MCP Tool chunk 병렬 처리 워커 수 |
| `TAD_MCP_ENABLE_TOOL_MERGE` | `true` | 유사 태스크 Tool을 공유 Tool로 병합할지 여부 |
| `TAD_MCP_MERGE_MIN_SIMILARITY` | `0.55` | 공유 Tool 병합 최소 유사도 |
| `TAD_MCP_MAX_TASKS_PER_TOOL` | `4` | 하나의 공유 Tool이 커버할 최대 태스크 수 |
| `TAD_MCP_ENABLE_TASK_DEDUP` | `true` | MCP 생성 전 유사 태스크를 대표 태스크로 사전 압축할지 여부 |
| `TAD_MCP_TASK_DEDUP_MIN_TASKS` | `12` | 사전 압축을 시작할 최소 태스크 수 |
| `TAD_MCP_TASK_DEDUP_MIN_SIMILARITY` | `0.72` | 대표 태스크로 묶는 최소 유사도 |
| `TAD_MCP_TASK_DEDUP_LOOSE_SIMILARITY` | `0.45` | 구조 유사(태그/입출력/의도) 조건에서 적용할 완화 유사도 |
| `TAD_MCP_TASK_DEDUP_MIN_TAG_OVERLAP` | `0.5` | 구조 유사 판정 시 최소 태그 중첩 비율 |
| `TAD_MCP_TASK_DEDUP_TEMPLATE_SIMILARITY` | `0.32` | 템플릿형 작업(예: 승인 라우팅) dedup 시 적용할 최소 유사도 |
| `ROUTER_MAX_FALLBACK_RATIO` | `0.2` | Router 허용 최대 임베딩 fallback 비율 |
| `ROUTER_MIN_EMBED_CALLS` | `5` | fallback 비율 판단 최소 임베딩 호출 수 |
| `ROUTER_DISABLE_ON_FALLBACK_RATIO` | `false` | `true`면 fallback 비율 초과 시 Router 비활성화 |
| `ROUTE_MIN_CONFIDENCE` | `0.35` | 라우팅 성공으로 인정할 최소 confidence |
| `ROUTE_MIN_PROB_GAP` | `0.0` | top1-top2 확률 차이 최소값 (`/api/route` 게이트) |
| `HIERARCHY_SIMPLE_THRESHOLD` | `0.45` | 단순 경로(`Master→Unit`) 선택 임계값 |
| `HIERARCHY_MAX_ORCHESTRATORS` | `2` | 복합 경로에서 선택할 Orchestrator 최대 수 |
| `HIERARCHY_MIN_ORCHESTRATOR_PROB` | `0.12` | Orchestrator 후보 채택 최소 확률 |

MCP Tool 단계 타임아웃/속도 설정:
- `TAD_MCP_TIMEOUT_MS=0` 또는 음수: LLM HTTP 타임아웃 비활성화
- `TAD_MCP_BATCH_SIZE=10`: 30개 태스크 기준 3개 chunk로 분할
- `TAD_MCP_MAX_WORKERS=4`: 최대 4개 chunk 병렬 처리
- `TAD_MCP_ENABLE_TOOL_MERGE=true`: 동일 Agent 내 유사 태스크를 공유 Tool로 통합
- `TAD_MCP_ENABLE_TASK_DEDUP=true`: LLM 호출 전 유사 태스크를 대표 태스크로 압축해 대기시간 감소
- `TAD_MCP_TASK_DEDUP_LOOSE_SIMILARITY=0.45`: 거의 동일 태스크가 아니어도 구조적으로 유사하면 dedup 허용
- `TAD_MCP_TASK_DEDUP_TEMPLATE_SIMILARITY=0.32`: 이름 템플릿/출력 구조가 같은 작업을 dedup 허용

---

## 테스트

```bash
# 전체 테스트 실행
.venv/bin/python -m pytest tests/ -v

# 모듈별 실행
.venv/bin/python -m pytest tests/test_tool_balancer.py -v   # 16개
.venv/bin/python -m pytest tests/test_homotopy_router.py -v # 14개
.venv/bin/python -m pytest tests/test_query_manifold.py -v  # 7개
```

## 라우팅 A/B 평가

```bash
# task 기반 자동 쿼리셋으로 strict_guard vs hybrid_guard 비교
.venv/bin/python -m tad_mapper.eval.routing_ab \
  --input data/samples/ai_opportunities_30.csv

# 커스텀 쿼리셋(JSON 배열) 사용
.venv/bin/python -m tad_mapper.eval.routing_ab \
  --input data/samples/ai_opportunities_30.csv \
  --queries data/samples/routing_queries.json

# confidence/gap 임계값 스윕 + 리더보드 생성
.venv/bin/python -m tad_mapper.eval.routing_ab \
  --input data/samples/ai_opportunities_30.csv \
  --queries data/samples/routing_queries.json \
  --conf-thresholds 0.1,0.2,0.3,0.4 \
  --gap-thresholds 0.0,0.05,0.1,0.15
```

## 계층 라우팅 계획 API

```bash
# Master→(Orchestrator)→Unit 계층 계획 조회
curl -X POST http://localhost:8000/api/route-hierarchy \
  -H "Content-Type: application/json" \
  -d '{"output_id":"<analyze_output_id>","query":"환율 분석 그리고 리스크 보고서 작성"}'

# 계층 계획 + 서브태스크별 Unit Tool 합성까지 한번에 실행
curl -X POST http://localhost:8000/api/route-hierarchy-and-compose \
  -H "Content-Type: application/json" \
  -d '{"output_id":"<analyze_output_id>","query":"환율 분석 그리고 리스크 보고서 작성"}'
```

응답의 `hierarchical_plan.path_type`:
- `master_unit`: 단순 쿼리 (직접 Unit 실행)
- `master_orchestrator_unit`: 복합/모호 쿼리 (Orchestrator가 서브태스크 분해 후 Unit 배정)

출력:
- `output/eval/routing_ab_*.json` : 모드별 요약 지표
- `output/eval/routing_ab_*.csv` : 쿼리별 상세 결과(정답 agent/group hit 포함)
- `output/eval/routing_ab_*.md` : threshold leaderboard 요약 리포트

평가용 쿼리셋 자동 생성:
```bash
.venv/bin/python -m tad_mapper.eval.generate_queries \
  --input data/samples/ai_opportunities_30.csv \
  --output data/samples/routing_queries.generated.json \
  --variants 3
```

실무형 50개 태스크 샘플:
- 입력 샘플: `data/samples/ai_opportunities_50_company.csv`
- 생성 쿼리셋: `data/samples/routing_queries.50_company.generated.json`

Tool 병합 A/B 평가:
```bash
.venv/bin/python -m tad_mapper.eval.tool_merge_ab \
  --input data/samples/ai_opportunities_30.csv \
  --queries data/samples/routing_queries.generated.json \
  --merge-similarity 0.45 \
  --max-tasks-per-tool 4
```

스윕(리더보드 자동 생성):
```bash
.venv/bin/python -m tad_mapper.eval.tool_merge_ab \
  --input data/samples/ai_opportunities_30.csv \
  --queries data/samples/routing_queries.generated.json \
  --merge-similarities 0.45,0.5,0.55 \
  --max-tasks-options 2,3,4
```

출력:
- `output/eval/tool_merge_ab_*.json` : 병합 OFF/ON 비교 지표
- `output/eval/tool_merge_ab_*.md` : Tool 수 절감 + 라우팅 일관성 요약

---

*TAD-Mapper v0.2.5 · Powered by Gemini 3.0 · TDA (Topological Data Analysis) + 수학적 정식화*
