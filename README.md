# TAD-Mapper 🗺️

**AI 적용 기회(User Journey)를 입력하면 위상수학적 알고리즘으로 Unit Agent와 MCP Tool을 자동 설계하는 시스템**

> **v0.2.2** — 안정화 리팩토링: 커버리지 계산 정확도 개선, Tool 합성 정렬 수정, Agent ID 안정화, 세션 TTL/LRU 관리

---

## 수학적 정식화

### 1. 사용자 쿼리 매니폴드 & 에이전트 피복 (Q ⊆ ∪ Uᵢ)

모든 사용자 쿼리가 속하는 위상 공간 Q를 정의하고, Unit Agent들이 이를 빈틈없이 덮는지 검증합니다.

$$\mathcal{Q} \subseteq \bigcup_{i \in Agents} U_i$$

### 2. 호모토피 라우팅 함수 Φ

Master Agent가 쿼리를 의미적으로 분류하여 적절한 Unit Agent로 라우팅합니다.

$$\Phi : \mathcal{Q} \to \{U_{Search}, U_{Stats}, \ldots, U_{Report}\} \qquad \Phi(x) = U_k \iff [x] = [x_k]$$

"자료 찾아줘", "데이터 검색해", "정보 좀 줘" → 호모토피 동치류 [x]가 같으므로 모두 동일한 Agent로 라우팅

### 3. MCP Tool 합성 (함수 합성)

Agent가 상황에 맞게 MCP Tool을 동적으로 조립하여 문제를 해결합니다.

$$y = (t_{\pi(m)} \circ \cdots \circ t_{\pi(2)} \circ t_{\pi(1)})(x)$$

---

## Repository

[https://github.com/Henjoyy/TDA-Agent](https://github.com/Henjoyy/TDA-Agent)

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
[1]  Feature Extraction    ← Gemini LLM → 6D 위상학적 벡터
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
│   │   ├── feature_extractor.py   # 6D 위상 특징 벡터 추출 (Gemini)
│   │   ├── tda_analyzer.py        # Mapper 알고리즘 + KMeans 클러스터링
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
| `result.json` | 구조화된 JSON 결과 |
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
| `TDA_N_INTERVALS` | `10` | Mapper 구간 수 |
| `TDA_OVERLAP_FRAC` | `0.3` | 구간 오버랩 비율 |

---

## 테스트

```bash
# 전체 테스트 실행 (35개)
.venv/bin/python -m pytest tests/ -v

# 모듈별 실행
.venv/bin/python -m pytest tests/test_tool_balancer.py -v   # 16개
.venv/bin/python -m pytest tests/test_homotopy_router.py -v # 12개
.venv/bin/python -m pytest tests/test_query_manifold.py -v  # 7개
```

---

*TAD-Mapper v0.2.2 · Powered by Gemini 3.0 · TDA (Topological Data Analysis) + 수학적 정식화*
