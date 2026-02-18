# TAD-Mapper Agent Analysis Skills & Guidelines

이 문서는 Gemini 모델이 사용자 여정(User Journey)을 분석하고 Unit Agent를 설계할 때 따라야 할 핵심 지침(Skills)을 정의합니다.

## 1. Unit Agent의 정의 (Definition)
Unit Agent는 **"단일한 비즈니스 목적을 달성하기 위해 긴밀하게 연관된 태스크들의 집합"**입니다.
- **Micro-Service**: 너무 작으면 관리 비용이 증가합니다. (예: "로그인 버튼 클릭 에이전트" ❌)
- **Monolith**: 너무 크면 전문성이 떨어집니다. (예: "전체 시스템 관리 에이전트" ❌)
- **Optimal**: 하나의 명확한 책임을 가집니다. (예: "무역 리스크 분석 에이전트" ✅)

## 2. 태스크 분석 기준 (10D Analysis Criteria)
태스크를 분석할 때는 다음 10가지 차원(Dimension)을 고려하여 0.0 ~ 1.0 사이의 벡터를 추출합니다:

1. **데이터 유형 (Data Type)**: 정형(DB/수치/API) vs 비정형(텍스트/이미지)
2. **추론 깊이 (Reasoning Depth)**: 단순 조회 vs 분석/예측/복합 추론
3. **자동화 가능성 (Automation Potential)**: 규칙 기반 완전 자동화 vs 사람의 주관적 판단 필수
4. **상호작용 유형 (Interaction Type)**: 읽기 전용 vs 외부 시스템 쓰기/실행/알림
5. **출력 복잡도 (Output Complexity)**: 단순 결과값 vs 복합 보고서/문서/시각화
6. **도메인 특화도 (Domain Specificity)**: 범용 서비스 vs 고도 전문 지식/규정 필요
7. **시간 민감도 (Temporal Sensitivity)**: 배치/지연 가능 vs 실시간 응답/모니터링 필수
8. **데이터 규모 (Data Volume)**: 단건/소량 처리 vs 대용량/고처리량 빅데이터
9. **보안 수준 (Security Level)**: 공개 데이터 vs 민감 정보/규제/개인정보 보호 중요
10. **상태 의존성 (State Dependency)**: 단발성(Stateless) vs 세션/워크플로/트랜잭션 상태 유지 강함

## 3. Agent Naming & Role Definition (Best Practices)

### ✅ 좋은 예 (Good)
- **이름**: "수출입 통계 분석 에이전트"
- **역할**: "관세청 및 무역 데이터베이스에서 수출입 실적을 수집하고, 품목별/국가별 통계 리포트를 생성합니다."
- **이유**: 구체적인 도메인(수출입)과 기능(통계 분석)이 명시됨.

### ❌ 나쁜 예 (Bad)
- **이름**: "데이터 처리 에이전트", "범용 에이전트 1", "Search Agent"
- **역할**: "데이터를 처리하고 정보를 검색합니다."
- **이유**: 너무 포괄적이고 모호함. 어떤 데이터인지, 무슨 목적에 쓰이는지 알 수 없음.

## 4. MCP Tool 설계 지침 (Tool Design)
Unit Agent가 사용할 도구(Tool)는 다음 원칙을 따릅니다:
- **Atomic**: 하나의 툴은 하나의 기능만 수행해야 합니다.
- **Schema**: 입력 파라미터는 명확한 타입(string, number, enum)을 가져야 합니다.
- **Naming**: `verb_noun` 형식을 엄수합니다 (예: `analyze_risk`, `fetch_customs_data`).

---
**Note to Model**: 위 가이드라인을 바탕으로, 주어진 태스크들이 최적의 위상학적 클러스터를 형성하도록 10차원 특징 벡터를 정밀하게 추출해주세요.
