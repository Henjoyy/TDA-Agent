# TAD-Mapper 분석 결과 리포트

**분석 일시:** 2026-02-18 12:43
**여정 제목:** Tmpbji490Lh
**총 태스크 수:** 30개
**발견된 Agent 수:** 2개

---

## 📊 Agent 구성 요약


### 1. Agent 0
- **역할:** 태스크 그룹 0 처리
- **역량:** task_execution
- **담당 태스크 (2개):**
  - 원산지 증명서(CO) 발급 신청
  - 바이어 미팅 일정 조율



### 2. Agent 1
- **역할:** 태스크 그룹 1 처리
- **역량:** task_execution
- **담당 태스크 (28개):**
  - 글로벌 관세율 조회
  - 수출입 통계 분석
  - 선적 서류 자동 검증
  - 물류비 견적 비교
  - 바이어 이메일 초안 작성
  - 환율 변동 알림
  - HS Code 자동 분류
  - 해상 운임 예측
  - 신용장(L/C) 독소조항 검토
  - 경쟁사 동향 뉴스 요약
  - AI 챗봇 고객 상담
  - 거래처 신용도 평가
  - 다국어 계약서 번역
  - 창고 재고 최적화 제안
  - 적하 보험료 산출
  - 해외 전시회 추천
  - 공급망 리스크 모니터링
  - FTA 혜택 조회
  - 제품 카탈로그 이미지 생성
  - 송장(Invoice) 데이터 추출(OCR)
  - 컨테이너 적재 시뮬레이션
  - 해외 바이어 발굴
  - 규제 준수(Compliance) 체크
  - SNS 마케팅 문구 작성
  - 클레임 대응 메일 작성
  - 시장 진입 전략 리포트
  - 수입 통관 진행상황 추적
  - 회의록 자동 요약 및 할일 추출




---

## 🔧 MCP Tool 스키마 목록 (30개)


### `execute_원산지_증명서_co_발급_신청`
**설명:** 수출 데이터 기반으로 원산지 증명서 발급 신청서를 자동으로 작성하고 제출합니다.
**담당 Agent:** agent_0

```json
{
  "name": "execute_원산지_증명서_co_발급_신청",
  "description": "수출 데이터 기반으로 원산지 증명서 발급 신청서를 자동으로 작성하고 제출합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_0",
    "source_task_id": "task_013",
    "source_task_name": "원산지 증명서(CO) 발급 신청",
    "confidence": 0.5
  }
}
```


### `execute_바이어_미팅_일정_조율`
**설명:** 시차가 있는 해외 바이어와의 화상 회의 가능한 교집합 시간을 찾아 미팅을 제안합니다.
**담당 Agent:** agent_0

```json
{
  "name": "execute_바이어_미팅_일정_조율",
  "description": "시차가 있는 해외 바이어와의 화상 회의 가능한 교집합 시간을 찾아 미팅을 제안합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_0",
    "source_task_id": "task_018",
    "source_task_name": "바이어 미팅 일정 조율",
    "confidence": 0.5
  }
}
```


### `execute_글로벌_관세율_조회`
**설명:** HS Code 기반으로 전 세계 국가별 관세율 및 협정 세율을 조회합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_글로벌_관세율_조회",
  "description": "HS Code 기반으로 전 세계 국가별 관세율 및 협정 세율을 조회합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_001",
    "source_task_name": "글로벌 관세율 조회",
    "confidence": 0.5
  }
}
```


### `execute_수출입_통계_분석`
**설명:** 특정 기간 동안의 품목별/국가별 수출입 통계를 분석하여 리포트를 생성합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_수출입_통계_분석",
  "description": "특정 기간 동안의 품목별/국가별 수출입 통계를 분석하여 리포트를 생성합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_002",
    "source_task_name": "수출입 통계 분석",
    "confidence": 0.5
  }
}
```


### `execute_선적_서류_자동_검증`
**설명:** 인보이스와 패킹리스트의 데이터 불일치(중량
**담당 Agent:** agent_1

```json
{
  "name": "execute_선적_서류_자동_검증",
  "description": "인보이스와 패킹리스트의 데이터 불일치(중량",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_003",
    "source_task_name": "선적 서류 자동 검증",
    "confidence": 0.5
  }
}
```


### `execute_물류비_견적_비교`
**설명:** 출발지/도착지 및 화물 정보를 입력하면 여러 포워더의 예상 견적을 비교 분석합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_물류비_견적_비교",
  "description": "출발지/도착지 및 화물 정보를 입력하면 여러 포워더의 예상 견적을 비교 분석합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_004",
    "source_task_name": "물류비 견적 비교",
    "confidence": 0.5
  }
}
```


### `execute_바이어_이메일_초안_작성`
**설명:** 신규 바이어에게 보낼 제품 소개 이메일 초안을 상대방 문화와 언어에 맞춰 작성합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_바이어_이메일_초안_작성",
  "description": "신규 바이어에게 보낼 제품 소개 이메일 초안을 상대방 문화와 언어에 맞춰 작성합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_005",
    "source_task_name": "바이어 이메일 초안 작성",
    "confidence": 0.5
  }
}
```


### `execute_환율_변동_알림`
**설명:** 설정한 목표 환율 도달 시 또는 급격한 변동 감지 시 담당자에게 즉시 알림을 보냅니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_환율_변동_알림",
  "description": "설정한 목표 환율 도달 시 또는 급격한 변동 감지 시 담당자에게 즉시 알림을 보냅니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_006",
    "source_task_name": "환율 변동 알림",
    "confidence": 0.5
  }
}
```


### `execute_hs_code_자동_분류`
**설명:** 제품 설명과 이미지를 분석하여 가장 적절한 HS Code를 추천하고 분류 근거를 제시합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_hs_code_자동_분류",
  "description": "제품 설명과 이미지를 분석하여 가장 적절한 HS Code를 추천하고 분류 근거를 제시합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_007",
    "source_task_name": "HS Code 자동 분류",
    "confidence": 0.5
  }
}
```


### `execute_해상_운임_예측`
**설명:** 과거 운임 데이터와 시장 지수(BDI 등)를 기반으로 향후 3개월간의 해상 운임 추이를 예측합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_해상_운임_예측",
  "description": "과거 운임 데이터와 시장 지수(BDI 등)를 기반으로 향후 3개월간의 해상 운임 추이를 예측합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_008",
    "source_task_name": "해상 운임 예측",
    "confidence": 0.5
  }
}
```


### `execute_신용장_l_c_독소조항_검토`
**설명:** 신용장 문구를 분석하여 불리하거나 위험한 독소 조항이 있는지 검토하고 수정안을 제안합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_신용장_l_c_독소조항_검토",
  "description": "신용장 문구를 분석하여 불리하거나 위험한 독소 조항이 있는지 검토하고 수정안을 제안합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_009",
    "source_task_name": "신용장(L/C) 독소조항 검토",
    "confidence": 0.5
  }
}
```


### `execute_경쟁사_동향_뉴스_요약`
**설명:** 주요 경쟁사의 신제품 출시
**담당 Agent:** agent_1

```json
{
  "name": "execute_경쟁사_동향_뉴스_요약",
  "description": "주요 경쟁사의 신제품 출시",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_010",
    "source_task_name": "경쟁사 동향 뉴스 요약",
    "confidence": 0.5
  }
}
```


### `execute_ai_챗봇_고객_상담`
**설명:** 무역박람회 부스 방문객의 일반적인 질문(가격
**담당 Agent:** agent_1

```json
{
  "name": "execute_ai_챗봇_고객_상담",
  "description": "무역박람회 부스 방문객의 일반적인 질문(가격",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_011",
    "source_task_name": "AI 챗봇 고객 상담",
    "confidence": 0.5
  }
}
```


### `execute_거래처_신용도_평가`
**설명:** 해외 거래처의 재무제표와 신용 리포트를 분석하여 거래 위험 등급을 산출합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_거래처_신용도_평가",
  "description": "해외 거래처의 재무제표와 신용 리포트를 분석하여 거래 위험 등급을 산출합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_012",
    "source_task_name": "거래처 신용도 평가",
    "confidence": 0.5
  }
}
```


### `execute_다국어_계약서_번역`
**설명:** 무역 계약서를 법률 용어를 고려하여 지정된 언어로 초벌 번역합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_다국어_계약서_번역",
  "description": "무역 계약서를 법률 용어를 고려하여 지정된 언어로 초벌 번역합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_014",
    "source_task_name": "다국어 계약서 번역",
    "confidence": 0.5
  }
}
```


### `execute_창고_재고_최적화_제안`
**설명:** 현재 재고량과 출고 패턴을 분석하여 적정 안전 재고량을 제안하고 발주 시점을 알립니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_창고_재고_최적화_제안",
  "description": "현재 재고량과 출고 패턴을 분석하여 적정 안전 재고량을 제안하고 발주 시점을 알립니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_015",
    "source_task_name": "창고 재고 최적화 제안",
    "confidence": 0.5
  }
}
```


### `execute_적하_보험료_산출`
**설명:** 화물 가액과 운송 조건을 기반으로 예상 적하 보험료를 산출합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_적하_보험료_산출",
  "description": "화물 가액과 운송 조건을 기반으로 예상 적하 보험료를 산출합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_016",
    "source_task_name": "적하 보험료 산출",
    "confidence": 0.5
  }
}
```


### `execute_해외_전시회_추천`
**설명:** 우리 회사의 주력 품목과 타겟 시장에 적합한 해외 유명 전시회를 추천합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_해외_전시회_추천",
  "description": "우리 회사의 주력 품목과 타겟 시장에 적합한 해외 유명 전시회를 추천합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_017",
    "source_task_name": "해외 전시회 추천",
    "confidence": 0.5
  }
}
```


### `execute_공급망_리스크_모니터링`
**설명:** 주요 공급망 국가의 자연재해
**담당 Agent:** agent_1

```json
{
  "name": "execute_공급망_리스크_모니터링",
  "description": "주요 공급망 국가의 자연재해",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_019",
    "source_task_name": "공급망 리스크 모니터링",
    "confidence": 0.5
  }
}
```


### `execute_fta_혜택_조회`
**설명:** 특정 품목 수출 시 활용 가능한 FTA 협정과 예상 절세액을 조회합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_fta_혜택_조회",
  "description": "특정 품목 수출 시 활용 가능한 FTA 협정과 예상 절세액을 조회합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_020",
    "source_task_name": "FTA 혜택 조회",
    "confidence": 0.5
  }
}
```


### `execute_제품_카탈로그_이미지_생성`
**설명:** 제품 컨셉 설명을 바탕으로 마케팅용 제품 연출 이미지를 생성합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_제품_카탈로그_이미지_생성",
  "description": "제품 컨셉 설명을 바탕으로 마케팅용 제품 연출 이미지를 생성합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_021",
    "source_task_name": "제품 카탈로그 이미지 생성",
    "confidence": 0.5
  }
}
```


### `execute_송장_invoice_데이터_추출_ocr`
**설명:** PDF나 이미지 형태의 인보이스에서 날짜
**담당 Agent:** agent_1

```json
{
  "name": "execute_송장_invoice_데이터_추출_ocr",
  "description": "PDF나 이미지 형태의 인보이스에서 날짜",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_022",
    "source_task_name": "송장(Invoice) 데이터 추출(OCR)",
    "confidence": 0.5
  }
}
```


### `execute_컨테이너_적재_시뮬레이션`
**설명:** 포장 규격과 수량을 입력하면 컨테이너 적재 효율을 극대화하는 적재 패턴을 시뮬레이션합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_컨테이너_적재_시뮬레이션",
  "description": "포장 규격과 수량을 입력하면 컨테이너 적재 효율을 극대화하는 적재 패턴을 시뮬레이션합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_023",
    "source_task_name": "컨테이너 적재 시뮬레이션",
    "confidence": 0.5
  }
}
```


### `execute_해외_바이어_발굴`
**설명:** 특정 품목을 수입하는 해외 잠재 바이어 리스트를 무역 데이터를 통해 발굴합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_해외_바이어_발굴",
  "description": "특정 품목을 수입하는 해외 잠재 바이어 리스트를 무역 데이터를 통해 발굴합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_024",
    "source_task_name": "해외 바이어 발굴",
    "confidence": 0.5
  }
}
```


### `execute_규제_준수_compliance_체크`
**설명:** 수출 예정 품목이 타겟 국가의 수입 금지 품목이거나 전략 물자인지 사전 스크리닝합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_규제_준수_compliance_체크",
  "description": "수출 예정 품목이 타겟 국가의 수입 금지 품목이거나 전략 물자인지 사전 스크리닝합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_025",
    "source_task_name": "규제 준수(Compliance) 체크",
    "confidence": 0.5
  }
}
```


### `execute_sns_마케팅_문구_작성`
**설명:** 제품 홍보를 위한 링크드인/페이스북용 마케팅 문구와 해시태그를 자동 생성합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_sns_마케팅_문구_작성",
  "description": "제품 홍보를 위한 링크드인/페이스북용 마케팅 문구와 해시태그를 자동 생성합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_026",
    "source_task_name": "SNS 마케팅 문구 작성",
    "confidence": 0.5
  }
}
```


### `execute_클레임_대응_메일_작성`
**설명:** 제품 불량 등 클레임 접수 시
**담당 Agent:** agent_1

```json
{
  "name": "execute_클레임_대응_메일_작성",
  "description": "제품 불량 등 클레임 접수 시",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_027",
    "source_task_name": "클레임 대응 메일 작성",
    "confidence": 0.5
  }
}
```


### `execute_시장_진입_전략_리포트`
**설명:** 타겟 국가의 경제 지표
**담당 Agent:** agent_1

```json
{
  "name": "execute_시장_진입_전략_리포트",
  "description": "타겟 국가의 경제 지표",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_028",
    "source_task_name": "시장 진입 전략 리포트",
    "confidence": 0.5
  }
}
```


### `execute_수입_통관_진행상황_추적`
**설명:** B/L 번호를 입력하면 관세청 유니패스와 연동하여 현재 통관 진행 단계를 실시간 조회합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_수입_통관_진행상황_추적",
  "description": "B/L 번호를 입력하면 관세청 유니패스와 연동하여 현재 통관 진행 단계를 실시간 조회합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_029",
    "source_task_name": "수입 통관 진행상황 추적",
    "confidence": 0.5
  }
}
```


### `execute_회의록_자동_요약_및_할일_추출`
**설명:** 바이어와의 미팅 녹취록이나 텍스트를 분석하여 핵심 내용 요약 및 Action Item을 추출합니다.
**담당 Agent:** agent_1

```json
{
  "name": "execute_회의록_자동_요약_및_할일_추출",
  "description": "바이어와의 미팅 녹취록이나 텍스트를 분석하여 핵심 내용 요약 및 Action Item을 추출합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "처리할 요청 내용"
      }
    },
    "required": [
      "query"
    ]
  },
  "annotations": {
    "assigned_agent": "agent_1",
    "source_task_id": "task_030",
    "source_task_name": "회의록 자동 요약 및 할일 추출",
    "confidence": 0.5
  }
}
```



---

## ⚠️ 경고 사항


✅ 논리적 구멍이 발견되지 않았습니다.



✅ 중복 할당이 발견되지 않았습니다.
