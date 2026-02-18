"""
리포트 생성기 - Jinja2 기반 Markdown/JSON 리포트
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from tad_mapper.engine.feature_extractor import TopologicalFeature
from tad_mapper.engine.tda_analyzer import DiscoveredAgent
from tad_mapper.models.agent import HoleWarning, MappingResult, OverlapWarning
from tad_mapper.models.mcp_tool import MCPToolSchema

_REPORT_TEMPLATE = """# TAD-Mapper 분석 결과 리포트

**분석 일시:** {{ timestamp }}
**여정 제목:** {{ journey_title }}
**총 태스크 수:** {{ total_tasks }}개
**발견된 Agent 수:** {{ agent_count }}개

---

## 📊 Agent 구성 요약

{% for agent in agents %}
### {{ loop.index }}. {{ agent.suggested_name or agent.agent_id }}
- **역할:** {{ agent.suggested_role or '미정의' }}
- **역량:** {{ agent.suggested_capabilities | join(', ') or '미정의' }}
- **담당 태스크 ({{ agent.task_ids | length }}개):**
{% for name in agent.task_names %}  - {{ name }}
{% endfor %}

{% endfor %}

---

## 🔧 MCP Tool 스키마 목록 ({{ tools | length }}개)

{% for tool in tools %}
### `{{ tool.name }}`
**설명:** {{ tool.description }}
**담당 Agent:** {{ tool.annotations.assigned_agent if tool.annotations else '미지정' }}

```json
{{ tool.to_dict() | tojson(indent=2) }}
```

{% endfor %}

---

## ⚠️ 경고 사항

{% if holes %}
### 논리적 구멍 (Hole) - {{ holes | length }}개
{% for hole in holes %}
- **[{{ hole.hole_type | upper }}]** {{ hole.description }}
  - 💡 제안: {{ hole.suggestion }}
{% endfor %}
{% else %}
✅ 논리적 구멍이 발견되지 않았습니다.
{% endif %}

{% if overlaps %}
### 중복 할당 (Overlap) - {{ overlaps | length }}개
{% for overlap in overlaps %}
- {{ overlap.description }}
{% endfor %}
{% else %}
✅ 중복 할당이 발견되지 않았습니다.
{% endif %}
"""


class ReportGenerator:
    """분석 결과를 Markdown 및 JSON으로 출력"""

    def generate_markdown(
        self,
        journey_title: str,
        agents: list[DiscoveredAgent],
        tools: list[MCPToolSchema],
        holes: list[HoleWarning],
        overlaps: list[OverlapWarning],
        total_tasks: int,
    ) -> str:
        """Jinja2 템플릿으로 Markdown 리포트 생성"""
        env = Environment(autoescape=False)
        env.filters["tojson"] = lambda v, indent=None: json.dumps(
            v, ensure_ascii=False, indent=indent
        )
        template = env.from_string(_REPORT_TEMPLATE)
        return template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            journey_title=journey_title,
            total_tasks=total_tasks,
            agent_count=len(agents),
            agents=agents,
            tools=tools,
            holes=holes,
            overlaps=overlaps,
        )

    def generate_json(
        self,
        journey_title: str,
        agents: list[DiscoveredAgent],
        tools: list[MCPToolSchema],
        holes: list[HoleWarning],
        overlaps: list[OverlapWarning],
        features: list[TopologicalFeature] | None = None,
    ) -> dict:
        """JSON 형식 결과 딕셔너리 생성"""
        feature_items = features or []
        return {
            "journey_title": journey_title,
            "timestamp": datetime.now().isoformat(),
            "feature_space": {
                "dimensions": len(feature_items[0].vector) if feature_items else 0,
                "dimension_names": [
                    "data_type",
                    "reasoning_depth",
                    "automation_potential",
                    "interaction_type",
                    "output_complexity",
                    "domain_specificity",
                    "temporal_sensitivity",
                    "data_volume",
                    "security_level",
                    "state_dependency",
                ],
                "task_features": [
                    {
                        "task_id": f.task_id,
                        "task_name": f.task_name,
                        "vector": [float(v) for v in f.vector.tolist()],
                    }
                    for f in feature_items
                ],
            },
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "name": a.suggested_name,
                    "name_ko": a.suggested_name,
                    "role": a.suggested_role,
                    "capabilities": a.suggested_capabilities,
                    "tool_prefix": a.tool_prefix,
                    "task_ids": a.task_ids,
                    "task_names": a.task_names,
                }
                for a in agents
            ],
            "mcp_tools": [t.to_dict() for t in tools],
            "holes": [h.model_dump() for h in holes],
            "overlaps": [o.model_dump() for o in overlaps],
        }

    def save(self, content: str, path: str | Path) -> None:
        """파일로 저장"""
        Path(path).write_text(content, encoding="utf-8")
