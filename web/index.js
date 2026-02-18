/* TAD-Mapper Dashboard Client Logic — v0.2.0 */

const AGENT_COLORS = [
  '#6366f1', '#f59e0b', '#10b981', '#ef4444',
  '#8b5cf6', '#06b6d4', '#f97316', '#84cc16',
];

let selectedFile = null;
let currentResult = null;
let currentOutputId = null;  // 세션 ID (라우팅/합성 API용)
// 0 이하이면 브라우저 측 강제 타임아웃을 적용하지 않음
const ANALYZE_TIMEOUT_MS = 0;

// ── File Upload ──────────────────────────────────────────────
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');

uploadArea.addEventListener('dragover', e => {
  e.preventDefault();
  uploadArea.classList.add('drag-over');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  selectedFile = file;
  uploadArea.querySelector('.upload-text').textContent = `✅ ${file.name}`;
  uploadArea.querySelector('.upload-hint').textContent =
    `${(file.size / 1024).toFixed(1)} KB · ${file.type || file.name.split('.').pop().toUpperCase()}`;
  analyzeBtn.disabled = false;
}

// ── Sample Data ──────────────────────────────────────────────
async function loadSample() {
  const sampleRes = await fetch('/api/sample').catch(() => null);
  if (sampleRes && sampleRes.ok) {
    const blob = await sampleRes.blob();
    setFile(new File([blob], 'trade_journey.json', { type: 'application/json' }));
  } else {
    alert('샘플 데이터를 불러오는 중 오류가 발생했습니다. 직접 파일을 업로드해주세요.');
  }
}

// ── Analysis ─────────────────────────────────────────────────
async function runAnalysis() {
  if (!selectedFile) return;

  showLoading(true);
  animateLoadingSteps();

  const formData = new FormData();
  formData.append('file', selectedFile);
  const nAgents = document.getElementById('nAgents').value;
  if (nAgents) formData.append('n_agents', nAgents);

  const controller = new AbortController();
  const timeoutId = ANALYZE_TIMEOUT_MS > 0
    ? setTimeout(() => controller.abort(), ANALYZE_TIMEOUT_MS)
    : null;

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
    if (timeoutId) clearTimeout(timeoutId);
    if (!res.ok) {
      const err = await res.json();
      throw new Error(extractErrorMessage(err, '분석 실패'));
    }
    const data = await res.json();
    currentResult = data;
    currentOutputId = data.output_id;
    showLoading(false);
    renderResults(data);
  } catch (e) {
    if (timeoutId) clearTimeout(timeoutId);
    showLoading(false);
    if (e.name === 'AbortError') {
      alert(`오류: 분석 요청 시간(${ANALYZE_TIMEOUT_MS / 1000}초)을 초과했습니다. 태스크 수를 줄이거나 다시 시도해주세요.`);
    } else {
      alert(`오류: ${e.message}`);
    }
  }
}

// ── Loading Animation ────────────────────────────────────────
function showLoading(show) {
  document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
}

function animateLoadingSteps() {
  const steps = ['step1', 'step2', 'step3', 'step4'];
  const msgs = [
    'Gemini AI가 특징 벡터를 추출하고 있습니다',
    'TDA Mapper 알고리즘으로 위상 구조를 분석합니다',
    'KMeans 클러스터링으로 Agent를 발견합니다',
    'MCP Tool JSON 스키마를 생성합니다',
  ];
  let i = 0;
  steps.forEach(id => document.getElementById(id).className = 'step-item');

  const interval = setInterval(() => {
    if (i > 0) document.getElementById(steps[i - 1]).className = 'step-item done';
    if (i < steps.length) {
      document.getElementById(steps[i]).className = 'step-item active';
      document.getElementById('loadingStep').textContent = msgs[i];
      i++;
    } else {
      clearInterval(interval);
    }
  }, 2500);
}

// ── Render Results ───────────────────────────────────────────
function renderResults(data) {
  const section = document.getElementById('results');
  section.style.display = 'block';
  section.scrollIntoView({ behavior: 'smooth' });

  renderSummary(data.summary, data.coverage, data.tool_balance);

  // agent_id → 자연어 이름 맵 생성
  const agentNameMap = {};
  (data.result.agents || []).forEach(a => {
    agentNameMap[a.agent_id] = a.name || a.agent_id;
  });

  renderAgents(data.result.agents, data.result.mcp_tools);
  renderTools(data.result.mcp_tools, agentNameMap);
  renderWarnings(data.result.holes, data.result.overlaps);
  renderCoverage(data.coverage, data.tool_balance, data.files, agentNameMap);
  renderViz(data.files);
  renderDownloads(data.files);
}

function renderSummary(s, coverage, balance) {
  const grid = document.getElementById('summaryGrid');
  const items = [
    { num: s.total_tasks, label: '총 태스크' },
    { num: s.agent_count, label: '발견된 Agent' },
    { num: s.mcp_tool_count, label: 'MCP Tool' },
    { num: s.hole_count, label: '구멍(Hole)' },
    { num: s.overlap_count, label: '중복(Overlap)' },
  ];

  if (coverage) {
    const pct = (coverage.coverage_ratio * 100).toFixed(0);
    items.push({
      num: pct + '%',
      label: 'Q ⊆ ∪Ui 커버리지',
      color: coverage.coverage_complete ? '#10b981' : '#f59e0b',
    });
  }
  if (balance) {
    items.push({
      num: balance.gini_coefficient.toFixed(2),
      label: 'Gini 불균형 지수',
      color: balance.gini_coefficient < 0.3 ? '#10b981' : '#ef4444',
    });
  }

  grid.innerHTML = items.map(i => `
    <div class="summary-card">
      <div class="summary-card-num" ${i.color ? `style="color:${i.color}"` : ''}>${i.num}</div>
      <div class="summary-card-label">${i.label}</div>
    </div>
  `).join('');
}

// ── Coverage Panel (Q ⊆ ∪Ui) ──────────────────────────────────
function renderCoverage(coverage, balance, files) {
  const panel = document.getElementById('coveragePanel');
  let html = '';

  if (coverage) {
    const pct = (coverage.coverage_ratio * 100).toFixed(1);
    const statusIcon = coverage.coverage_complete ? '✅' : '⚠️';
    const statusColor = coverage.coverage_complete ? '#10b981' : '#f59e0b';

    html += `
        <div class="coverage-card">
          <div class="coverage-header">
            <span class="coverage-status" style="color:${statusColor}">${statusIcon} 커버리지 ${pct}%</span>
            <span class="coverage-badge">${coverage.coverage_complete ? '완전 피복 (Q ⊆ ∪Ui 충족)' : '미완 — 갭 존재'}</span>
          </div>
          <div class="coverage-bars">
            <div class="coverage-bar-row">
              <span>커버리지</span>
              <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:#6366f1"></div></div>
              <span>${pct}%</span>
            </div>
            <div class="coverage-bar-row">
              <span>중첩(Overlap)</span>
              <div class="bar-track"><div class="bar-fill" style="width:${(coverage.overlap_ratio * 100).toFixed(1)}%;background:#f59e0b"></div></div>
              <span>${(coverage.overlap_ratio * 100).toFixed(1)}%</span>
            </div>
            <div class="coverage-bar-row">
              <span>갭(Gap)</span>
              <div class="bar-track"><div class="bar-fill" style="width:${(coverage.gap_ratio * 100).toFixed(1)}%;background:#ef4444"></div></div>
              <span>${(coverage.gap_ratio * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>`;
  }

  if (balance) {
    const overloaded = balance.overloaded_agents;
    const giniColor = balance.gini_coefficient < 0.3 ? '#10b981' : balance.gini_coefficient < 0.5 ? '#f59e0b' : '#ef4444';

    html += `
        <div class="coverage-card" style="margin-top:16px">
          <div class="coverage-header">
            <span style="font-weight:600">🔧 MCP Tool 균형 분석</span>
            <span class="coverage-badge" style="color:${giniColor}">Gini = ${balance.gini_coefficient.toFixed(3)}</span>
          </div>
          <p style="font-size:13px;color:#64748b;margin:8px 0">${balance.summary}</p>
          <div class="tool-count-grid">
            ${Object.entries(balance.agent_tool_counts).map(([id, cnt]) => {
      const isOver = overloaded.includes(id);
      return `<div class="tool-count-chip ${isOver ? 'overloaded' : ''}">
                  <span>${id}</span><span class="chip-cnt">${cnt}</span>
                </div>`;
    }).join('')}
          </div>
          ${balance.rebalanced ? `<div class="rebalance-badge">♻️ Agile 재분배 완료 (${balance.rebalance_iterations}회 반복)</div>` : ''}
        </div>`;
  }

  panel.innerHTML = html || '<p style="color:#64748b">커버리지 데이터가 없습니다.</p>';

  // Manifold 시각화 iFrame
  if (files && files.query_manifold) {
    document.getElementById('manifoldFrame').src = files.query_manifold;
    document.getElementById('manifoldVizCard').style.display = 'block';
  }
}

// ── Query Routing (Φ: Q → Uk) ─────────────────────────────────
async function routeQuery() {
  if (!currentOutputId) {
    alert('먼저 분석을 실행하세요.');
    return;
  }
  const query = document.getElementById('queryInput').value.trim();
  if (!query) return;

  const resultDiv = document.getElementById('routingResult');
  resultDiv.style.display = 'block';
  resultDiv.innerHTML = '<div class="routing-loading">🧭 라우팅 중...</div>';
  document.getElementById('compositionPanel').style.display = 'none';

  try {
    const res = await fetch('/api/route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ output_id: currentOutputId, query }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(extractErrorMessage(err, '라우팅 실패'));
    }
    const data = await res.json();
    renderRoutingResult(data, query);
  } catch (e) {
    resultDiv.innerHTML = `<div class="warning-card hole"><div class="warning-type" style="color:#ef4444">오류</div><div class="warning-desc">${e.message}</div></div>`;
  }
}

function renderRoutingResult(data, query) {
  const r = data.routing;
  const ambiguousBadge = r.is_ambiguous
    ? `<span class="ambiguous-badge">⚠️ 모호 — ${r.ambiguity_reason}</span>`
    : '';
  const alts = (r.alternatives || []).slice(0, 3).map(a =>
    `<span class="alt-chip">${a.agent_name} (${(a.similarity * 100).toFixed(0)}%)</span>`
  ).join('');

  const confidence = (r.confidence * 100).toFixed(0);
  const confColor = r.confidence >= 0.6 ? '#10b981' : r.confidence >= 0.3 ? '#f59e0b' : '#ef4444';

  document.getElementById('routingResult').innerHTML = `
    <div class="routing-card">
      <div class="routing-query">"${escapeHtml(query)}"</div>
      <div class="routing-arrow">↓ Φ(x) = U<sub>k</sub></div>
      <div class="routing-target">
        <span class="routing-agent">${r.target_agent_name}</span>
        <span class="routing-conf" style="color:${confColor}">신뢰도 ${confidence}%</span>
      </div>
      <div class="routing-class">호모토피 클래스: <code>${r.homotopy_class_id}</code></div>
      ${ambiguousBadge}
      ${alts ? `<div class="routing-alts">대안: ${alts}</div>` : ''}
      <button class="btn btn-outline" style="margin-top:12px" onclick="composeTools('${r.target_agent_id}', '${escapeHtml(query)}')">
        🔧 Tool 합성 계획 보기 (∘)
      </button>
    </div>`;
}

async function composeTools(agentId, query) {
  const compPanel = document.getElementById('compositionPanel');
  const compResult = document.getElementById('compositionResult');
  compPanel.style.display = 'block';
  compResult.innerHTML = '<div class="routing-loading">🔧 Tool 합성 계획 생성 중...</div>';

  try {
    const res = await fetch('/api/compose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ output_id: currentOutputId, query, agent_id: agentId }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(extractErrorMessage(err, '합성 실패'));
    }
    const data = await res.json();
    renderCompositionResult(data);
  } catch (e) {
    compResult.innerHTML = `<div class="warning-card"><div class="warning-desc">${e.message}</div></div>`;
  }
}

function renderCompositionResult(data) {
  const plan = data.composition_plan;
  const math = data.math;
  const steps = (plan.steps || []).sort((a, b) => a.order - b.order);

  const stepsHtml = steps.map(s => `
    <div class="compose-step">
      <div class="compose-order">t<sub>π(${s.order})</sub></div>
      <div class="compose-body">
        <div class="compose-tool">${s.tool_name}()</div>
        <div class="compose-rationale">${s.rationale || ''}</div>
        <div class="compose-flow">${s.input_from} → ${s.output_to}</div>
      </div>
    </div>
  `).join('<div class="compose-arrow">↓ ∘</div>');

  document.getElementById('compositionResult').innerHTML = `
    <div class="compose-card">
      <div class="compose-formula">${math.formula}</div>
      <div class="compose-steps">${stepsHtml}</div>
      ${plan.estimated_output ? `<div class="compose-output">📦 예상 출력: ${plan.estimated_output}</div>` : ''}
      ${!plan.is_valid ? `<div class="compose-error">⚠️ ${(plan.validation_errors || []).join(', ')}</div>` : ''}
    </div>`;
}

function renderAgents(agents, allTools) {
  const grid = document.getElementById('agentsGrid');
  grid.innerHTML = agents.map((a, i) => {
    const color = AGENT_COLORS[i % AGENT_COLORS.length];
    const tasks = a.task_names.map(n => `<div class="task-chip">${n}</div>`).join('');
    const caps = (a.capabilities || []).map(c => `<span class="cap-tag">${c}</span>`).join('');

    // 이 Agent에 할당된 MCP Tool 목록
    const agentTools = (allTools || []).filter(
      t => t.annotations && t.annotations.assigned_agent === a.agent_id
    );
    const toolsHtml = agentTools.length
      ? `<div class="agent-tools-section">
                <div class="agent-tools-label">🔧 담당 MCP Tools (${agentTools.length}개)</div>
                <div class="agent-tools-list">${agentTools.map(t =>
        `<span class="agent-tool-chip">${t.name}()</span>`
      ).join('')}</div>
               </div>`
      : '';

    return `
      <div class="agent-card">
        <div class="agent-header">
          <div class="agent-dot" style="background:${color}"></div>
          <div class="agent-name">${a.name || a.agent_id}</div>
        </div>
        <div class="agent-role">${a.role || ''}</div>
        <div class="agent-tasks">${tasks}</div>
        ${caps ? `<div class="agent-caps">${caps}</div>` : ''}
        ${toolsHtml}
      </div>
    `;
  }).join('');
}

function renderTools(tools, agentNameMap) {
  const list = document.getElementById('toolsList');
  list.innerHTML = tools.map((t, i) => {
    const agentId = t.annotations?.assigned_agent || '';
    // raw agent_id 대신 자연어 이름 표시
    const agentLabel = agentNameMap && agentNameMap[agentId]
      ? agentNameMap[agentId]
      : agentId;
    const schema = JSON.stringify(t, null, 2);
    return `
      <div class="tool-card">
        <div class="tool-header" onclick="toggleTool(${i})">
          <div>
            <div class="tool-name">${t.name}()</div>
            <div class="tool-agent">${agentLabel ? `🤖 ${agentLabel}` : ''}</div>
          </div>
          <span class="tool-toggle" id="toggle-${i}">▼</span>
        </div>
        <div class="tool-body" id="tool-body-${i}">
          <div class="tool-desc">${t.description}</div>
          <div class="code-block">${escapeHtml(schema)}</div>
        </div>
      </div>
    `;
  }).join('');
}

function toggleTool(i) {
  const body = document.getElementById(`tool-body-${i}`);
  const toggle = document.getElementById(`toggle-${i}`);
  const open = body.classList.toggle('open');
  toggle.textContent = open ? '▲' : '▼';
}

function renderWarnings(holes, overlaps) {
  const list = document.getElementById('warningsList');
  let html = '';

  if (!holes.length && !overlaps.length) {
    html = '<div class="warning-card ok"><div class="warning-type" style="color:#10b981">✅ 이상 없음</div><div class="warning-desc">논리적 구멍과 중복 할당이 발견되지 않았습니다.</div></div>';
  }

  holes.forEach(h => {
    html += `
      <div class="warning-card hole">
        <div class="warning-type" style="color:#ef4444">🕳 Hole · ${h.hole_type}</div>
        <div class="warning-desc">${h.description}</div>
        <div class="warning-suggestion">💡 ${h.suggestion}</div>
      </div>
    `;
  });

  overlaps.forEach(o => {
    html += `
      <div class="warning-card overlap">
        <div class="warning-type" style="color:#f59e0b">⚠️ Overlap</div>
        <div class="warning-desc">${o.description}</div>
      </div>
    `;
  });

  list.innerHTML = html;
}

function renderViz(files) {
  if (files.mapper_graph) {
    document.getElementById('mapperFrame').src = files.mapper_graph;
  }
  if (files.feature_radar) {
    document.getElementById('radarFrame').src = files.feature_radar;
  }
}

function renderDownloads(files) {
  const bar = document.getElementById('downloadBar');
  const links = [
    { label: '📄 리포트 (Markdown)', url: files.report_md },
    { label: '📦 결과 (JSON)', url: files.result_json },
    { label: '📊 Mapper 그래프', url: files.mapper_graph },
  ].filter(l => l.url);

  bar.innerHTML = links.map(l =>
    `<a href="${l.url}" class="btn btn-outline" download>${l.label}</a>`
  ).join('');
}

// ── Tabs ─────────────────────────────────────────────────────
function switchTab(name, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(`tab-${name}`).classList.add('active');
}

// ── Utils ────────────────────────────────────────────────────
function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function extractErrorMessage(err, fallback) {
  if (!err) return fallback;
  if (typeof err.detail === 'string') return err.detail;
  if (err.detail && typeof err.detail === 'object') {
    if (typeof err.detail.message === 'string') return err.detail.message;
    return JSON.stringify(err.detail);
  }
  if (typeof err.message === 'string') return err.message;
  return fallback;
}
