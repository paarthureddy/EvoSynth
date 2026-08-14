/* ═══════════════════════════════════════════════════════
   EvoSynth — app.js
   Live dashboard: polls /api/state and renders the
   full Hybrid AutoML process with rich interactions.
═══════════════════════════════════════════════════════ */
'use strict';

const ALGO_COLOR = { GA: '#10b981', PSO: '#00e5ff', DE: '#f59e0b', CMAES: '#8b5cf6' };

// Per-algo fitness history for chart { GA:[{iter,fit},...], ... }
const history = { GA: [], PSO: [], DE: [], CMAES: [] };
let lastEventCount = 0;
let pollTimer = null;
let lastIter = -1;

// ── Slider helpers ────────────────────────────────────────────────────────
function syncSlider(el, lblId)    { document.getElementById(lblId).textContent = el.value; }
function syncSliderPct(el, lblId) { document.getElementById(lblId).textContent = el.value + '%'; }
function syncSliderMs(el, lblId)  { document.getElementById(lblId).textContent = el.value + 'ms'; }

// ── Start run ─────────────────────────────────────────────────────────────
async function startRun() {
  const payload = {
    budget:         parseInt(document.getElementById('cfg-budget').value),
    k_iters:        parseInt(document.getElementById('cfg-kiters').value),
    elite_pct:      parseFloat(document.getElementById('cfg-elite').value) / 100.0,
    inject_pct:     parseFloat(document.getElementById('cfg-inject').value) / 100.0,
    swap_thr:       parseFloat(document.getElementById('cfg-swap').value) / 100.0,
    step_delay:     parseFloat(document.getElementById('cfg-speed').value) / 1000.0,
    evaluator_type: document.getElementById('cfg-evaluator').value,
  };

  Object.keys(history).forEach(k => history[k] = []);
  lastEventCount = 0;
  lastIter = -1;

  try {
    const res  = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.ok) { alert('Error: ' + data.msg); return; }
  } catch {
    alert('Could not reach server. Make sure server.py is running.');
    return;
  }

  document.getElementById('config-overlay').classList.add('hidden');
  document.getElementById('dashboard').classList.remove('hidden');
  initLegend();
  startPolling();
}

// ── Stop run ──────────────────────────────────────────────────────────────
async function stopRun() {
  await fetch('/api/stop', { method: 'POST' }).catch(() => {});
}

// ── Legend ────────────────────────────────────────────────────────────────
function initLegend() {
  document.getElementById('chart-legend').innerHTML =
    Object.entries(ALGO_COLOR).map(([a, c]) =>
      `<div class="leg-item"><div class="leg-dot" style="background:${c};box-shadow:0 0 6px ${c}"></div>${a}</div>`
    ).join('');
}

// ══════════════════════════════════════════════════════
// POLLING
// ══════════════════════════════════════════════════════
function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  poll();
  pollTimer = setInterval(poll, 100);
}

async function poll() {
  try {
    const res = await fetch('/api/state');
    if (!res.ok) throw new Error();
    const s = await res.json();
    render(s);
    setLive(true);
    if (s.phase === 'CONVERGED') {
      clearInterval(pollTimer);
      setLive(false, 'Complete ✓');
      setPhasePill('CONVERGED');
    }
  } catch {
    setLive(false);
  }
}

function setLive(ok, label) {
  const dot = document.getElementById('live-dot');
  const lbl = document.getElementById('live-label');
  dot.className = 'live-dot ' + (ok ? 'active' : 'error');
  lbl.textContent = label || (ok ? 'LIVE' : 'Waiting…');
  lbl.style.color = ok ? '#22c55e' : 'var(--muted)';
}

// ══════════════════════════════════════════════════════
// RENDER
// ══════════════════════════════════════════════════════
function render(s) {
  renderKPIs(s);
  renderProgress(s);
  renderIterDetail(s);
  renderIslands(s);
  renderBestParams(s);
  renderScoreTable(s);
  renderEvents(s);
  updateChartHistory(s);
  drawChart();
}

// ── KPIs ──────────────────────────────────────────────────────────────────
function renderKPIs(s) {
  setPhasePill(s.phase);
  setText('kpi-best',    s.best_fitness ? s.best_fitness.toFixed(2) + '%' : '—');
  setText('kpi-algo',    s.best_algo    || '—');
  setText('kpi-evals',   `${s.evals_used} / ${s.budget}`);
  setText('kpi-iter',    s.iteration    || 0);
  setText('kpi-elapsed', (s.elapsed || 0) + 's');
}

function setPhasePill(phase) {
  const map = {
    INIT: 'INITIALIZATION', T1_WARMUP: 'WARM-UP', SCORING: 'SCORING',
    PARALLEL: 'PARALLEL', CONVERGED: 'CONVERGED ✓', INITIALIZATION: 'INITIALIZATION'
  };
  const cls = phase === 'CONVERGED' ? 'done' : (phase ? 'running' : 'idle');
  document.getElementById('kpi-phase').innerHTML =
    `<span class="phase-pill ${cls}">${map[phase] || phase || 'IDLE'}</span>`;
}

// ── Progress bar ──────────────────────────────────────────────────────────
function renderProgress(s) {
  const pct = s.budget > 0 ? (s.evals_used / s.budget * 100).toFixed(1) : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-text').textContent =
    `${s.evals_used} / ${s.budget}  (${pct}%)`;
}

// ── Iteration Detail Stats ─────────────────────────────────────────────────
function renderIterDetail(s) {
  const el = document.getElementById('iter-detail');
  if (!el) return;

  // Compute per-tier bests
  const t1pops = Object.values(s.t1_islands || {}).flat();
  const t2pops = Object.values(s.t2_islands || {}).flat();
  const t1best = t1pops.length ? Math.max(...t1pops.map(p => p.fitness)) : 0;
  const t2best = t2pops.length ? Math.max(...t2pops.map(p => p.fitness)) : 0;
  const allPop = [...t1pops, ...t2pops];
  const popSize = allPop.length;
  const avgFit  = popSize ? (allPop.reduce((a, b) => a + b.fitness, 0) / popSize).toFixed(2) : '—';

  const newBest = (s.iteration || 0) > lastIter && s.best_fitness;
  if (s.iteration) lastIter = s.iteration;

  el.innerHTML = `
    <div class="iter-stat">
      <div class="iter-stat-label">T1 Best</div>
      <div class="iter-stat-val" style="color:var(--cyan)">${t1best ? t1best.toFixed(2) + '%' : '—'}</div>
    </div>
    <div class="iter-stat">
      <div class="iter-stat-label">T2 Best</div>
      <div class="iter-stat-val" style="color:var(--gold)">${t2best ? t2best.toFixed(2) + '%' : '—'}</div>
    </div>
    <div class="iter-stat">
      <div class="iter-stat-label">Pop Avg</div>
      <div class="iter-stat-val" style="color:var(--green)">${avgFit !== '—' ? avgFit + '%' : '—'}</div>
    </div>`;
}

// ── Island Architecture ───────────────────────────────────────────────────
function renderIslands(s) {
  renderTierRow('t1-islands', s.t1_islands || {});
  renderTierRow('t2-islands', s.t2_islands || {});
}

function renderTierRow(id, islands) {
  const el = document.getElementById(id);
  if (!Object.keys(islands).length) {
    el.innerHTML = `<div class="island-placeholder">Waiting…</div>`;
    return;
  }
  el.innerHTML = Object.entries(islands).map(([alg, pop]) => {
    const color  = ALGO_COLOR[alg] || '#fff';
    const cls    = alg.toLowerCase();
    const best   = pop.length ? Math.max(...pop.map(p => p.fitness)) : 0;
    const worst  = pop.length ? Math.min(...pop.map(p => p.fitness)) : 0;
    const avg    = pop.length ? (pop.reduce((a, p) => a + p.fitness, 0) / pop.length) : 0;
    const spread = (best - worst).toFixed(2);

    // Build dots with rich CSS tooltip
    const dots = pop.slice(0, 15).map((p, rank) => {
      const a = Math.min(1, Math.max(0.2, 0.2 + 0.8 * ((p.fitness - 70) / 30)));
      // Build params lines for tooltip
      const paramLines = p.params
        ? p.params.split(', ').map(kv => {
            const [k, v] = kv.split(': ');
            return `<div class="tt-row"><span class="tt-key">${k}</span><span class="tt-val">${v}</span></div>`;
          }).join('')
        : '';
      return `
        <div class="tooltip-host">
          <div class="idot" style="background:${color};opacity:${a.toFixed(2)}"></div>
          <div class="tt">
            <div style="border-bottom:1px solid rgba(255,255,255,.08);padding-bottom:6px;margin-bottom:6px">
              <span class="tt-fit">${p.fitness.toFixed(4)}%</span>
              <span style="color:var(--muted);margin-left:8px;font-size:9px">Rank #${rank + 1}</span>
            </div>
            <div class="tt-row"><span class="tt-key">Source</span><span class="tt-src">${p.src}</span></div>
            ${paramLines}
          </div>
        </div>`;
    }).join('');

    return `
      <div class="island-card ${cls}">
        <div class="island-name ${cls}">${alg}</div>
        <div class="island-best-val" style="color:${color}">${best.toFixed(2)}%</div>
        <div style="font-size:9px;color:var(--muted);margin-bottom:6px">avg ${avg.toFixed(2)}% · Δ${spread}%</div>
        <div class="island-dots">${dots}</div>
        <div class="island-cnt">${pop.length} individuals</div>
      </div>`;
  }).join('');
}

// ── Best config params ────────────────────────────────────────────────────
function renderBestParams(s) {
  const el = document.getElementById('best-params-grid');
  const p  = s.best_params || {};
  if (!Object.keys(p).length) {
    el.innerHTML = `<div class="param-empty">Searching…</div>`; return;
  }
  el.innerHTML = Object.entries(p).map(([k, v]) => {
    const val = typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(6) : v;
    return `<div class="param-card">
      <div class="param-name">${k.replace(/_/g, ' ')}</div>
      <div class="param-val">${val}</div>
    </div>`;
  }).join('');
}

// ── Composite score table ─────────────────────────────────────────────────
function renderScoreTable(s) {
  const tbody = document.getElementById('score-tbody');
  const comp  = s.composite || {};
  if (!Object.keys(comp).length) {
    tbody.innerHTML = `<tr><td colspan="7" class="tbl-empty">Waiting for first checkpoint…</td></tr>`;
    return;
  }
  const rows = Object.entries(comp).sort((a, b) => b[1].composite - a[1].composite);
  tbody.innerHTML = rows.map(([alg, d], i) => {
    const color = ALGO_COLOR[alg] || '#fff';
    const pct   = Math.round((d.composite || 0) * 100);
    const medal = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `${i + 1}.`;
    return `<tr>
      <td><span style="margin-right:6px">${medal}</span><strong style="color:${color}">${alg}</strong></td>
      <td><span class="tier-pill ${d.tier}">${d.tier}</span></td>
      <td>${(d.accuracy  || 0).toFixed(2)}%</td>
      <td>${(d.speed     || 0).toFixed(3)}</td>
      <td>${(d.diversity || 0).toFixed(3)}</td>
      <td><strong style="color:${color}">${(d.composite || 0).toFixed(3)}</strong></td>
      <td>
        <div class="sbar-wrap">
          <div class="sbar-fill" style="width:${pct}%;background:${color};box-shadow:0 0 6px ${color}40"></div>
        </div>
      </td>
    </tr>`;
  }).join('');
}

// ── Event log ─────────────────────────────────────────────────────────────
function renderEvents(s) {
  const evs = s.events || [];
  if (evs.length === lastEventCount) return;
  lastEventCount = evs.length;
  const el = document.getElementById('event-log');
  el.innerHTML = [...evs].reverse().map(ev =>
    `<div class="ev ${ev.kind}">
       <div class="ev-time">${ev.t}s</div>
       <div class="ev-kind ${ev.kind}">${ev.kind}</div>
       <div class="ev-msg">${ev.msg}</div>
     </div>`
  ).join('');
  setText('event-count', evs.length + ' events');
}

// ── Chart history update ──────────────────────────────────────────────────
function updateChartHistory(s) {
  const comp = s.composite || {};
  const iter = s.iteration || 0;
  Object.entries(comp).forEach(([alg, d]) => {
    if (!history[alg]) history[alg] = [];
    const last = history[alg][history[alg].length - 1];
    if (!last || last.iter !== iter) {
      history[alg].push({ iter, fit: d.accuracy || 0 });
      if (history[alg].length > 500) history[alg].shift();
    }
  });
}

// ── Convergence Chart ─────────────────────────────────────────────────────
function drawChart() {
  const canvas = document.getElementById('chart-canvas');
  if (!canvas) return;
  const W = canvas.offsetWidth || 560;
  const H = canvas.offsetHeight || 230;
  canvas.width  = W * (window.devicePixelRatio || 1);
  canvas.height = H * (window.devicePixelRatio || 1);
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  ctx.clearRect(0, 0, W, H);

  const PAD = { t: 12, r: 16, b: 24, l: 46 };

  // Background
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, 'rgba(0,15,40,.5)');
  grad.addColorStop(1, 'rgba(0,5,15,.1)');
  ctx.fillStyle = grad;
  ctx.roundRect(0, 0, W, H, 10);
  ctx.fill();

  // Grid lines
  const Y_LABELS = [70, 75, 80, 85, 90, 95, 100];
  const allIters = Object.values(history).flatMap(h => h.map(p => p.iter));
  const minI = allIters.length ? Math.min(...allIters) : 0;
  const maxI = allIters.length ? Math.max(...allIters) : 10;
  const rI   = Math.max(1, maxI - minI);
  const toX  = i => PAD.l + ((i - minI) / rI) * (W - PAD.l - PAD.r);
  const toY  = f => (H - PAD.b) - ((f - 70) / 30) * (H - PAD.t - PAD.b);

  Y_LABELS.forEach(v => {
    const y = toY(v);
    ctx.strokeStyle = 'rgba(255,255,255,.04)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(W - PAD.r, y); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,.25)';
    ctx.font = '9px JetBrains Mono';
    ctx.textAlign = 'right';
    ctx.fillText(v + '%', PAD.l - 6, y + 3);
  });

  // X axis label
  ctx.fillStyle = 'rgba(255,255,255,.25)';
  ctx.font = '9px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('Iteration →', W / 2, H - 2);

  // Draw lines per algo
  Object.entries(history).forEach(([alg, pts]) => {
    if (pts.length < 1) return;
    const color = ALGO_COLOR[alg] || '#fff';

    if (pts.length === 1) {
      // Single point
      ctx.beginPath();
      ctx.arc(toX(pts[0].iter), toY(pts[0].fit), 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      return;
    }

    // Gradient fill under line
    const fillGrad = ctx.createLinearGradient(0, toY(100), 0, toY(70));
    fillGrad.addColorStop(0, color + '30');
    fillGrad.addColorStop(1, color + '00');
    ctx.beginPath();
    ctx.moveTo(toX(pts[0].iter), H - PAD.b);
    pts.forEach(p => ctx.lineTo(toX(p.iter), toY(p.fit)));
    ctx.lineTo(toX(pts[pts.length - 1].iter), H - PAD.b);
    ctx.closePath();
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Main line with glow
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    ctx.lineJoin    = 'round';
    ctx.lineCap     = 'round';
    ctx.shadowColor = color;
    ctx.shadowBlur  = 10;
    pts.forEach((p, i) =>
      i === 0 ? ctx.moveTo(toX(p.iter), toY(p.fit)) : ctx.lineTo(toX(p.iter), toY(p.fit))
    );
    ctx.stroke();
    ctx.shadowBlur = 0;

    // End-point dot
    const last = pts[pts.length - 1];
    ctx.beginPath();
    ctx.arc(toX(last.iter), toY(last.fit), 5, 0, Math.PI * 2);
    ctx.fillStyle   = color;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 14;
    ctx.fill();
    ctx.shadowBlur  = 0;

    // Label near end point
    ctx.fillStyle   = color;
    ctx.font        = 'bold 10px Inter';
    ctx.textAlign   = 'left';
    ctx.fillText(`${alg} ${last.fit.toFixed(1)}%`, toX(last.iter) + 8, toY(last.fit) + 4);
  });
}

// ── Util ──────────────────────────────────────────────────────────────────
function setText(id, v) { const e = document.getElementById(id); if (e) e.textContent = v; }

// ── Resize chart ──────────────────────────────────────────────────────────
window.addEventListener('resize', drawChart);
