/* ═══════════════════════════════════════════════════════
   EvoSynth — app.js
   Polls /api/status + state.json and renders the
   full PDF-spec Hybrid AutoML process live.
═══════════════════════════════════════════════════════ */
'use strict';

const ALGO_COLOR = { GA:'#34d399', PSO:'#00f0ff', DE:'#fbbf24', CMAES:'#a78bfa' };

// Per-algo fitness history for chart  { GA:[{iter,fit},...], ... }
const history = { GA:[], PSO:[], DE:[], CMAES:[] };
let lastEventCount = 0;
let pollTimer = null;

// ── Slider helpers (called from HTML) ─────────────────────────────────────
function syncSlider(el, lblId) {
  document.getElementById(lblId).textContent = el.value;
}
function syncSliderPct(el, lblId) {
  document.getElementById(lblId).textContent = el.value + '%';
}
function syncSliderMs(el, lblId) {
  document.getElementById(lblId).textContent = el.value + 'ms';
}

// ── Start run ─────────────────────────────────────────────────────────────
async function startRun() {
  const payload = {
    budget:     parseInt(document.getElementById('cfg-budget').value),
    k_iters:    parseInt(document.getElementById('cfg-kiters').value),
    elite_pct:  parseFloat(document.getElementById('cfg-elite').value) / 100.0,
    inject_pct: parseFloat(document.getElementById('cfg-inject').value) / 100.0,
    swap_thr:   parseFloat(document.getElementById('cfg-swap').value) / 100.0,
    step_delay: parseFloat(document.getElementById('cfg-speed').value) / 1000.0,
    evaluator_type: document.getElementById('cfg-evaluator').value
  };

  // Reset history
  Object.keys(history).forEach(k => history[k] = []);
  lastEventCount = 0;

  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.ok) { alert('Error: ' + data.msg); return; }
  } catch(e) {
    alert('Could not reach server. Make sure server.py is running.');
    return;
  }

  // Show dashboard, hide config
  document.getElementById('config-overlay').classList.add('hidden');
  document.getElementById('dashboard').classList.remove('hidden');
  initLegend();
  startPolling();
}

// ── Stop run ──────────────────────────────────────────────────────────────
async function stopRun() {
  await fetch('/api/stop', { method:'POST' }).catch(()=>{});
}

// ── Init chart legend ─────────────────────────────────────────────────────
function initLegend() {
  document.getElementById('chart-legend').innerHTML =
    Object.entries(ALGO_COLOR).map(([a,c]) =>
      `<div class="leg-item"><div class="leg-dot" style="background:${c}"></div>${a}</div>`
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
    const res = await fetch(`/api/state`);
    if (!res.ok) throw new Error();
    const s = await res.json();
    render(s);
    setLive(true);

    // Stop polling once converged
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
  setText('kpi-best',    s.best_fitness ? s.best_fitness.toFixed(2)+'%' : '—');
  setText('kpi-algo',    s.best_algo   || '—');
  setText('kpi-evals',   `${s.evals_used} / ${s.budget}`);
  setText('kpi-iter',    s.iteration   || 0);
  setText('kpi-elapsed', (s.elapsed||0)+'s');
}

function setPhasePill(phase) {
  const map = { INIT:'INIT', T1_WARMUP:'WARM-UP', SCORING:'SCORING',
                PARALLEL:'PARALLEL', CONVERGED:'CONVERGED ✓' };
  const cls = phase === 'CONVERGED' ? 'done' : (phase ? 'running' : 'idle');
  document.getElementById('kpi-phase').innerHTML =
    `<span class="phase-pill ${cls}">${map[phase]||phase||'IDLE'}</span>`;
}

// ── Progress ──────────────────────────────────────────────────────────────
function renderProgress(s) {
  const pct = s.budget > 0 ? (s.evals_used/s.budget*100).toFixed(1) : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-text').textContent =
    `${s.evals_used} / ${s.budget}  (${pct}%)`;
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
    const color = ALGO_COLOR[alg] || '#fff';
    const best  = pop.length ? Math.max(...pop.map(p=>p.fitness)) : 0;
    const cls   = alg.toLowerCase();
    const dots  = pop.slice(0,15).map(p => {
      const a = Math.min(1, Math.max(0.15, 0.15 + 0.85*((p.fitness-70)/25)));
      const tooltip = `Fitness: ${p.fitness.toFixed(2)}%\nSource: ${p.src}\nParams: ${p.params || ''}`;
      return `<div class="idot" style="background:${color};opacity:${a.toFixed(2)}" title="${tooltip}"></div>`;
    }).join('');
    return `<div class="island-card">
      <div class="island-name ${cls}">${alg}</div>
      <div class="island-best-val" style="color:${color}">${best.toFixed(1)}%</div>
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
  el.innerHTML = Object.entries(p).map(([k,v]) => {
    const val = typeof v==='number' && !Number.isInteger(v) ? v.toFixed(6) : v;
    return `<div class="param-card">
      <div class="param-name">${k.replace(/_/g,' ')}</div>
      <div class="param-val">${val}</div>
    </div>`;
  }).join('');
}

// ── Composite score table ─────────────────────────────────────────────────
function renderScoreTable(s) {
  const tbody = document.getElementById('score-tbody');
  const comp  = s.composite || {};
  if (!Object.keys(comp).length) {
    tbody.innerHTML=`<tr><td colspan="7" class="tbl-empty">Waiting for first checkpoint…</td></tr>`;
    return;
  }
  const rows = Object.entries(comp).sort((a,b)=>b[1].composite-a[1].composite);
  tbody.innerHTML = rows.map(([alg,d]) => {
    const color = ALGO_COLOR[alg]||'#fff';
    const pct   = Math.round((d.composite||0)*100);
    return `<tr>
      <td style="color:${color};font-weight:700">${alg}</td>
      <td><span class="tier-pill ${d.tier}">${d.tier}</span></td>
      <td>${(d.accuracy||0).toFixed(2)}%</td>
      <td>${(d.speed||0).toFixed(3)}</td>
      <td>${(d.diversity||0).toFixed(3)}</td>
      <td style="color:${color};font-weight:700">${(d.composite||0).toFixed(3)}</td>
      <td><div class="sbar-wrap"><div class="sbar-fill" style="width:${pct}%;background:${color}"></div></div></td>
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
  Object.entries(comp).forEach(([alg,d]) => {
    if (!history[alg]) history[alg] = [];
    const last = history[alg][history[alg].length-1];
    if (!last || last.iter !== iter) {
      history[alg].push({ iter, fit: d.accuracy||0 });
      if (history[alg].length > 500) history[alg].shift();
    }
  });
}

// ── Convergence Chart ─────────────────────────────────────────────────────
function drawChart() {
  const canvas = document.getElementById('chart-canvas');
  if (!canvas) return;
  const W = canvas.offsetWidth || 560;
  const H = canvas.offsetHeight || 220;
  canvas.width  = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,W,H);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,.04)';
  ctx.lineWidth = 1;
  for(let x=0;x<=W;x+=W/8){ ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke(); }
  for(let y=0;y<=H;y+=H/5){ ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke(); }

  // Axis labels
  ctx.fillStyle='rgba(255,255,255,.18)';
  ctx.font='9px JetBrains Mono';
  ctx.fillText('100%',2,10);
  ctx.fillText(' 70%',2,H-4);
  ctx.fillText('iteration →',W-74,H-4);

  const allIters = Object.values(history).flatMap(h=>h.map(p=>p.iter));
  const minI = allIters.length ? Math.min(...allIters) : 0;
  const maxI = allIters.length ? Math.max(...allIters) : 1;
  const rI   = Math.max(1, maxI-minI);

  const PAD = 6;
  const toX = i => PAD + ((i-minI)/rI)*(W-2*PAD);
  const toY = f => H-PAD - ((f-70)/30)*(H-2*PAD);

  Object.entries(history).forEach(([alg,pts]) => {
    if (pts.length < 2) return;
    const color = ALGO_COLOR[alg]||'#fff';
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 5;
    pts.forEach((p,i) => {
      i===0 ? ctx.moveTo(toX(p.iter),toY(p.fit))
             : ctx.lineTo(toX(p.iter),toY(p.fit));
    });
    ctx.stroke();
    ctx.shadowBlur=0;

    // Last point
    const last=pts[pts.length-1];
    ctx.beginPath();
    ctx.arc(toX(last.iter),toY(last.fit),4,0,Math.PI*2);
    ctx.fillStyle=color;
    ctx.fill();

    // Label
    ctx.fillStyle=color;
    ctx.font='bold 10px Inter';
    ctx.fillText(`${alg} ${last.fit.toFixed(1)}%`, toX(last.iter)+6, toY(last.fit)+3);
  });
}

// ── Util ──────────────────────────────────────────────────────────────────
function setText(id,v){const e=document.getElementById(id);if(e)e.textContent=v;}

// ── Resize chart ──────────────────────────────────────────────────────────
window.addEventListener('resize', drawChart);
