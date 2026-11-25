# 06 — Telemetry API, Live Dashboard & Documentation Site

EvoSynth's "glass-box" observability works in three parts:

```
DynamicOptimizer (controller.py)            FastAPI relay (telemetry/server.py)          React dashboard (dashboard/)
  EventReporter.emit(type, data) ──HTTP POST /event──►  fan-out to client queues ──SSE GET /stream──► EventSource → React state
  LoggerWriter (stdout mirror) ──"log" events────┘
```

SSE (server-sent events) was chosen over WebSockets because the traffic is one-way, from optimizer to UI. SSE also reconnects automatically and is simple to run.

---

## 1. Engine side: `EventReporter` and `LoggerWriter` (`src/evosynth/telemetry/`)

- **`EventReporter`** (`reporter.py`) is created by `RunContext` when a run starts (or lazily by `get_reporter()`), and the controller uses the active one.
  - It probes `GET http://localhost:8000/docs` with a 0.5 s timeout (`[telemetry]` settings) and sets `enabled = True` only if the probe succeeds.
  - Live telemetry is therefore **off** unless the API server is already running when the run starts.
- **`emit(event_type, payload)`** always appends the event, with a timestamp, to the run's `events.jsonl` (except `log` events). When enabled, it also starts a daemon thread that POSTs `{"type": event_type, "data": payload}` to `http://localhost:8000/event` with a 1 s timeout. Failures are swallowed, so telemetry can never slow down or crash a run.
- **`LoggerWriter`** (`run_logger.py`) replaces `sys.stdout` for the duration of a run. Each `write`:
  - echoes to the real console;
  - appends to the run folder's `run.log`, flushing every time so the log survives crashes and Ctrl-C;
  - emits a `log` event when the text is non-empty and is not just `"\n"`.

## 2. Event catalogue

| `type` | `data` payload | Emitted when |
|---|---|---|
| `log` | `{text}` | Any printed output |
| `step` | `{num, name}` | Start of each Step 0–6. Names: "Search Space Definition", "Initial Population (N=100)" (literal text, not the actual N), "Tier Split (4-Way)", "Tier 1 Competition (n iters)", "Composite Scoring + Top-2 Promotion", "Tier 2 [...] + Tier 1 [...]", "Final Results". |
| `progress` | `{evals, budget}` | After each init chunk and each T1/T2 iteration |
| `tier_state` | `{T1: [...], T2: [...]}` | After the split (`T2: ["(Reserved)", "(Reserved)"]`), and after promotion, SWAP and rollback |
| `metric` | `{algo, accuracy}` | A family's best fitness improves. `algo` is the bare family name, e.g. `PSO`. |
| `composite` | `{algo, composite, diversity}` | Every printed scoring table, once per island. Note that `composite` here carries the **UCB score**, and `algo` has a tier prefix, e.g. `T1_PSO`. |
| `switch` | `{action, algo, details}` | Meta-controller interventions: `Promotion`, `Swap`, `Rollback`, `Injection`, `Elite Sharing` |

Example `switch` payloads emitted by the code:

```json
{"action":"Promotion","algo":"System","details":"Promoted ['CMAES', 'GA'] to Tier 2"}
{"action":"Swap","algo":"PSO <-> GA","details":"Demoting T2 algorithm due to low composite score"}
{"action":"Rollback","algo":"PSO -> GA","details":"Promoted algorithm failed grace period"}
{"action":"Injection","algo":"T2_DE","details":"Replaced bottom 4 due to stagnation"}
{"action":"Elite Sharing","algo":"Tier 2 Only","details":"Swapped top individuals between T2 islands"}
```

## 3. FastAPI relay (`src/evosynth/telemetry/server.py`; command `evosynth serve`)

```python
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
clients = set()                                  # one asyncio.Queue per connected browser

@app.post("/event")                              # engine → server
async def receive_event(request):
    data = await request.json()
    for q in list(clients): await q.put(data)    # broadcast to every client
    return {"status": "ok"}

@app.get("/stream")                              # server → browser (text/event-stream)
async def stream():
    queue = asyncio.Queue(); clients.add(queue)
    # yields f"data: {json.dumps(event)}\n\n" for each event; removes queue on disconnect
```

- Start it with `evosynth serve` (`--host` / `--port` override `[telemetry]`). This runs uvicorn on `0.0.0.0:8000` with log level warning.
- FastAPI's own `/docs` page doubles as the engine's liveness probe.
- **Behaviour:**
  - The server is stateless and in-memory.
  - There is no history replay: a browser that connects late sees only future events.
  - There is no authentication.
  - CORS is fully open, which is acceptable for local demos.

## 4. React dashboard (`dashboard/`)

**Stack:**
- React 19, Vite 8 and Tailwind CSS 4 (through `@tailwindcss/vite`);
- `oxlint` for linting;
- npm scripts `dev`, `build`, `lint` and `preview`.

The entire UI is in `src/App.jsx`. `src/main.jsx` mounts it and `index.css` imports Tailwind. The dev server runs on `http://localhost:5173`.

**State and wiring:**

```js
const es = new EventSource("http://localhost:8000/stream");
es.onmessage = e => { const {type, data} = JSON.parse(e.data); switch (type) {
  case "log":        setLogs(prev => [...prev, data.text]);                 // terminal mirror
  case "step":       setStep(data);                                         // {num, name}
  case "progress":   setProgress(data);                                     // {evals, budget}
  case "tier_state": setTierState(data);                                    // {T1, T2}
  case "metric":     setMetrics(prev => ({...prev, [data.algo]: data.accuracy}));
  case "switch":     setSwitches(prev => [data, ...prev].slice(0, 50));     // newest first, max 50
}};
```

The `composite` event is received but has no handler, so it is ignored. The initial progress budget placeholder is 2000.

**Layout** (dark slate theme):

| Panel | Shows | How to read it |
|---|---|---|
| Header | "EvoSynth Dynamic Optimizer", a pulsing dot, "Step n: name", and an **Evaluations** progress bar (`evals / budget`) | Where the run is in the 7-stage workflow and how much budget is spent |
| **Tier Architecture** | Chips for **TIER 1 (Active)** in amber and **TIER 2 (Background)** in purple | Which optimizers currently hold which tier. Chips change on promotion, SWAP and rollback. |
| **Live Best Accuracy** | A leaderboard of each family's best accuracy, sorted, with a trophy on the leader | Which family has found the best configuration so far |
| **UCB1 Bandit Switch Feed** | Cards for each `switch` event: action badge, algorithm, details | The audit trail of every meta-controller decision |
| **Terminal** (`evosynth_execution.log`) | The full stdout mirror, auto-scrolled | The same tables and lines as the console, including the scoring tables with diversity, composite, bonus and UCB |

**Reading the metrics** (from the project's "interpreting data" guide):
- **Diversity:**
  - the normal range is about 0.2–0.7;
  - values below 0.1 mean the island has collapsed onto one basin, which is the condition for injection;
  - high values are welcome in Tier 1 but should fall in Tier 2.
- **Speed / trend:** a positive value means the island is still improving; 0 means a plateau.
- **UCB score = composite + bonus.** The composite score reflects empirical performance. The bonus favours families that have consumed fewer evaluations.

**To run:** start `evosynth serve` first, then `cd dashboard && npm install && npm run dev`, then start an experiment (e.g. `evosynth run`). See [12-setup-and-usage.md](12-setup-and-usage.md).

## 5. Documentation site (`docs/`)

- **Tool:** VitePress, with `config.mts` set to `base: "/EvoSynth/"`, title "EvoSynth", a logo, a custom theme and `style.css`.
- **Navigation:** Guide, Algorithms, API Reference, Comparative Analysis.
- **Sidebar sections and pages:**

| Section | Pages |
|---|---|
| Getting Started | Introduction, System Architecture, Installation |
| Core Concepts | Meta-Optimizer (UCB1), Universal Latent Space, Cooperative Co-evolution, Composite Scoring |
| Optimization Algorithms | Genetic Algorithm, Particle Swarm, Differential Evolution, CMA-ES |
| API Reference | Controller, Evaluator, Search Space |
| Dashboard | Overview, SSE Protocol, Interpreting Data |
| Developer | Adding an Optimizer, Custom Evaluator |
| Standalone | Comparative Analysis (`research/comparative_analysis.md`) |

- **Local build:** `cd docs && npm install && npm run docs:dev`.
- **Deployment:** `.github/workflows/deploy-docs.yml` runs on pushes to `main` or `PostReview2` that touch `docs/**` (or on manual dispatch). It installs Node 20, builds VitePress and deploys to GitHub Pages. The public docs link is `https://evosynth-2027.github.io/evosynth-docs/`.
- **Long-form documents in `docs/research/`:**
  - `comparative_analysis.md` (published as the Comparative Analysis page);
  - `PANEL_DEFENSE_MASTER_GUIDE.md` (excluded from the site);
  - `AutoML_Relevant_Papers.txt`.
- **This handbook** (`docs/handbook/`) is excluded from the site (`srcExclude` in `config.mts`).

  Their content is covered in files 09–11 of this folder.
