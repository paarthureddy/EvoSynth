import { useState, useEffect, useRef } from "react";

function App() {
  const [logs, setLogs] = useState([]);
  const [step, setStep] = useState({ num: 0, name: "Initializing..." });
  const [progress, setProgress] = useState({ evals: 0, budget: 2000 });
  const [tierState, setTierState] = useState({ T1: [], T2: [] });
  const [metrics, setMetrics] = useState({});
  const [switches, setSwitches] = useState([]);

  const terminalRef = useRef(null);

  useEffect(() => {
    const eventSource = new EventSource("http://localhost:8000/stream");

    eventSource.onmessage = (event) => {
      const parsed = JSON.parse(event.data);

      switch (parsed.type) {
        case "log":
          setLogs((prev) => [...prev, parsed.data.text]);
          break;
        case "step":
          setStep(parsed.data);
          break;
        case "progress":
          setProgress(parsed.data);
          break;
        case "tier_state":
          setTierState(parsed.data);
          break;
        case "metric":
          setMetrics((prev) => ({
            ...prev,
            [parsed.data.algo]: parsed.data.accuracy,
          }));
          break;
        case "switch":
          setSwitches((prev) => [parsed.data, ...prev].slice(0, 50));
          break;
        default:
          break;
      }
    };

    return () => eventSource.close();
  }, []);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  const progressPercent = Math.min(
    100,
    (progress.evals / progress.budget) * 100,
  );

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-6 font-sans selection:bg-indigo-500/30">
      <div className="max-w-7xl mx-auto space-y-6">
        <header className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col md:flex-row justify-between items-start md:items-center shadow-xl">
          <div>
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-linear-to-r from-indigo-400 to-purple-400">
              EvoSynth Dynamic Optimizer
            </h1>
            <p className="text-sm text-slate-400 mt-1 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              Step {step.num}: {step.name}
            </p>
          </div>

          <div className="mt-4 md:mt-0 w-full md:w-72">
            <div className="flex justify-between text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">
              <span>Evaluations</span>
              <span>
                {progress.evals} / {progress.budget}
              </span>
            </div>
            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-linear-to-r from-indigo-500 to-purple-500 transition-all duration-300 ease-out"
                style={{ width: `${progressPercent}%` }}
              ></div>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
              <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 pb-2 border-b border-slate-800">
                Tier Architecture
              </h2>

              <div className="space-y-4">
                <div>
                  <div className="text-xs font-semibold text-amber-500 mb-2">
                    TIER 1 (Active)
                  </div>
                  <div className="flex flex-wrap gap-2 min-h-12 p-3 rounded-xl bg-slate-950/50 border border-slate-800/50">
                    {tierState.T1.map((algo, i) => (
                      <span
                        key={i}
                        className="px-3 py-1 text-xs font-bold rounded-md bg-amber-500/10 text-amber-400 border border-amber-500/20 shadow-[0_0_10px_rgba(245,158,11,0.1)]"
                      >
                        {algo}
                      </span>
                    ))}
                    {tierState.T1.length === 0 && (
                      <span className="text-xs text-slate-600 italic m-auto">
                        Empty
                      </span>
                    )}
                  </div>
                </div>

                <div>
                  <div className="text-xs font-semibold text-purple-500 mb-2">
                    TIER 2 (Background)
                  </div>
                  <div className="flex flex-wrap gap-2 min-h-12 p-3 rounded-xl bg-slate-950/50 border border-slate-800/50">
                    {tierState.T2.map((algo, i) => (
                      <span
                        key={i}
                        className="px-3 py-1 text-xs font-bold rounded-md bg-purple-500/10 text-purple-400 border border-purple-500/20 shadow-[0_0_10px_rgba(168,85,247,0.1)]"
                      >
                        {algo}
                      </span>
                    ))}
                    {tierState.T2.length === 0 && (
                      <span className="text-xs text-slate-600 italic m-auto">
                        Empty
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
              <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 pb-2 border-b border-slate-800">
                Live Best Accuracy
              </h2>
              <div className="space-y-3">
                {Object.keys(metrics).length === 0 ? (
                  <div className="text-xs text-slate-600 italic text-center py-4">
                    Waiting for evaluations...
                  </div>
                ) : (
                  Object.entries(metrics)
                    .sort((a, b) => b[1] - a[1])
                    .map(([algo, acc], idx) => (
                      <div
                        key={algo}
                        className="flex items-center justify-between p-3 rounded-xl bg-slate-950/50 border border-slate-800/50"
                      >
                        <div className="flex items-center gap-3">
                          <span
                            className={`text-xs font-bold ${idx === 0 ? "text-emerald-400" : "text-slate-400"}`}
                          >
                            {idx === 0 ? "🏆" : `#${idx + 1}`}
                          </span>
                          <span className="text-sm font-semibold text-slate-200">
                            {algo}
                          </span>
                        </div>
                        <span className="text-sm font-mono text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded-md border border-emerald-400/20">
                          {acc.toFixed(4)}%
                        </span>
                      </div>
                    ))
                )}
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col h-64">
              <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 pb-2 border-b border-slate-800 shrink-0">
                UCB1 Bandit Switch Feed
              </h2>
              <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin scrollbar-thumb-slate-700">
                {switches.length === 0 ? (
                  <div className="text-xs text-slate-600 italic flex items-center justify-center h-full">
                    No switching events yet.
                  </div>
                ) : (
                  switches.map((sw, i) => (
                    <div
                      key={i}
                      className="flex items-start gap-3 p-3 rounded-xl bg-slate-950/50 border border-slate-800/50 transition-all hover:bg-slate-900"
                    >
                      <span className="mt-0.5 px-2 py-1 text-[10px] font-bold uppercase rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/20">
                        {sw.action}
                      </span>
                      <div>
                        <strong className="text-sm text-slate-200 block">
                          {sw.algo}
                        </strong>
                        <span className="text-xs text-slate-400">
                          {sw.details}
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="bg-[#0D1117] border border-slate-800 rounded-2xl overflow-hidden shadow-lg h-96 flex flex-col">
              <div className="bg-slate-900 border-b border-slate-800 px-4 py-2 flex items-center gap-2 shrink-0">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
                  <div className="w-3 h-3 rounded-full bg-amber-500/80"></div>
                  <div className="w-3 h-3 rounded-full bg-emerald-500/80"></div>
                </div>
                <span className="ml-2 text-xs font-mono text-slate-500">
                  evosynth_execution.log
                </span>
              </div>
              <div
                ref={terminalRef}
                className="p-4 overflow-y-auto font-mono text-[13px] leading-relaxed text-slate-300 flex-1 whitespace-pre-wrap scrollbar-thin scrollbar-thumb-slate-700"
              >
                {logs.length === 0 ? (
                  <span className="text-slate-600">
                    Waiting for engine start...
                  </span>
                ) : (
                  logs.join("")
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
