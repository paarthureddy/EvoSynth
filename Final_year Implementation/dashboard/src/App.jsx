import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [logs, setLogs] = useState([]);
  const [step, setStep] = useState({ num: 0, name: "Initializing..." });
  const [progress, setProgress] = useState({ evals: 0, budget: 2000 });
  const [tierState, setTierState] = useState({ T1: [], T2: [] });
  const [metrics, setMetrics] = useState({});
  const [switches, setSwitches] = useState([]);
  
  const terminalRef = useRef(null);

  useEffect(() => {
    const eventSource = new EventSource('http://localhost:8000/stream');

    eventSource.onmessage = (event) => {
      const parsed = JSON.parse(event.data);
      
      switch (parsed.type) {
        case 'log':
          setLogs(prev => [...prev, parsed.data.text]);
          break;
        case 'step':
          setStep(parsed.data);
          break;
        case 'progress':
          setProgress(parsed.data);
          break;
        case 'tier_state':
          setTierState(parsed.data);
          break;
        case 'metric':
          setMetrics(prev => ({
            ...prev,
            [parsed.data.algo]: parsed.data.accuracy
          }));
          break;
        case 'switch':
          setSwitches(prev => [parsed.data, ...prev].slice(0, 50));
          break;
        default:
          break;
      }
    };

    return () => eventSource.close();
  }, []);

  // Auto-scroll terminal
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  const progressPercent = Math.min(100, (progress.evals / progress.budget) * 100);

  return (
    <div className="dashboard-container">
      <header>
        <div className="title">
          <h1>Dynamic Evolutionary Optimizer</h1>
          <p>Step {step.num}: {step.name}</p>
        </div>
        <div className="progress-container">
          <div>Evaluations: {progress.evals} / {progress.budget}</div>
          <div className="progress-bar-bg">
            <div className="progress-bar-fill" style={{ width: `${progressPercent}%` }}></div>
          </div>
        </div>
      </header>

      <div className="grid-main">
        <div className="panel">
          <h2>Tier Architecture</h2>
          <div className="tier-row">
            <div className="tier-label">Tier 2</div>
            <div className="tier-box">
              {tierState.T2.map((algo, i) => (
                <div key={i} className="algo-tag t2">{algo}</div>
              ))}
              {tierState.T2.length === 0 && <span style={{color: '#9ca3af', fontStyle: 'italic', alignSelf: 'center'}}>Empty</span>}
            </div>
          </div>
          <div className="tier-row">
            <div className="tier-label">Tier 1</div>
            <div className="tier-box">
              {tierState.T1.map((algo, i) => (
                <div key={i} className="algo-tag t1">{algo}</div>
              ))}
              {tierState.T1.length === 0 && <span style={{color: '#9ca3af', fontStyle: 'italic', alignSelf: 'center'}}>Empty</span>}
            </div>
          </div>

          <h2 style={{marginTop: '30px'}}>Live Best Accuracy</h2>
          <div>
            {Object.entries(metrics).sort((a, b) => b[1] - a[1]).map(([algo, acc]) => (
              <div key={algo} className="metric-row">
                <strong>{algo}</strong>
                <span>{acc.toFixed(4)}%</span>
              </div>
            ))}
            {Object.keys(metrics).length === 0 && <div style={{color: '#9ca3af', fontStyle: 'italic'}}>Waiting for evaluations...</div>}
          </div>
        </div>

        <div className="panel">
          <h2>Switching & Strategy Events</h2>
          <div className="switch-feed">
            {switches.map((sw, i) => (
              <div key={i} className="switch-event">
                <strong>[{sw.action}]</strong> {sw.algo}
                <div style={{color: '#6b7280', fontSize: '12px', marginTop: '4px'}}>{sw.details}</div>
              </div>
            ))}
            {switches.length === 0 && <div style={{color: '#9ca3af', fontStyle: 'italic'}}>No switching events yet. Will trigger during Tier 2 loop.</div>}
          </div>
        </div>
      </div>

      <div className="panel" style={{padding: 0, overflow: 'hidden'}}>
        <div style={{padding: '10px 15px', background: '#e5e7eb', fontWeight: 600, fontSize: '14px', borderBottom: '1px solid #d1d5db'}}>Terminal Output</div>
        <div className="terminal" ref={terminalRef}>
          {logs.join('')}
        </div>
      </div>
    </div>
  );
}

export default App;
