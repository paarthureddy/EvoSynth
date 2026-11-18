# Server-Sent Events (SSE) Protocol

## Architecture

EvoSynth utilizes standard HTTP **Server-Sent Events (SSE)** rather than WebSockets because telemetry data is primarily unidirectional (Python optimizer $\to$ React UI). SSE offers automatic reconnection, low overhead, and straightforward firewall traversal.

```text
  Python Optimizer ────POST /event────► FastAPI Telemetry ────SSE Stream────► React Dashboard
 (controller.py)                         (api_server.py)                     (EventSource Hook)
```

---

## Event Types & Payloads

The FastAPI endpoint (`GET /stream`) broadcasts text-formatted event streams:

### 1. `metric`
Triggered whenever any algorithm achieves a new best fitness score:
```json
{
  "type": "metric",
  "data": {
    "algo": "PSO",
    "accuracy": 94.67
  }
}
```

### 2. `composite`
Broadcast when the UCB1 Multi-Armed Bandit updates algorithm rankings:
```json
{
  "type": "composite",
  "data": {
    "algo": "T1_PSO",
    "composite": 0.721,
    "diversity": 0.337
  }
}
```

### 3. `tier_state`
Broadcast when algorithms are promoted or demoted between tiers:
```json
{
  "type": "tier_state",
  "data": {
    "T1": ["DE", "GA"],
    "T2": ["PSO", "CMAES"]
  }
}
```

### 4. `switch`
Records critical meta-controller intervention events:
```json
{
  "type": "switch",
  "data": {
    "action": "Injection",
    "algo": "T2_GA",
    "details": "Replaced bottom 20% due to stagnation"
  }
}
```
