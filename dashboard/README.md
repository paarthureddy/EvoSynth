# EvoSynth Live Telemetry Dashboard

The **EvoSynth Dashboard** is a real-time, glass-box telemetry interface built with **React**, **Vite**, and **Tailwind CSS**. It connects to the FastAPI backend via **Server-Sent Events (SSE)** to provide instant visual observability into the Multi-Armed Bandit's decisions, algorithm migrations, diversity metrics, and accuracy progressions.

---

## 🖥️ Key Features

1. **Glass-Box Meta-Control Observability:**
   * Watch the UCB1 Multi-Armed Bandit dynamically promote algorithms from **Tier 1 (Exploration)** to **Tier 2 (Exploitation)** in real-time.
   * View live composite scores breaking down Fitness, Velocity, Population Diversity, and Trend.

2. **Real-Time Telemetry Streaming:**
   * Utilizes lightweight Server-Sent Events (SSE) streaming at `http://localhost:8000/stream`.
   * Automatically streams log messages, fitness leaps, island tier changes, and algorithm switches without page refreshes.

3. **Multi-Model & Architecture Telemetry:**
   * Tracks model switches across XGBoost, LightGBM, and Random Forest during optimization.
   * Visualizes best-so-far hyperparameter configurations and accuracy convergence curves.

---

## 🛠️ Technology Stack

* **Framework:** React 18
* **Build Tool:** Vite
* **Styling:** Tailwind CSS & Lucide Icons
* **State & Networking:** Server-Sent Events (SSE) native `EventSource` hook
* **Charts:** Recharts / Custom dynamic SVG visualizations

---

## 🚀 Running the Dashboard

### 1. Ensure the Backend Telemetry Server is Running

In the root `EvoSynth/` directory:
```bash
python api_server.py
```
This boots the FastAPI SSE server at `http://localhost:8000`.

### 2. Launch the Frontend Development Server

Navigate to the dashboard directory:
```bash
cd dashboard
npm install
npm run dev
```

The application will start at:
```text
Local:   http://localhost:5173/
```

Open this URL in your web browser. Once the optimization engine begins running, the dashboard will immediately illuminate with live metrics, algorithm rankings, and terminal logs.

---

## 📡 SSE Event Schema

The FastAPI server emits JSON-encoded payloads over the SSE channel:

| Event Type | Payload Attributes | Description |
| :--- | :--- | :--- |
| `metric` | `{ algo: string, accuracy: number }` | Emitted whenever an algorithm sets a new best fitness score. |
| `composite` | `{ algo: string, composite: number, diversity: number }` | Emitted when UCB1 and composite scores are calculated. |
| `tier_state` | `{ T1: string[], T2: string[] }` | Emitted when algorithms are promoted, demoted, or rolled back. |
| `progress` | `{ evals: number, budget: number }` | Tracks progress against the total evaluation budget. |
| `switch` | `{ action: string, algo: string, details: string }` | Records elite migration, random injection, or algorithm rollback. |
| `log` | `{ text: string }` | Live terminal stdout mirror. |
