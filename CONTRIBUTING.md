# Contributing to EvoSynth

Thank you for your interest in contributing to **EvoSynth**! We welcome contributions from researchers, machine learning engineers, and open-source developers.

---

## 🛠️ Development Setup

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/paarthureddy/EvoSynth.git
   cd EvoSynth
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run Tabular Search Verification:**
   ```bash
   python main.py
   ```

4. **Run Vision NAS Verification:**
   ```bash
   python run_vision_nas.py
   ```

---

## 🌿 Branching Strategy & Git Workflow

- Always create a descriptive feature branch from `main`:
  ```bash
  git checkout -b feat/add-cmaes-enhancement
  ```
- Write clear, concise commit messages following Conventional Commits (`feat:`, `fix:`, `docs:`, `perf:`).
- Open a Pull Request targeting `main`.

---

## 🧩 Adding a New Optimizer

EvoSynth's modular architecture makes it easy to add new optimization algorithms:
1. Subclass the base optimizer interface under `automl_engine/engine/`.
2. Implement `step()` and latent space vector transformation in `automl_engine/core/search_space.py`.
3. Register the new heuristic in `automl_engine/engine/controller.py` so the UCB1 Multi-Armed Bandit can assign exploration arms to it.

---

## 📜 Code of Conduct

Please maintain a collaborative, respectful, and inclusive environment. All participants are expected to adhere to standard academic and open-source community standards.
