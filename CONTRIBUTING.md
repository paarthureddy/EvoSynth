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
   pip install -e ".[all]"    # Python 3.11+; installs the `evosynth` command
   ```

3. **Check the Settings and Run a Quick Search:**
   ```bash
   evosynth settings check
   evosynth run --budget 36 --pop-size 20 --t1-iters 2 --k-iters 1 --no-refit
   ```

4. **Run Vision NAS Verification:**
   ```bash
   evosynth nas
   ```

5. **Run the tests** (about 5 seconds, no model training):
   ```bash
   python -m pytest
   ```

6. **Settings rule:** never hard-code a tunable value in Python. Add it to the matching file in
   `settings/` with a comment saying what it controls, where it is used
   (`Used in: <file> → <function>`) and its range, and read it with `evosynth.config.get_settings()`.

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
1. Implement the ask / tell interface (`ask(n)` returns latent vectors in $[0, 1]^d$, `tell(individuals)` updates the optimizer) in `src/evosynth/optimizers/`.
2. Put its constants in `settings/optimizers.toml` (with "Used in" comments).
3. Register it in `src/evosynth/engine/controller.py` (`_make_optimizer`) so the UCB1 Multi-Armed Bandit can assign exploration arms to it.

See `docs/developer/adding-optimizer.md` for a full example.

---

## 📜 Code of Conduct

Please maintain a collaborative, respectful, and inclusive environment. All participants are expected to adhere to standard academic and open-source community standards.
