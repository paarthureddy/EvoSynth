---
layout: home

hero:
  name: "EvoSynth"
  text: "Dynamic Multi-Armed Bandit Meta-Optimizer"
  tagline: "Automated Model Selection and Hyperparameter Tuning using Competitive Evolutionary Swarms"
  image:
    src: /logo.png
    alt: EvoSynth Logo
  actions:
    - theme: brand
      text: Get Started
      link: /guide/introduction
    - theme: alt
      text: Comparative Analysis
      link: /comparative_analysis
    - theme: alt
      text: GitHub
      link: https://github.com/paarthureddy/EvoSynth

features:
  - icon: 🎯
    title: Dynamic Multi-Armed Bandit (UCB1)
    details: Intelligently allocates computational budget to the most effective search algorithm in real time, balancing exploration and exploitation.
  - icon: 🧬
    title: Multi-Algorithm Metaheuristic Portfolio
    details: Simultanously deploys Genetic Algorithm, Particle Swarm Optimization, Differential Evolution, and CMA-ES in a tiered island architecture.
  - icon: 🌐
    title: Universal Latent Space Continuous Mapping
    details: Projects mixed continuous, discrete, and categorical hyperparameters into an n-dimensional [0, 1] hypercube, enabling continuous velocity physics across all data types.
  - icon: 🔀
    title: Automated Model & Architecture Selection
    details: Dynamically switches model families (XGBoost, LightGBM, Random Forest) and Neural Network architectures (PyTorch CNN) within the same search loop.
  - icon: ⚡
    title: High-Throughput Parallel Evaluations
    details: Asynchronous multi-threading combined with Early Stopping prunes dead-end configurations, reducing multi-hour runs down to minutes.
  - icon: 📊
    title: Glass-Box Observability
    details: Real-time telemetry via FastAPI Server-Sent Events (SSE) and an interactive React dashboard for live inspection of bandit decisions.
---
