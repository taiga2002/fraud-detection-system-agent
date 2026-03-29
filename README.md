# ShinkaEvolve Fraud Detection Agent
## Introduction
This project applies ShinkaEvolve, a framework for LLM-guided program evolution, to the domain of Japanese banking fraud detection. ShinkaEvolve uses large language models to iteratively mutate and improve code through evolutionary search that autonomously discovers new detection logic and evolves its own features without human redesign, adapting as fraud patterns shift.

**What I built**: A three-layer fraud detection system with an autonomous agent loop that analyzes account profiles, transaction behavior, and network patterns to identify money laundering with 99.1% precision and 99.8% recall on synthetic data calibrated to official Japanese government statistics — and continuously self-improves when performance degrades.

---

## Quick Start

```bash
# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run complete pipeline (Steps 1-3 + ShinkaEvolve visualization)
./run_all.sh

# Run ShinkaEvolve experiment (requires Ollama with qwen2.5:14b)
cd experiments/shinkaevolve
python run_experiment.py --num-generations 20 --results-dir results
```

---

## What This Project Does

We applied **ShinkaEvolve** [1] (LLM-guided program evolution framework) to a concrete Japan-specific problem: detecting online banking fraud (特殊詐欺), which costs Japanese consumers ~¥30B+ annually.

### The Core Contribution

ShinkaEvolve evolves three aspects of the fraud detection pipeline simultaneously:

| EVOLVE-BLOCK | What It Evolves | Example Discovery |
|-------------|----------------|-------------------|
| **Feature Engineering** | Interaction features from 23 base columns | `ib_enable_ratio` = IB days / account age (5th most important feature) |
| **Model Hyperparameters** | LightGBM params (num_leaves, regularization, etc.) | Higher L1/L2 regularization handles noise better |
| **Fusion Logic** | How model scores are combined for decisions | Non-linear power boost when account risk > 0.3 |

### Key Finding: Task Difficulty Calibration

The most important finding is that **evolutionary search only works when the task is genuinely hard**:

| Experiment | Baseline | Best Evolved | Improvement |
|-----------|----------|-------------|-------------|
| Easy task (leaky features) | 0.920 | 0.934 | +1.5% (plateau gen 2) |
| **Hardened task** (realistic) | **0.528** | **0.640** | **+21.2%** (still climbing gen 19) |

We hardened the task by removing 5 near-perfect proxy features, adding 30% Gaussian noise, and flipping 5% of labels. This created a task where the LLM-evolved features genuinely matter.

See `outputs/figures/shinkaevolve_evolution_comparison.png` for the visual comparison.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│              FRAUD DETECTION PIPELINE                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Layer 1: Account Risk     (LightGBM, ROC AUC 0.975)   │
│  Layer 2: Transfer Risk    (LightGBM, 30 features)      │
│  Layer 3: Network Risk     (NetworkX graph analytics)    │
│                                                          │
│  ┌──────────────────────────────────────────────┐        │
│  │  ShinkaEvolve-Discovered Components          │        │
│  │                                              │        │
│  │  7 evolved features (IB timing, device       │        │
│  │  interactions, temporal anomalies)            │        │
│  │  Evolved hyperparams (lr=0.03, L1/L2=1.5)   │        │
│  │  Non-linear fusion boost                     │        │
│  └──────────────────────────────────────────────┘        │
│                                                          │
│  Fusion → Decision → Reason Codes → Action              │
└──────────────────────────────────────────────────────────┘
```

---

## Pipeline Results

| Step | What | Precision | Recall | F1 | Alert Rate |
|------|------|-----------|--------|-----|------------|
| Step 1 | Baseline rules | 38.2% | 89.8% | 0.536 | 11.8% |
| Step 2 | 3-layer ML fusion + evolved features | 98.3% | 99.8% | 0.991 | 5.1% |
| Step 3 | Optimized fusion weights (Optuna) | 99.1% | 99.8% | 0.994 | 5.1% |
| Step 4 | Agent loop (recovery after 15% drift) | +11.9% | +23.9% | +28.2% | 8.0% |

Note: Steps 1-3 metrics are on synthetic data. Step 4 shows relative improvement after concept drift.

---

## Project Structure

```
sakana-ai-2/
├── experiments/
│   ├── shinkaevolve/               # ShinkaEvolve experiment (LLM-guided evolution)
│   │   ├── initial.py              # Evolvable pipeline (3 EVOLVE-BLOCKs)
│   │   ├── evaluate.py             # Fitness function (run_shinka_eval)
│   │   ├── config.yaml             # ShinkaEvolve config for Ollama
│   │   ├── system_prompt.txt       # Domain-specific LLM guidance
│   │   ├── run_experiment.py       # Launcher
│   │   ├── results/                # Easy task run (20 gen)
│   │   ├── results_hardened/       # Hardened task run (20 gen)
│   │   └── results_iteration_N/    # Agent loop evolution results
│   ├── agent_loop/                 # Agent loop outputs & logs
│   │   ├── iteration_log.csv       # Metrics history
│   │   └── README.md               # Agent loop documentation
│   ├── shinka/                     # Step 3 Optuna outputs (evolved_config.pkl)
│   └── fusion/                     # Step 2 trained models (.pkl)
├── src/
│   ├── agent/                      # Agent loop (autonomous self-improvement)
│   ├── data/                       # Synthetic data generation
│   ├── features/                   # Feature engineering (base + evolved)
│   ├── models/                     # LightGBM models + baseline rules
│   ├── scoring/                    # Fusion decision layer
│   ├── graph/                      # NetworkX mule network detection
│   ├── evaluation/                 # Metrics + visualization
│   └── simulation/                 # Optuna optimizer
├── docs/
│   ├── shinkaevolve.md             # ShinkaEvolve integration + discoveries
│   ├── learning-notes.md           # Development learnings
│   ├── final-report.md             # Technical report
│   └── domain-research-ja.md       # Japanese domain research (FSA/NPA refs)
├── archive/deprecated/             # Superseded code (preserved for reference)
├── tools/presentation/             # PowerPoint generation tooling
├── run_all.sh                      # End-to-end pipeline
├── .gitignore
└── requirements.txt
```

---

## ShinkaEvolve Setup

### Prerequisites

- Python 3.10+ with `pip install -r requirements.txt`
- [Ollama](https://ollama.ai) with `ollama pull qwen2.5:14b`

### Running the Experiment

```bash
cd experiments/shinkaevolve

# Quick test (5 generations, ~7 min)
python run_experiment.py --num-generations 5 --results-dir results_test

# Full run (20 generations, ~55 min)
python run_experiment.py --num-generations 20 --results-dir results

# Or via shinka_run CLI
shinka_run --task-dir . --results_dir results --num_generations 20 \
  --config-fname config.yaml --verbose
```

### What the LLM Discovers

The best evolved program (gen 19, hardened run) introduced:
- `evolved_ib_enable_ratio` — IB enablement timing relative to account age
- `evolved_avg_recent_txn` — log ratio of volume to transaction count
- `evolved_weekend_late_night_volume` — temporal anomaly interaction
- `evolved_susp_device_ib` — suspicious device x IB timing
- Higher regularization (L1=L2=1.5) to handle noisy data
- Non-linear power transform in fusion (transfer_risk^1.2 when account_risk > 0.3)

---

## Agent Loop: Autonomous Self-Improvement

In addition to the initial ShinkaEvolve run, we've implemented an **Agent Loop** that demonstrates **continuous, autonomous self-improvement**. The agent automatically:

1. **Monitors** fraud detection performance metrics
2. **Detects** performance degradation (simulating concept drift)
3. **Triggers** ShinkaEvolve to discover new evolved features
4. **Integrates** evolved features directly into production code
5. **Re-evaluates** and loops

### Running the Agent Loop

```bash
# Run 3 iterations with 10 generations per evolution
python src/agent/agent_loop.py --iterations 3 --generations-per-evolution 10

# Quick test (1 iteration, 5 generations)
python src/agent/agent_loop.py --iterations 1 --generations-per-evolution 5
```

Or include it in the full pipeline:

```bash
RUN_AGENT_LOOP=true ./run_all.sh
```

### How It Works

The agent loop demonstrates agentic self-improvement by:

- **Autonomy**: No manual intervention — the agent detects degradation, evolves solutions, and integrates them automatically
- **Closed-loop**: Updated features are written directly to `src/features/evolved_features.py` by the agent
- **Metrics-driven**: Evolution is triggered only when performance metrics degrade below threshold
- **Concept drift simulation**: For demo purposes, we simulate fraud patterns getting harder (this represents real-world concept drift over months)

### Actual Results (2 iterations, 10 generations, 15% drift)

```
Iteration 1: No degradation — system stable
   → 15% concept drift applied (779 labels flipped, noise to 10 features)

Iteration 2: Degradation DETECTED (P -7.0%, R -25.4%, F1 -17.8%)
   → ShinkaEvolve evolved 10 generations (best score 0.597)
   → Extracted & integrated 7 new features into evolved_features.py
   → Retrained models with new features
   → Recovery: Precision +11.9%, Recall +23.9%, F1 +28.2%, AUC +3.4%
```

Results are logged to `experiments/agent_loop/iteration_log.csv`.

### Files

- `src/agent/` — Agent loop implementation
  - `agent_loop.py` — Main orchestrator
  - `performance_monitor.py` — Metrics tracking & degradation detection
  - `evolution_trigger.py` — ShinkaEvolve launcher
  - `code_integrator.py` — Feature extraction & code update
  - `concept_drift_simulator.py` — Fraud pattern hardening

- `experiments/agent_loop/` — Agent loop outputs
  - `iteration_log.csv` — Metrics history
  - `README.md` — Detailed documentation

See [experiments/agent_loop/README.md](experiments/agent_loop/README.md) for full documentation.

---

## Limitations

- **Synthetic data** — models trained on generated data with realistic ambiguity (overlapping distributions, stealthy fraud patterns, ambiguous normal transactions). The hardened synthetic data narrows the gap to real-world performance, but some degradation is still expected in production.
- **No temporal validation** — random train/test split, not time-based. Can't assess concept drift.
- **14B local LLM** — qwen2.5:14b produces broken code ~35% of the time. A stronger model would improve mutation success rate.
- **20 generations** — the hardened run was still improving at gen 19. More generations would find better solutions.

---

## Documentation

- [docs/shinkaevolve.md](docs/shinkaevolve.md) — ShinkaEvolve integration, discoveries, and results
- [docs/learning-notes.md](docs/learning-notes.md) — What we learned about making evolutionary search effective
- [docs/final-report.md](docs/final-report.md) — Full technical report


---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Evolution Framework | ShinkaEvolve (`shinka-evolve` v0.0.1) |
| LLM | Ollama qwen2.5:14b (local, free) |
| ML Framework | LightGBM 4.6 |
| Graph Analytics | NetworkX 3.6 |
| Data Processing | Pandas 3.0, NumPy 2.4 |
| Environment | Local (Apple M4 Pro, no cloud) |

---

**References:**

[1] ShinkaEvolve: Automated Scientific Discovery through LLM-Guided Program Evolution. https://sakana.ai/shinka-evolve/

[2] LightGBM: Ke et al., NeurIPS 2017. https://github.com/microsoft/LightGBM

[3] FSA Japan — Online banking fraud statistics (FY2024)

[4] NPA Japan — Special fraud (特殊詐欺) reports
