# Source Code Directory

**Total:** ~4,476 lines of Python across 15 files

---

## Directory Structure

```
src/
├── data/              # Synthetic data generation (FSA/NPA calibrated)
├── features/          # Feature engineering (base + evolved)
├── models/            # ML models (account, transfer, baseline rules)
├── graph/             # Network analytics (mule detection)
├── scoring/           # Fusion layer (with evolved non-linear boost)
├── evaluation/        # Metrics, comparisons, ShinkaEvolve plots
├── simulation/        # Optuna fusion weight optimizer
└── demo/              # Interactive Streamlit dashboards (2 apps)
```

---

## Pipeline Flow

```
Step 1: Baseline
  data/ → models/baseline_rules.py → evaluation/baseline_evaluation.py

Step 2: ML Fusion
  data/ → features/ → models/ → graph/ → scoring/ → evaluation/fusion_evaluation.py

Step 3: Optuna Optimization
  simulation/simple_evolution.py → evaluation/evolved_evaluation.py

Step 4: ShinkaEvolve Visualization
  evaluation/plot_shinkaevolve_results.py
```

**Run pipeline:** `./run_all.sh`
**Run demos:** `./launch_demos.sh both`
**ShinkaEvolve experiments:** `cd experiments/shinkaevolve && python run_experiment.py --num-generations 20`
