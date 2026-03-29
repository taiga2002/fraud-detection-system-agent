# Learning Notes

Personal notes from building the ShinkaEvolve-based fraud detection system. Organized for interview discussions and project retrospectives.

---

## Table of Contents

1. [ShinkaEvolve Key Findings](#shinkaevolve-key-findings) — The core contribution
2. [Development Journey](#development-journey) — How the project evolved
3. [Design Decisions & Rationale](#design-decisions--rationale) — Why I made specific choices
4. [What Worked Well](#what-worked-well) — Successes and good patterns
5. [What Didn't Work](#what-didnt-work) — Debugging and lessons learned

7. [Improvements Made](#improvements-made) — Issues found and fixed
8. [Future Optimizations](#future-optimizations) — What I'd do with more time

---

## ShinkaEvolve Key Findings

### The Main Contribution: Task Calibration Methodology

**Finding:** ShinkaEvolve's effectiveness depends critically on baseline task difficulty.

| Experiment | Baseline AUC | ShinkaEvolve Improvement | Still Improving at Gen 20? |
|-----------|--------------|--------------------------|---------------------------|
| Easy task (leaky features) | 1.000 | +1.5% (plateau gen 2) | No |
| Hardened task (realistic) | 0.686 | **+21.2%** (new best gen 19) | **Yes** |

**The 14x difference** came from task calibration, not algorithm changes.

### The 5-Step Calibration Process

1. **Audit correlations:** `df.corr()['target'].abs().sort_values()` → identify features > 0.7
2. **Remove leaky features:** Drop near-perfect proxies
3. **Add noise if needed:** 30% Gaussian + 5% label flips if baseline AUC > 0.95
4. **Reweight fitness:** Emphasize non-saturated components (F1 > AUC if AUC maxed)
5. **Verify baseline:** Ensure 0.60-0.80 range before running evolution

**Transferability:** This methodology applies to any ShinkaEvolve project — image classification, game AI, optimization, etc.

### What the LLM Discovered

**7 evolved features:**
- `evolved_ib_enable_ratio` (5th importance: 24,525) — IB timing / account age
- `evolved_avg_recent_txn` (7th: 9,112) — log-ratio volume to count
- `evolved_log_amount` (13th: 583) — log-transformed amount
- Plus 4 interaction features (device × velocity, weekend × late-night, etc.)

**9 evolved hyperparameters:**
- L1/L2 regularization: 0.1/1.0 → 1.5/1.5 (noise robustness)
- num_leaves: 31 → 45, learning_rate: 0.05 → 0.03
- Full details in `docs/shinkaevolve.md`

**1 evolved fusion logic:**
- Non-linear boost: `transfer_risk^1.2` when `account_risk > 0.3`
- Amplifies high-risk combinations (Japanese fraud pattern)

### Random Search Comparison

**Honest finding:** Random search found best score 0.701 vs ShinkaEvolve 0.640 on 20-iteration budget.

**Interpretation:**
- Random is competitive on small budgets (can get lucky)
- ShinkaEvolve's value = **interpretability** (domain-aware features vs opaque combos)
- Expected advantage at scale (50+ generations, LLM-guided should pull ahead)

---

## Development Journey

### Phase 1: Baseline System (Steps 1-2)

**Step 1:** Hand-written rules based on FSA/NPA fraud patterns
- Result: 38.2% precision, 89.8% recall, 11.8% alert rate
- Lesson: Rules are interpretable but imprecise

**Step 2:** Three-layer ML fusion
- Account risk (17 features, AUC 0.975)
- Transfer risk (30 features with 7 evolved)
- Network risk (graph analytics, 310-node mule cluster)
- Result: 98.3% precision, 99.8% recall, 5.7% alert rate

### Phase 2: Simple Evolution (Step 3)

Applied Optuna (Bayesian optimization with TPE sampler) to tune 4 parameters (fusion weights + threshold):
- Converged at trial 10 out of 100
- Discovered optimal threshold: 0.277 (vs naive 0.300)
- Result: 99.1% precision, 99.8% recall, 5.1% alert rate

**Why Optuna:** Right tool for simple parameter tuning before tackling full ShinkaEvolve

### Phase 3: ShinkaEvolve Exploration

Built simplified reimplementation (500 lines) evolving single-line expressions:
- Taught us: LLM-guided mutations, archive diversity, fitness design
- Result: +0.99% on easy data (disappointing)
- **Diagnosis:** Task too easy (AUC=1.0, solved in 1 boosting round)

### Phase 4: Real ShinkaEvolve Integration

Integrated actual `shinka-evolve` package with 3 EVOLVE-BLOCKs:
- Fixed task difficulty (removed leaky features, added noise)
- Result: +21.2% on hardened task, still climbing gen 20
- **Key insight:** Task calibration is critical

### Phase 5: Production Integration

Extracted best evolved components:
- `src/features/evolved_features.py` — 7 LLM-discovered features
- `src/models/transfer_risk_model.py` — evolved hyperparameters
- `src/scoring/fusion_layer.py` — non-linear fusion boost
- Validation: Feature importance shows 5th, 7th, 13th positions

---

## Design Decisions & Rationale

### Why 3 EVOLVE-BLOCKs Instead of 1?

**Decision:** Evolve features + hyperparameters + fusion simultaneously

**Rationale:**
- Tests interdependencies (better features need different hyperparams)
- More realistic (production would tune all three together)
- Demonstrates ShinkaEvolve's power (full pipeline synthesis, not just parameter tuning)

**Tradeoff:** Higher failure rate (38% vs ~20% for single block) but more interesting mutations

### Why Multi-Objective Fitness?

**Problem with single metrics:**
- Just AUC → could output "always fraud" (100% recall, terrible precision)
- Just F1 → could achieve high F1 with 50% alert rate (overwhelms analysts)

**Solution:** Composite score with 4 components
```python
combined_score = 0.40 * roc_auc          # Discrimination
             + 0.30 * f1_score         # Precision-recall balance
             + 0.20 * prevention_rate  # Business value
             + 0.10 * alert_quality    # Penalize extreme alert rates
```

**Critical component:** `alert_quality = 0` if alert_rate < 1% or > 50% (prevents degenerate solutions)

### Why qwen2.5:14b Instead of GPT-4?

| Factor | qwen2.5:14b (used) | GPT-4 API |
|--------|-------------------|-----------|
| Cost | $0 (local) | $15-25 per 20-gen run |
| Success rate | 62% | ~90% |
| Time per mutation | 45-60s | 5-10s |
| Domain understanding | Good | Better |
| Reproducibility | Fully offline | Requires API keys |

**Judgment:** For proof-of-concept, free + reproducible > perfect + costly

### Why 20 Generations Not 50?

- Runtime: 20 gens = 55 min, 50 gens = 2.3 hours
- Evidence: Best found at gen 19 → clearly still improving
- Judgment: Show proof-of-concept, acknowledge limitation

### Why 3-Seed Averaging?

- Run each program 3 times with different random seeds
- Reduces variance from random train/test split
- Detects unstable programs (high std dev)
- Tradeoff: 3x slower but more robust

### Why Data Subsample to 10K?

- Use 10K of 50K transactions
- 5x faster training (3s vs 15s per model)
- Judgment: 10K enough for stable feature importance

---

## What Worked Well

### Three-Layer Architecture

Separating account risk, transfer risk, and network risk made the system debuggable:
- Can see which layer triggered a false positive
- Fusion weights are tunable knobs (increase network weight if bank cares more about mule detection)
- Each layer can be improved independently

### Reason Codes from Config

All reason codes (AR01-AR05, TR01-TR07, NR01-NR06) in `configs/japan_calibration.yaml`:
- Single source of truth
- Earlier: hardcoded in 3 files with inconsistent wording (maintenance nightmare)
- Now: config-driven and consistent

### Japan-Specific Calibration

Grounding synthetic data in FSA/NPA statistics:
- 11,009 cases FY2024, ¥2.26M average loss, 70% existing IB accounts
- Amount distributions, account age patterns from official sources
- More credible than generic fraud data

### Human-in-the-Loop Design

"No automated blocking without human approval" from the start:
- System suggests actions, analyst decides
- Comprehensive reason codes for investigation
- Natural victim support protocol (partial freeze, 30-min call, police hotline)

### Iterative Validation

Test each component before integrating:
- initial.py runs standalone → verified
- evaluate.py works with run_shinka_eval → verified
- Config parses correctly → verified
- Then run full experiment

---

## What Didn't Work

### Ground Truth Leakage

**The problem:** Transfer risk model used `receiver_is_mule` as a feature — literally the label.
- Achieved ROC AUC 1.000 (should have been immediate red flag)
- Inflated precision to 95% (realistic is 75-85%)

**How it happened:** Joining receiver account info pulled `is_mule` along with legitimate features.

**The fix:** Built `exclude_ground_truth=True` in feature engineering, removed leaky features.

**Lesson:** Always check feature importance. If one feature dominates, investigate.

### Naive ShinkaEvolve Run Failed

**The problem:** First run showed +1.5%, plateau at gen 2.

**Root cause:** Task was too easy (AUC=1.0, 5 features with corr > 0.7).

**The fix:** Hardened task (remove leaky features, add noise).

**Lesson:** Evolutionary search only shows value when baseline is genuinely challenging.

### Evolution Convergence Was Fragile (Resolved)

The original GA used early stopping with a fragile criterion: last 5 generations with 0.001 threshold.
- On noisier fitness landscapes, this could stop too early
- Resolved by switching to Optuna, which uses proper Bayesian optimization (TPE sampler) and does not rely on fragile convergence criteria
- Optuna's pruning and trial budget approach is more principled

### Random Train/Test Split

Used random split, but production needs temporal split:
- Random leaks temporal information
- Can't assess concept drift
- `transfer_risk_model_v2.py` (now archived) supported temporal split but was never wired into the main pipeline

### Simple GA / Optuna vs Full ShinkaEvolve

Initially tried ShinkaEvolve for weight tuning:
- Search space: just 4 continuous numbers
- Too simple for program synthesis framework
- Replaced the simple GA with Optuna for weight tuning — the right tool for the job
- Lesson: ShinkaEvolve shines on feature discovery, not parameter tuning. Optuna is the right choice for simple hyperparameter optimization


## Development Journey

### Step 1: Baseline Rules

Hand-written rules based on FSA/NPA patterns:
- New beneficiary + high amount
- Recently enabled IB
- Balance drain ratio
- Result: 38.2% precision, 89.8% recall, 11.8% alert rate

**Lesson:** Rules are interpretable but imprecise. Every rule added improved recall but increased false positives.

### Step 2: Three-Layer Fusion

Built separate risk models:
- **Account risk:** LightGBM, 17 features, AUC 0.975
- **Transfer risk:** LightGBM, 30 features (7 evolved), AUC 0.9999
- **Network risk:** NetworkX graph, 310-node mule cluster

Fusion layer weighted them (0.25, 0.45, 0.30) with reason codes.

**Lesson:** Separating concerns made system debuggable. Biggest architectural decision.

### Step 3: Optuna Bayesian Optimization

Optuna (TPE sampler) tuned fusion weights + threshold:
- Converged at trial 10 out of 100
- Result: 99.1% precision, 99.8% recall
- Cut alerts by 57% (5,924 → 2,537)

**Lesson:** Simple problems need simple tools. Optuna is the right choice for parameter tuning.

### Step 4: ShinkaEvolve Naive Run

Used real `shinka-evolve` package:
- Result: +1.5%, plateau gen 2
- **Diagnosis:** Task too easy (AUC=1.0, leaky features)
- **Action:** Harden task

### Step 5: ShinkaEvolve Hardened Run

Applied task calibration:
- Removed 5 leaky features (corr > 0.7)
- Added 30% noise + 5% label flips
- Result: +21.2%, still climbing gen 20

**Lesson:** Task difficulty determines evolution effectiveness.

### Step 6: Production Integration

Extracted evolved components:
- Features → `src/features/evolved_features.py`
- Hyperparams → `src/models/transfer_risk_model.py`
- Fusion → `src/scoring/fusion_layer.py`
- Validation: Feature importance ranks 5th, 7th, 13th

---

## Design Decisions & Rationale

### Multi-Objective Fitness Function

**Why composite score?**

Single metric problems:
- Just AUC → could output "always fraud" (100% recall, 0% precision)
- Just F1 → could get high F1 with 50% alert rate (analysts overwhelmed)

Composite prevents gaming:
```python
combined = 0.40 * auc + 0.30 * f1 + 0.20 * prevention + 0.10 * alert_quality
```

Alert quality penalizes extremes:
```python
if alert_rate < 0.01 or alert_rate > 0.5:
    alert_quality = 0.0  # Degenerate solutions scored zero
```

### 3-Seed Averaging

**Why:** Reduces variance from random train/test split, detects unstable programs

**Tradeoff:** 3x slower (30s vs 10s per program)

**Judgment:** Worth it for robustness

### Data Subsampling

**Why:** Use 10K of 50K transactions

**Benefit:** 5x faster training (3s vs 15s)

**Tradeoff:** Noisier fitness estimates

**Judgment:** 10K sufficient for stable rankings

### Hardening Steps

**Why 30% noise (not 10% or 50%)?**
- 10% too gentle (AUC still ~0.95)
- 50% too harsh (baseline unlearnable)
- 30% hit sweet spot (AUC ~0.70)

**Why 5% label flips (not 10%)?**
- 5% fraud class → 5% flips ≈ 10% of fraud mislabeled
- More would make task too noisy

**Why remove features at 0.7 threshold (not 0.5 or 0.9)?**
- >0.7 = near-perfect proxy (AUC alone would be >0.90)
- <0.5 = useful signal, not leakage
- 0.7 is the standard cutoff in ML literature

---

## What Worked Well

### Iterative Validation

Test before integrating:
- initial.py runs standalone → pass
- evaluate.py works with run_shinka_eval → pass
- Config parses → pass
- Then run full experiment

### Diagnostic Thinking

When naive run showed +1.5%:
- Didn't conclude "ShinkaEvolve doesn't work"
- Diagnosed: task too easy
- Fixed: hardened task
- Result: +21.2%

### Honest Baselines

Ran random search even though it might outperform:
- Result: Random found 0.701 vs ShinkaEvolve 0.640
- Reported truthfully, interpreted fairly

### Production Integration

Didn't stop at experiments:
- Extracted evolved code
- Integrated into main pipeline
- Validated via feature importance

---

## What Didn't Work

### Ground Truth Leakage

`receiver_is_mule` used as feature → AUC 1.000 (red flag)

**Fix:** `exclude_ground_truth=True`, removed leaky proxies

**Lesson:** Check feature importance. If one dominates, investigate.

### Naive ShinkaEvolve Run

+1.5% improvement, plateau gen 2

**Root cause:** 5 features corr > 0.7, AUC=1.0

**Fix:** Task calibration methodology

**Lesson:** Baseline difficulty determines evolution value

### Path Handling Inconsistencies

Scripts used `../../` paths, broke when run from different directories

**Fix:** `PROJECT_ROOT` in all 10 files

**Lesson:** Always use absolute paths resolved from `__file__`

### Dead Code Accumulated

`shinka_inspired_evolution.py` (645 lines) never called

**Fix:** Removed during cleanup

**Lesson:** Regular code audits prevent cruft

---

## Improvements Made

### Code Fixes

| Issue | Fix |
|-------|-----|
| Ground truth leakage | Removed from features, built exclude_ground_truth=True |
| Leaky proxy features (5 features corr > 0.7) | Removed from hardened experiment + main pipeline |
| Path handling (../../ broke from root) | PROJECT_ROOT in all 10 scripts |
| Dead code (645 lines unused) | Removed shinka_inspired_evolution.py |
| Missing dependency | Added shinka-evolve to requirements.txt |

### Documentation Fixes

| Issue | Fix |
|-------|-----|
| Narrative structure (led with metrics) | Restructured to lead with ShinkaEvolve finding |
| Buried key insight | Task calibration now in executive summary |
| No random baseline | Added and reported honestly (0.701 vs 0.640) |
| Missing discoveries doc | Documented in `docs/shinkaevolve.md` |

### Experimental Fixes

| Issue | Fix |
|-------|-----|
| Easy task (+1.5%) | Diagnosed, hardened, re-ran (+21.2%) |
| No baseline comparison | Random search (20 iterations) |
| Evolved features had 0 importance | Removed leaky features from main pipeline → 5th, 7th, 13th |

---

## Future Optimizations

### High Priority

1. **Real banking data validation** — Partner with Japanese megabank, measure actual improvement
2. **Temporal train/test split** — Train on months 1-8, test on 9-10
3. **Longer ShinkaEvolve runs** — 50-100 generations (still climbing at gen 20)

### Medium Priority

4. **Stronger LLM** — qwen2.5:72b or GPT-4 to reduce 38% failure rate
5. **SHAP feature selection** — Reduce from 30 to ~20 features
6. **Pareto front visualization** — Multiple precision/recall operating points

### Low Priority

7. **Adaptive mutation rate** — Start high, decay as population converges
8. **Config-driven thresholds** — Move 0.8/0.6/0.4 from code to config
9. **Unit tests** — Core functions (especially evolved_features.py)
10. **Grid search baseline** — More thorough than random search

---

## End-to-End Pipeline Performance

After all improvements:

| Step | Precision | Recall | F1 | Alert Rate |
|------|-----------|--------|----|------------|
| Step 1: Baseline rules | 38.2% | 89.8% | 0.536 | 11.8% |
| Step 2: ML Fusion + evolved features | 98.3% | 99.8% | 0.991 | 5.1% |
| Step 3: Optimized weights (Optuna) | 99.1% | 99.8% | 0.994 | 5.1% |

**ShinkaEvolve (hardened):** +21.2% improvement over baseline

**Random search:** +17.1% but opaque features

---

**Note:** These are personal notes for my own reflection and interview prep, not formal documentation.
