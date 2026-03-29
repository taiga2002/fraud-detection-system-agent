# Agent Loop: Autonomous Fraud Detection Self-Improvement

This directory contains results and logs from the **Agent Loop** — a demonstration of autonomous, continuous self-improvement for the fraud detection system.

## Overview

The Agent Loop automatically:

1. **Monitors** fraud detection performance metrics
2. **Detects** degradation due to concept drift (fraud patterns changing)
3. **Triggers** ShinkaEvolve to discover new evolved features
4. **Integrates** best evolved features into production code
5. **Re-evaluates** and loops

This demonstrates the agentic self-improvement capability described in Part 5 of the project documentation.

## Architecture

```
┌─────────────────────────────────────────┐
│     AGENT ORCHESTRATOR                  │
│     (agent_loop.py)                     │
└─────────────────────────────────────────┘
            │
            ▼
    ┌──────────────────┐
    │ Iteration Loop   │
    │ (N cycles)       │
    └──────────────────┘
            │
    ┌───────┼───────┐
    │       │       │
    ▼       ▼       ▼
 MONITOR  EVOLVE  INTEGRATE
  (1)      (2)      (3)
    │       │       │
    └───────┼───────┘
            │
            ▼
        EVALUATE
         (4)
            │
            ▼
        LOOP BACK
```

## Running the Agent Loop

### Basic Usage

```bash
python src/agent/agent_loop.py --iterations 3 --generations-per-evolution 10
```

### Command-Line Arguments

- `--iterations N`: Number of iterations to run (default: 3)
- `--generations-per-evolution N`: Generations per ShinkaEvolve cycle (default: 10)

### Actual Output (from a real run: 2 iterations, 10 generations, 15% drift)

```
================================================================================
    AGENT LOOP FOR FRAUD DETECTION SELF-IMPROVEMENT
================================================================================

Step 1: Establishing Baseline Performance
  ✓ Baseline: P=0.524, R=0.465, F1=0.493, AUC=0.738

================================================================================
AGENT LOOP - ITERATION 1/2
================================================================================

Step 1: Monitor Performance
  ✓ Current: P=0.550, R=0.481, F1=0.513, AUC=0.746

Step 2: Detect Degradation
  Metrics change: Precision: 0.0%, Recall: 0.0%, F1: 0.0%
  ✓ No degradation detected - system performing well

Step 6: Simulate Concept Drift
  💥 Simulating concept drift (severity: 15.0%)...
     ✓ Added noise to 10 features
     ✓ Flipped 779 labels (1.56%)

================================================================================
AGENT LOOP - ITERATION 2/2
================================================================================

Step 1: Monitor Performance
  ✓ Current: P=0.487, R=0.347, F1=0.405, AUC=0.701

Step 2: Detect Degradation
  Metrics change: Precision: 7.0%, Recall: 25.4%, F1: 17.8%
  ⚠️  DEGRADATION DETECTED - Triggering evolution...

Step 3: Trigger ShinkaEvolve Evolution
  🔄 Triggering ShinkaEvolve evolution (iteration 2)...
     Generations: 10
  ✓ ShinkaEvolve completed

Step 4: Integrate Evolved Features
  ✓ Extracted engineer_evolved_features function
  ✓ Updated src/features/evolved_features.py
  🏋️  Retraining models with new features...
     ✓ Models retrained

Step 5: Re-evaluate System
  ✓ Evolved:  P=0.545, R=0.429, F1=0.519, AUC=0.725

  📈 Improvements:
     📈 precision: 0.487 → 0.545 (+11.9%)
     📈 recall: 0.347 → 0.429 (+23.9%)
     📈 f1_score: 0.405 → 0.519 (+28.2%)
     📈 roc_auc: 0.701 → 0.725 (+3.4%)

================================================================================
                          AGENT LOOP COMPLETE
================================================================================
  ✓ Completed 2 iterations
  ✓ Results logged to: experiments/agent_loop/iteration_log.csv
```

## Output Files

- `iteration_log.csv`: CSV log with metrics for each iteration
  - Columns: `iteration`, `action`, `precision`, `recall`, `f1_score`, `combined_score`, `roc_auc`, etc.
  - Actions: `baseline`, `stable`, `evolved`, `failed`

- `results_iteration_N/`: ShinkaEvolve results for iteration N
  - `best/main.py`: Best evolved program
  - `metrics.json`: Evolution metrics
  - Other generation archives

## Key Design Decisions

### 1. Simulation of Concept Drift

In production, fraud patterns shift naturally over months. For demo purposes, we simulate this degradation by:
- Adding Gaussian noise to features
- Flipping labels to introduce annotation errors
- Increasing severity with each iteration (5%, 10%, 15%, etc.)

This allows us to demonstrate the agent loop cycle in minutes rather than months.

### 2. Automatic Code Integration

The agent literally writes its own code improvements by:
- Extracting `engineer_evolved_features()` from the best evolved program
- Automatically updating `src/features/evolved_features.py`
- Retraining models with new features

This demonstrates true **autonomy** and **closed-loop self-improvement**.

### 3. Minimal Agent Design

The agent loop is purposefully minimal to:
- Stay focused on demonstrating the core capability (autonomous improvement)
- Integrate cleanly with the existing ShinkaEvolve infrastructure
- Complete in reasonable time (3 iterations × 10 generations = ~30-45 min)

### 4. Production Deployment

In a real production system, the agent loop would:
- Run on a scheduled job (e.g., daily via cron or Airflow)
- Monitor real fraud patterns using production data
- Trigger evolution when degradation exceeds thresholds
- Include drift detection, model monitoring, and rollback capabilities
- Log all changes for audit and compliance

## Files Modified

- `src/features/evolved_features.py` - Updated with evolved features from each iteration
- `experiments/agent_loop/iteration_log.csv` - Created, logs all iteration metrics
- `experiments/shinkaevolve/results_iteration_N/` - ShinkaEvolve results stored here

## Testing the Agent Loop

### Quick Test (1 iteration, 5 generations)

```bash
python src/agent/agent_loop.py --iterations 1 --generations-per-evolution 5
```

Expected runtime: ~10-15 minutes

### Full Demo (3 iterations, 10 generations)

```bash
python src/agent/agent_loop.py --iterations 3 --generations-per-evolution 10
```

Expected runtime: ~30-45 minutes

## Monitoring Progress

To monitor iteration results:

```bash
cat experiments/agent_loop/iteration_log.csv
```

Or use pandas to analyze:

```python
import pandas as pd
df = pd.read_csv('experiments/agent_loop/iteration_log.csv')
print(df[['iteration', 'action', 'precision', 'recall', 'f1_score']])
```

## Troubleshooting

### ShinkaEvolve Results Not Found

If you get "Best program not found", check:
- ShinkaEvolve run completed successfully
- Results directory exists at `experiments/shinkaevolve/results_iteration_N/best/main.py`
- Check logs for ShinkaEvolve errors

### Feature Extraction Failed

If function extraction fails:
- Check that evolved program has proper `engineer_evolved_features()` function
- Verify EVOLVE-BLOCK markers are present
- Check for syntax errors in evolved code

### Model Retraining Failed

If retraining times out:
- Data files may be missing or corrupted
- Try regenerating data: `python src/data/generate_synthetic_data.py`
- Check disk space and memory

## Next Steps

For production deployment:
1. Replace concept drift simulation with real fraud pattern monitoring
2. Add decision thresholds for when to trigger evolution
3. Implement model rollback if evolved features degrade performance
4. Add compliance logging and audit trail
5. Package as scheduled task/DAG for workflow management
