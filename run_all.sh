#!/bin/bash

# Complete Fraud Detection Pipeline (Steps 1-3)
# ShinkaEvolve-Integrated Japanese Banking Fraud Detection

set -e  # Exit on any error

echo "================================================================================"
echo "FRAUD DETECTION SYSTEM - COMPLETE PIPELINE"
echo "================================================================================"

# Environment setup
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH
PYTHON=./venv/bin/python

# Ensure output directories exist
mkdir -p data/processed
mkdir -p experiments/fusion
mkdir -p experiments/shinka
mkdir -p outputs/tables
mkdir -p outputs/figures

# Step 1: Baseline
echo ""
echo "=== STEP 1: BASELINE SYSTEM ==="
echo ""

echo "1.1: Generating synthetic data..."
$PYTHON src/data/generate_synthetic_data.py

echo ""
echo "1.2: Running baseline evaluation..."
$PYTHON src/evaluation/baseline_evaluation.py

# Step 2: Fusion
echo ""
echo "=== STEP 2: FUSION SYSTEM ==="
echo ""

echo "2.1: Training account risk model..."
$PYTHON src/models/account_risk_model.py

echo ""
echo "2.2: Engineering transaction features..."
$PYTHON src/features/transaction_features.py

echo ""
echo "2.3: Training transfer risk model..."
$PYTHON src/models/transfer_risk_model.py

echo ""
echo "2.4: Building mule network detector..."
$PYTHON src/graph/mule_network_detector.py

echo ""
echo "2.5: Running fusion evaluation..."
$PYTHON src/evaluation/fusion_evaluation.py

# Step 3: Evolution
echo ""
echo "=== STEP 3: FUSION WEIGHT OPTIMIZATION ==="
echo ""

echo "3.1: Running fusion weight optimizer (Optuna)..."
$PYTHON src/simulation/simple_evolution.py

echo ""
echo "3.2: Evaluating optimized policy..."
$PYTHON src/evaluation/evolved_evaluation.py

# Step 4: ShinkaEvolve Visualization (if results exist)
echo ""
echo "=== STEP 4: SHINKAEVOLVE RESULTS ==="
echo ""

if [ -d "experiments/shinkaevolve/results" ] && [ -d "experiments/shinkaevolve/results_hardened" ]; then
    echo "4.1: Generating ShinkaEvolve comparison plots..."
    $PYTHON src/evaluation/plot_shinkaevolve_results.py
else
    echo "Skipping: ShinkaEvolve results not found. Run experiments/shinkaevolve/run_experiment.py first."
fi

# Step 5: Agent Loop (Optional — demonstrates autonomous self-improvement)
echo ""
echo "=== STEP 5: AGENT LOOP (OPTIONAL) ==="
echo ""

if [ "$RUN_AGENT_LOOP" = "true" ]; then
    echo "5.1: Running agent loop for autonomous self-improvement (3 iterations)..."
    $PYTHON src/agent/agent_loop.py --iterations 3 --generations-per-evolution 10
    echo ""
    echo "5.2: Agent loop results:"
    echo "  - Iteration log: experiments/agent_loop/iteration_log.csv"
    echo "  - Evolved features: src/features/evolved_features.py"
else
    echo "Skipping: Set RUN_AGENT_LOOP=true to enable (takes ~30-45 minutes)"
fi

# Summary
echo ""
echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  - Step 1 Baseline: outputs/tables/baseline_metrics.csv"
echo "  - Step 2 Fusion: outputs/tables/fusion_metrics.csv"
echo "  - Step 3 Evolved: outputs/tables/evolved_comparison.csv"
echo ""
echo "Models:"
echo "  - experiments/fusion/account_risk_model.pkl"
echo "  - experiments/fusion/transfer_risk_model.pkl"
echo "  - experiments/fusion/mule_network_detector.pkl"
echo "  - experiments/shinka/evolved_config.pkl"
echo ""
echo "See FINAL_REPORT.md for comprehensive analysis."
echo "================================================================================"
