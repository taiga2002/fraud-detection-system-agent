"""
Fusion Weight Optimizer using Optuna

Optimizes the fusion layer weights (account, transfer, network) and decision
threshold using Bayesian optimization (TPE sampler). This replaces the previous
genetic algorithm approach — Optuna is more sample-efficient for this small
4-parameter search space.

The optimized configuration is saved to experiments/shinka/evolved_config.pkl
and loaded by the fusion layer at runtime.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score
import pickle
import sys
import os
import optuna

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scoring.fusion_layer import FusionLayer


def evaluate_config(account_weight, transfer_weight, network_weight, threshold,
                    transactions_df, accounts_df, graph_features_dict):
    """
    Evaluate a fusion weight configuration.

    Returns:
        Dictionary with fitness components
    """
    config = {
        'account_weight': account_weight,
        'transfer_weight': transfer_weight,
        'network_weight': network_weight,
        'thresholds': {
            'critical': 0.8,
            'high': 0.6,
            'medium': 0.4,
            'low': 0.3
        }
    }

    fusion = FusionLayer(config)

    # Evaluate on sample for speed
    sample_size = min(2000, len(transactions_df))
    sample_txns = transactions_df.sample(n=sample_size, random_state=42)

    predictions = []
    ground_truth = []
    prevented_loss = 0

    for _, txn in sample_txns.iterrows():
        sender_account = accounts_df[accounts_df['account_id'] == txn['sender']].iloc[0]

        account_score = sender_account.get('account_risk_score', 0)
        transfer_score = txn.get('transfer_risk_score', 0)
        network_score = sender_account.get('network_risk_score', 0)

        fused_score = fusion.fuse_scores(account_score, transfer_score, network_score)

        pred = 1 if fused_score >= threshold else 0
        truth = int(txn['is_laundering'])

        predictions.append(pred)
        ground_truth.append(truth)

        if pred == 1 and truth == 1:
            prevented_loss += txn['amount_jpy']

    y_true = np.array(ground_truth)
    y_pred = np.array(predictions)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    alert_rate = y_pred.sum() / len(y_pred)

    # Multi-objective fitness
    fitness = (
        0.35 * precision +
        0.35 * recall +
        0.20 * (prevented_loss / 10_000_000_000) +
        0.10 * (1 - alert_rate)
    )

    return {
        'fitness': fitness,
        'precision': precision,
        'recall': recall,
        'prevented_loss': prevented_loss,
        'alert_rate': alert_rate
    }


def main():
    """Optimize fusion weights using Optuna"""
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    print("Loading data...")
    transactions_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'transactions_with_risk.csv'))
    accounts_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'accounts_with_network_risk.csv'))

    with open(os.path.join(PROJECT_ROOT, 'experiments', 'fusion', 'mule_network_detector.pkl'), 'rb') as f:
        detector_data = pickle.load(f)
        graph_features_dict = detector_data['account_graph_features']

    print(f"Loaded {len(transactions_df)} transactions")

    print("=" * 80)
    print("BAYESIAN OPTIMIZATION - FUSION WEIGHTS (Optuna TPE)")
    print("=" * 80)

    def objective(trial):
        # Sample weights from Dirichlet-like space (3 values summing to 1)
        raw_a = trial.suggest_float('raw_account', 0.01, 1.0)
        raw_t = trial.suggest_float('raw_transfer', 0.01, 1.0)
        raw_n = trial.suggest_float('raw_network', 0.01, 1.0)
        total = raw_a + raw_t + raw_n
        account_weight = raw_a / total
        transfer_weight = raw_t / total
        network_weight = raw_n / total

        threshold = trial.suggest_float('threshold', 0.1, 0.9)

        result = evaluate_config(
            account_weight, transfer_weight, network_weight, threshold,
            transactions_df, accounts_df, graph_features_dict
        )

        # Store weights for retrieval
        trial.set_user_attr('account_weight', account_weight)
        trial.set_user_attr('transfer_weight', transfer_weight)
        trial.set_user_attr('network_weight', network_weight)
        trial.set_user_attr('precision', result['precision'])
        trial.set_user_attr('recall', result['recall'])
        trial.set_user_attr('alert_rate', result['alert_rate'])

        return result['fitness']

    # Run optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Enqueue baseline as first trial
    study.enqueue_trial({
        'raw_account': 0.25,
        'raw_transfer': 0.45,
        'raw_network': 0.30,
        'threshold': 0.3
    })

    n_trials = 100

    def log_callback(study, trial):
        if (trial.number + 1) % 10 == 0 or trial.number == 0:
            best = study.best_trial
            print(f"\n  Trial {trial.number + 1}/{n_trials}")
            print(f"    Best fitness: {best.value:.4f}")
            print(f"    Precision: {best.user_attrs['precision']:.3f}, "
                  f"Recall: {best.user_attrs['recall']:.3f}, "
                  f"Alert Rate: {best.user_attrs['alert_rate']:.3f}")
            print(f"    Weights: A={best.user_attrs['account_weight']:.2f}, "
                  f"T={best.user_attrs['transfer_weight']:.2f}, "
                  f"N={best.user_attrs['network_weight']:.2f}, "
                  f"Threshold={best.params['threshold']:.2f}")

    study.optimize(objective, n_trials=n_trials, callbacks=[log_callback])

    # Extract best
    best = study.best_trial
    best_individual = {
        'account_weight': best.user_attrs['account_weight'],
        'transfer_weight': best.user_attrs['transfer_weight'],
        'network_weight': best.user_attrs['network_weight'],
        'threshold': best.params['threshold']
    }

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"Account Weight:   {best_individual['account_weight']:.3f}")
    print(f"Transfer Weight:  {best_individual['transfer_weight']:.3f}")
    print(f"Network Weight:   {best_individual['network_weight']:.3f}")
    print(f"Threshold:        {best_individual['threshold']:.3f}")

    # Save best configuration
    os.makedirs(os.path.join(PROJECT_ROOT, 'experiments', 'shinka'), exist_ok=True)
    config_path = os.path.join(PROJECT_ROOT, 'experiments', 'shinka', 'evolved_config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump({
            'best_individual': best_individual,
            'optimization_history': [
                {'trial': t.number, 'fitness': t.value,
                 'account_weight': t.user_attrs.get('account_weight'),
                 'transfer_weight': t.user_attrs.get('transfer_weight'),
                 'network_weight': t.user_attrs.get('network_weight'),
                 'threshold': t.params.get('threshold')}
                for t in study.trials
            ]
        }, f)

    print(f"\n✓ Optimized configuration saved to: {config_path}")

    # Compare baseline vs optimized
    print("\n" + "=" * 80)
    print("BASELINE VS OPTIMIZED")
    print("=" * 80)
    print(f"                      Baseline    Optimized")
    print(f"Account Weight:       0.250      {best_individual['account_weight']:.3f}")
    print(f"Transfer Weight:      0.450      {best_individual['transfer_weight']:.3f}")
    print(f"Network Weight:       0.300      {best_individual['network_weight']:.3f}")
    print(f"Threshold:            0.300      {best_individual['threshold']:.3f}")
    print("=" * 80)

    print(f"\n✓ Bayesian optimization complete ({n_trials} trials)!")


if __name__ == "__main__":
    main()
