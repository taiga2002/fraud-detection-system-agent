"""
Fusion System Evaluation

Evaluate the full fusion system (account + transfer + network) and compare to Step 1 baseline.
Target: >85% precision while maintaining high recall.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve,
    roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scoring.fusion_layer import FusionLayer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_data():
    """Load all enriched data"""
    transactions_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'transactions_with_risk.csv'))
    accounts_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'accounts_with_network_risk.csv'))

    # Load graph features
    with open(os.path.join(PROJECT_ROOT, 'experiments', 'fusion', 'mule_network_detector.pkl'), 'rb') as f:
        detector_data = pickle.load(f)
        graph_features_dict = detector_data['account_graph_features']

    return transactions_df, accounts_df, graph_features_dict


def evaluate_fusion(transactions_df, accounts_df, graph_features_dict):
    """
    Run fusion system on all transactions.

    Returns:
        DataFrame with fusion predictions
    """
    print("Evaluating fusion system...")

    fusion = FusionLayer()
    results = []

    for idx, txn in transactions_df.iterrows():
        if idx % 10000 == 0:
            print(f"  Processed {idx}/{len(transactions_df)} transactions...")

        # Get account info
        sender_account = accounts_df[accounts_df['account_id'] == txn['sender']].iloc[0]
        receiver_account = accounts_df[accounts_df['account_id'] == txn['receiver']].iloc[0]

        # Get graph features for sender
        graph_features = graph_features_dict.get(txn['sender'], {})

        # Evaluate
        decision = fusion.evaluate_transaction(
            txn,
            sender_account,
            receiver_account,
            graph_features
        )

        results.append({
            'txn_id': txn['txn_id'],
            'ground_truth': txn['is_laundering'],
            'fused_score': decision.suspicious_transfer_score,
            'mule_score': decision.mule_account_score,
            'account_score': decision.account_risk_score,
            'transfer_score': decision.transfer_risk_score,
            'network_score': decision.network_risk_score,
            'action': decision.action.value,
            'risk_level': decision.risk_level.value,
            'num_reason_codes': len(decision.reason_codes),
            'amount_jpy': txn['amount_jpy']
        })

    print(f"  Completed {len(transactions_df)} transactions.")
    return pd.DataFrame(results)


def generate_metrics(results_df, threshold=0.5):
    """Generate classification metrics"""
    y_true = results_df['ground_truth'].values
    y_pred = (results_df['fused_score'] >= threshold).astype(int)
    y_scores = results_df['fused_score'].values

    # Basic metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except:
        roc_auc = 0.0

    # Alert metrics
    total_alerts = y_pred.sum()
    alert_rate = total_alerts / len(y_pred)

    # High-loss recall
    high_loss_mask = results_df['amount_jpy'] >= 5000000
    if high_loss_mask.sum() > 0:
        high_loss_recall = recall_score(
            y_true[high_loss_mask],
            y_pred[high_loss_mask],
            zero_division=0
        )
    else:
        high_loss_recall = 0.0

    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'total_alerts': int(total_alerts),
        'alert_rate': alert_rate,
        'high_loss_recall': high_loss_recall
    }


def plot_comparison(fusion_results_df, baseline_results_df, output_path):
    """Plot PR curves comparing fusion to baseline"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # PR Curve
    y_true = fusion_results_df['ground_truth'].values

    # Fusion
    fusion_precision, fusion_recall, _ = precision_recall_curve(
        y_true, fusion_results_df['fused_score'].values
    )
    ax1.plot(fusion_recall, fusion_precision, linewidth=2, label='Fusion System', color='#2E86AB')

    # Baseline
    baseline_precision, baseline_recall, _ = precision_recall_curve(
        y_true, baseline_results_df['risk_score'].values
    )
    ax1.plot(baseline_recall, baseline_precision, linewidth=2, label='Step 1 Baseline', color='#A23B72', linestyle='--')

    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curve Comparison', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Score distribution
    ax2.hist(fusion_results_df[fusion_results_df['ground_truth']==1]['fused_score'],
             bins=20, alpha=0.7, label='Fusion: Fraud', color='red', edgecolor='black')
    ax2.hist(fusion_results_df[fusion_results_df['ground_truth']==0]['fused_score'],
             bins=20, alpha=0.5, label='Fusion: Normal', color='blue', edgecolor='black')

    ax2.set_xlabel('Risk Score', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Risk Score Distribution (Fusion)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✓ Comparison plot saved to: {output_path}")
    plt.close()


def main():
    """Main fusion evaluation pipeline"""
    print("=" * 80)
    print("FUSION SYSTEM EVALUATION")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    transactions_df, accounts_df, graph_features_dict = load_data()
    print(f"   {len(transactions_df)} transactions")
    print(f"   {len(accounts_df)} accounts")
    print(f"   {len(graph_features_dict)} graph feature records")

    # Evaluate fusion system
    print("\n2. Running fusion evaluation...")
    fusion_results = evaluate_fusion(transactions_df, accounts_df, graph_features_dict)

    # Load baseline results for comparison
    print("\n3. Loading Step 1 baseline results...")
    try:
        # Re-run baseline evaluation to get comparable results
        from baseline_evaluation import evaluate_baseline, load_config
        from models.baseline_rules import BaselineRuleEngine

        config = load_config()
        baseline_engine = BaselineRuleEngine(config)
        baseline_results = evaluate_baseline(baseline_engine, transactions_df, accounts_df)
    except Exception as e:
        print(f"   Warning: Could not load baseline results: {e}")
        baseline_results = None

    # Generate metrics at multiple thresholds
    print("\n4. Calculating metrics...")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    fusion_metrics = []

    for threshold in thresholds:
        metrics = generate_metrics(fusion_results, threshold=threshold)
        fusion_metrics.append(metrics)
        print(f"   Threshold {threshold:.1f}: Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")

    # Save fusion metrics
    fusion_metrics_df = pd.DataFrame(fusion_metrics)
    os.makedirs(os.path.join(PROJECT_ROOT, 'outputs', 'tables'), exist_ok=True)
    fusion_metrics_df.to_csv(os.path.join(PROJECT_ROOT, 'outputs', 'tables', 'fusion_metrics.csv'), index=False)
    print("\n✓ Fusion metrics saved to: outputs/tables/fusion_metrics.csv")

    # Generate plots
    print("\n5. Generating visualizations...")
    os.makedirs(os.path.join(PROJECT_ROOT, 'outputs', 'figures'), exist_ok=True)

    if baseline_results is not None:
        plot_comparison(
            fusion_results,
            baseline_results,
            os.path.join(PROJECT_ROOT, 'outputs', 'figures', 'fusion_vs_baseline.png')
        )

    # Summary
    print("\n" + "=" * 80)
    print("FUSION SYSTEM SUMMARY (Threshold = 0.5)")
    print("=" * 80)
    best_metrics = fusion_metrics[2]  # threshold=0.5
    print(f"Precision:             {best_metrics['precision']:.3f}")
    print(f"Recall:                {best_metrics['recall']:.3f}")
    print(f"F1 Score:              {best_metrics['f1_score']:.3f}")
    print(f"ROC AUC:               {best_metrics['roc_auc']:.3f}")
    print(f"Total Alerts:          {best_metrics['total_alerts']:,} ({best_metrics['alert_rate']*100:.1f}%)")
    print(f"High-Loss Recall:      {best_metrics['high_loss_recall']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:      {best_metrics['true_positives']:,}")
    print(f"  False Positives:     {best_metrics['false_positives']:,}")
    print(f"  True Negatives:      {best_metrics['true_negatives']:,}")
    print(f"  False Negatives:     {best_metrics['false_negatives']:,}")

    # Compare to baseline if available
    if baseline_results is not None:
        baseline_metrics_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'tables', 'baseline_metrics.csv'))
        baseline_at_05 = baseline_metrics_df[baseline_metrics_df['threshold'] == 0.5].iloc[0]

        print("\n" + "=" * 80)
        print("IMPROVEMENT OVER BASELINE")
        print("=" * 80)
        precision_improvement = ((best_metrics['precision'] - baseline_at_05['precision']) /
                                baseline_at_05['precision'] * 100)
        recall_improvement = ((best_metrics['recall'] - baseline_at_05['recall']) /
                            baseline_at_05['recall'] * 100)

        print(f"Precision:     {baseline_at_05['precision']:.3f} → {best_metrics['precision']:.3f} "
              f"({'+'if precision_improvement > 0 else ''}{precision_improvement:.1f}%)")
        print(f"Recall:        {baseline_at_05['recall']:.3f} → {best_metrics['recall']:.3f} "
              f"({'+'if recall_improvement > 0 else ''}{recall_improvement:.1f}%)")
        print(f"Alert Rate:    {baseline_at_05['alert_rate']*100:.1f}% → {best_metrics['alert_rate']*100:.1f}%")

    print("\n" + "=" * 80)
    print("✓ Step 2 fusion evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
