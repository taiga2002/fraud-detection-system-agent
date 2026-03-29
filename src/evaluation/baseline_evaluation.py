"""
Baseline Rule Engine Evaluation

Evaluate the baseline hand-written rule engine on synthetic data and generate:
1. Metrics table (precision, recall, F1, etc.)
2. Precision-Recall curve plot
3. Confusion matrix
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
import yaml

# Add parent directory to path to import baseline_rules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.baseline_rules import BaselineRuleEngine, Action


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_data():
    """Load synthetic accounts and transactions data"""
    accounts_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'accounts.csv'))
    transactions_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'transactions.csv'))
    return accounts_df, transactions_df


def load_config():
    """Load japan_calibration.yaml"""
    with open(os.path.join(PROJECT_ROOT, 'configs', 'japan_calibration.yaml'), 'r') as f:
        return yaml.safe_load(f)


def evaluate_baseline(engine, transactions_df, accounts_df):
    """
    Run baseline rule engine on all transactions and collect decisions.

    Args:
        engine: BaselineRuleEngine instance
        transactions_df: DataFrame of transactions
        accounts_df: DataFrame of accounts

    Returns:
        DataFrame with predictions and ground truth
    """
    results = []

    print(f"Evaluating {len(transactions_df)} transactions...")

    for idx, txn in transactions_df.iterrows():
        if idx % 5000 == 0:
            print(f"  Processed {idx}/{len(transactions_df)} transactions...")

        # Get account info
        sender_id = txn['sender']
        receiver_id = txn['receiver']

        sender_info = accounts_df[accounts_df['account_id'] == sender_id].iloc[0].to_dict()
        receiver_info = accounts_df[accounts_df['account_id'] == receiver_id].iloc[0].to_dict()

        # Add placeholder account risk score based on observable features only
        # In production, this comes from the account risk model (not ground truth)
        risk_score = 0.1  # Default low risk
        if sender_info.get('account_age_days', 999) < 30:
            risk_score += 0.3
        if sender_info.get('ib_days_since_enabled', 999) < 14:
            risk_score += 0.3
        if sender_info.get('suspicious_device', False):
            risk_score += 0.3
        sender_info['account_risk_score'] = min(risk_score, 1.0)

        # Evaluate transaction
        decision = engine.evaluate_transaction(
            txn.to_dict(),
            sender_info,
            receiver_info
        )

        # Store result
        results.append({
            'txn_id': txn['txn_id'],
            'ground_truth': txn['is_laundering'],
            'risk_score': decision.risk_score,
            'action': decision.action.value,
            'risk_level': decision.risk_level.value,
            'num_reason_codes': len(decision.reason_codes),
            'amount_jpy': txn['amount_jpy']
        })

    print(f"  Completed evaluation of {len(transactions_df)} transactions.")
    return pd.DataFrame(results)


def generate_metrics(results_df, alert_threshold=0.5):
    """
    Generate classification metrics.

    Args:
        results_df: DataFrame with predictions and ground truth
        alert_threshold: Risk score threshold for binary classification

    Returns:
        Dictionary of metrics
    """
    # Binary predictions based on risk score threshold
    y_true = results_df['ground_truth'].values
    y_pred = (results_df['risk_score'] >= alert_threshold).astype(int)
    y_scores = results_df['risk_score'].values

    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # False positive rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except:
        roc_auc = 0.0

    # Alert volume metrics
    total_alerts = y_pred.sum()
    alert_rate = total_alerts / len(y_pred)

    # High-loss case recall (>¥5M threshold from NPA)
    high_loss_mask = results_df['amount_jpy'] >= 5000000
    high_loss_true = y_true[high_loss_mask]
    high_loss_pred = y_pred[high_loss_mask]

    if len(high_loss_true) > 0:
        high_loss_recall = recall_score(high_loss_true, high_loss_pred, zero_division=0)
    else:
        high_loss_recall = 0.0

    metrics = {
        'threshold': alert_threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'false_positive_rate': fpr,
        'total_alerts': int(total_alerts),
        'alert_rate': alert_rate,
        'high_loss_recall': high_loss_recall
    }

    return metrics


def plot_precision_recall_curve(results_df, output_path):
    """Generate precision-recall curve plot"""
    y_true = results_df['ground_truth'].values
    y_scores = results_df['risk_score'].values

    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(10, 6))
    plt.plot(recall_vals, precision_vals, linewidth=2, label='Baseline Rule Engine')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve\nJapan Online-Banking Scam Detection (Baseline)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✓ Precision-Recall curve saved to: {output_path}")
    plt.close()


def plot_confusion_matrix(results_df, alert_threshold, output_path):
    """Generate confusion matrix heatmap"""
    y_true = results_df['ground_truth'].values
    y_pred = (results_df['risk_score'] >= alert_threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Suspicious'],
                yticklabels=['Normal', 'Suspicious'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'Confusion Matrix (Threshold={alert_threshold})\nBaseline Rule Engine', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✓ Confusion matrix saved to: {output_path}")
    plt.close()


def main():
    """Main evaluation pipeline"""
    print("=" * 80)
    print("BASELINE RULE ENGINE EVALUATION")
    print("=" * 80)

    # Load data and config
    print("\n1. Loading data and configuration...")
    accounts_df, transactions_df = load_data()
    config = load_config()

    print(f"   Loaded {len(accounts_df)} accounts")
    print(f"   Loaded {len(transactions_df)} transactions")
    print(f"   Ground truth: {transactions_df['is_laundering'].sum()} laundering, "
          f"{(~transactions_df['is_laundering']).sum()} normal")

    # Initialize engine
    print("\n2. Initializing baseline rule engine...")
    engine = BaselineRuleEngine(config)

    # Evaluate on all transactions
    print("\n3. Running baseline evaluation...")
    results_df = evaluate_baseline(engine, transactions_df, accounts_df)

    # Generate metrics for multiple thresholds
    print("\n4. Calculating metrics...")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    metrics_list = []

    for threshold in thresholds:
        metrics = generate_metrics(results_df, alert_threshold=threshold)
        metrics_list.append(metrics)
        print(f"   Threshold {threshold:.1f}: Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")

    # Save metrics table
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(PROJECT_ROOT, 'outputs', 'tables', 'baseline_metrics.csv')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Metrics table saved to: {metrics_path}")

    # Generate plots
    print("\n5. Generating visualizations...")
    os.makedirs(os.path.join(PROJECT_ROOT, 'outputs', 'figures'), exist_ok=True)

    plot_precision_recall_curve(
        results_df,
        os.path.join(PROJECT_ROOT, 'outputs', 'figures', 'baseline_pr_curve.png')
    )

    plot_confusion_matrix(
        results_df,
        alert_threshold=0.5,
        output_path=os.path.join(PROJECT_ROOT, 'outputs', 'figures', 'baseline_confusion_matrix.png')
    )

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY (Threshold = 0.5)")
    print("=" * 80)
    best_metrics = metrics_list[2]  # threshold=0.5
    print(f"Precision:             {best_metrics['precision']:.3f}")
    print(f"Recall:                {best_metrics['recall']:.3f}")
    print(f"F1 Score:              {best_metrics['f1_score']:.3f}")
    print(f"ROC AUC:               {best_metrics['roc_auc']:.3f}")
    print(f"False Positive Rate:   {best_metrics['false_positive_rate']:.3f}")
    print(f"Total Alerts:          {best_metrics['total_alerts']:,} ({best_metrics['alert_rate']*100:.1f}%)")
    print(f"High-Loss Recall:      {best_metrics['high_loss_recall']:.3f} (cases >¥5M)")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:      {best_metrics['true_positives']:,}")
    print(f"  False Positives:     {best_metrics['false_positives']:,}")
    print(f"  True Negatives:      {best_metrics['true_negatives']:,}")
    print(f"  False Negatives:     {best_metrics['false_negatives']:,}")
    print("=" * 80)

    print("\n✓ Step 1 baseline evaluation complete!")
    print("\nOutputs generated:")
    print("  - Metrics table: outputs/tables/baseline_metrics.csv")
    print("  - PR curve: outputs/figures/baseline_pr_curve.png")
    print("  - Confusion matrix: outputs/figures/baseline_confusion_matrix.png")


if __name__ == "__main__":
    main()
