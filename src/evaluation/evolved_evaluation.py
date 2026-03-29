"""
Evolved Policy Evaluation

Evaluate the evolved policy from ShinkaEvolve and compare to:
1. Step 1 Baseline (hand-written rules)
2. Step 2 Fusion Baseline (static weights, threshold 0.3)
3. Step 3 Evolved (optimized threshold 0.319)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve,
    roc_auc_score
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


def load_evolved_config():
    """Load best evolved configuration"""
    with open(os.path.join(PROJECT_ROOT, 'experiments', 'shinka', 'evolved_config.pkl'), 'rb') as f:
        data = pickle.load(f)
        return data['best_individual']


def evaluate_policy(transactions_df, accounts_df, graph_features_dict, config, threshold):
    """
    Evaluate a policy configuration.

    Returns:
        Dictionary with metrics
    """
    fusion = FusionLayer(config)

    predictions = []
    ground_truth = []
    prevented_loss = 0

    for idx, txn in transactions_df.iterrows():
        if idx % 10000 == 0:
            print(f"  Processed {idx}/{len(transactions_df)}...")

        # Get account info
        sender_account = accounts_df[accounts_df['account_id'] == txn['sender']].iloc[0]
        receiver_account = accounts_df[accounts_df['account_id'] == txn['receiver']].iloc[0]
        graph_features = graph_features_dict.get(txn['sender'], {})

        # Get fused score
        account_score = sender_account.get('account_risk_score', 0)
        transfer_score = txn.get('transfer_risk_score', 0)
        network_score = sender_account.get('network_risk_score', 0)

        fused_score = fusion.fuse_scores(account_score, transfer_score, network_score)

        # Predict
        pred = 1 if fused_score >= threshold else 0
        truth = int(txn['is_laundering'])

        predictions.append(pred)
        ground_truth.append(truth)

        # If we correctly flagged a fraud, count prevented loss
        if pred == 1 and truth == 1:
            prevented_loss += txn['amount_jpy']

    # Calculate metrics
    y_true = np.array(ground_truth)
    y_pred = np.array(predictions)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    total_fraud_loss = transactions_df[transactions_df['is_laundering'] == True]['amount_jpy'].sum()
    prevention_rate = prevented_loss / total_fraud_loss if total_fraud_loss > 0 else 0

    alert_rate = y_pred.sum() / len(y_pred)
    total_alerts = int(y_pred.sum())

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'prevented_loss': prevented_loss,
        'prevention_rate': prevention_rate,
        'total_fraud_loss': total_fraud_loss,
        'alert_rate': alert_rate,
        'total_alerts': total_alerts,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }


def main():
    """Main evaluation pipeline"""
    print("=" * 80)
    print("EVOLVED POLICY EVALUATION")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    transactions_df, accounts_df, graph_features_dict = load_data()
    print(f"   {len(transactions_df)} transactions")
    print(f"   {len(accounts_df)} accounts")

    # Load evolved config
    print("\n2. Loading evolved configuration...")
    evolved_config_dict = load_evolved_config()
    print(f"   Account Weight:   {evolved_config_dict['account_weight']:.3f}")
    print(f"   Transfer Weight:  {evolved_config_dict['transfer_weight']:.3f}")
    print(f"   Network Weight:   {evolved_config_dict['network_weight']:.3f}")
    print(f"   Threshold:        {evolved_config_dict['threshold']:.3f}")

    # Prepare configurations
    baseline_config = {
        'account_weight': 0.25,
        'transfer_weight': 0.45,
        'network_weight': 0.30,
        'thresholds': {
            'critical': 0.8,
            'high': 0.6,
            'medium': 0.4,
            'low': 0.3
        }
    }

    evolved_config = {
        'account_weight': evolved_config_dict['account_weight'],
        'transfer_weight': evolved_config_dict['transfer_weight'],
        'network_weight': evolved_config_dict['network_weight'],
        'thresholds': baseline_config['thresholds']  # Use same action thresholds
    }

    evolved_threshold = evolved_config_dict['threshold']

    # Evaluate Step 2 Fusion Baseline (threshold 0.3)
    print("\n3. Evaluating Step 2 Fusion Baseline (threshold=0.3)...")
    fusion_baseline_metrics = evaluate_policy(
        transactions_df, accounts_df, graph_features_dict,
        baseline_config, threshold=0.3
    )

    # Evaluate Step 3 Evolved (threshold 0.319)
    print("\n4. Evaluating Step 3 Evolved Policy (threshold=0.319)...")
    evolved_metrics = evaluate_policy(
        transactions_df, accounts_df, graph_features_dict,
        evolved_config, threshold=evolved_threshold
    )

    # Load Step 1 baseline metrics for comparison
    try:
        baseline_metrics_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'tables', 'baseline_metrics.csv'))
        step1_at_03 = baseline_metrics_df[baseline_metrics_df['threshold'] == 0.3].iloc[0]
        step1_available = True
    except:
        print("   Warning: Step 1 baseline metrics not found")
        step1_available = False

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON: STEP 1 BASELINE vs STEP 2 FUSION vs STEP 3 EVOLVED")
    print("=" * 80)
    print(f"\n{'Metric':<25} {'Step 1 Baseline':<18} {'Step 2 Fusion':<18} {'Step 3 Evolved':<18}")
    print("-" * 80)

    if step1_available:
        print(f"{'Precision':<25} {step1_at_03['precision']:>15.3f}  {fusion_baseline_metrics['precision']:>15.3f}  {evolved_metrics['precision']:>15.3f}")
        print(f"{'Recall':<25} {step1_at_03['recall']:>15.3f}  {fusion_baseline_metrics['recall']:>15.3f}  {evolved_metrics['recall']:>15.3f}")
        print(f"{'F1 Score':<25} {step1_at_03['f1_score']:>15.3f}  {fusion_baseline_metrics['f1_score']:>15.3f}  {evolved_metrics['f1_score']:>15.3f}")
        print(f"{'Alert Rate (%)':<25} {step1_at_03['alert_rate']*100:>15.1f}  {fusion_baseline_metrics['alert_rate']*100:>15.1f}  {evolved_metrics['alert_rate']*100:>15.1f}")
        print(f"{'Total Alerts':<25} {int(step1_at_03['total_alerts']):>15,d}  {fusion_baseline_metrics['total_alerts']:>15,d}  {evolved_metrics['total_alerts']:>15,d}")
    else:
        print(f"{'Precision':<25} {'N/A':<18} {fusion_baseline_metrics['precision']:>15.3f}  {evolved_metrics['precision']:>15.3f}")
        print(f"{'Recall':<25} {'N/A':<18} {fusion_baseline_metrics['recall']:>15.3f}  {evolved_metrics['recall']:>15.3f}")
        print(f"{'F1 Score':<25} {'N/A':<18} {fusion_baseline_metrics['f1_score']:>15.3f}  {evolved_metrics['f1_score']:>15.3f}")
        print(f"{'Alert Rate (%)':<25} {'N/A':<18} {fusion_baseline_metrics['alert_rate']*100:>15.1f}  {evolved_metrics['alert_rate']*100:>15.1f}")
        print(f"{'Total Alerts':<25} {'N/A':<18} {fusion_baseline_metrics['total_alerts']:>15,d}  {evolved_metrics['total_alerts']:>15,d}")

    print(f"\n{'Prevented Loss (¥M)':<25} {'N/A':<18} {fusion_baseline_metrics['prevented_loss']/1e6:>15.1f}  {evolved_metrics['prevented_loss']/1e6:>15.1f}")
    print(f"{'Prevention Rate (%)':<25} {'N/A':<18} {fusion_baseline_metrics['prevention_rate']*100:>15.1f}  {evolved_metrics['prevention_rate']*100:>15.1f}")

    # Improvement analysis
    print("\n" + "=" * 80)
    print("EVOLVED vs FUSION BASELINE IMPROVEMENT")
    print("=" * 80)

    precision_improvement = ((evolved_metrics['precision'] - fusion_baseline_metrics['precision']) /
                            fusion_baseline_metrics['precision'] * 100)
    recall_change = ((evolved_metrics['recall'] - fusion_baseline_metrics['recall']) /
                    fusion_baseline_metrics['recall'] * 100)
    f1_improvement = ((evolved_metrics['f1_score'] - fusion_baseline_metrics['f1_score']) /
                     fusion_baseline_metrics['f1_score'] * 100)
    alert_reduction = ((fusion_baseline_metrics['total_alerts'] - evolved_metrics['total_alerts']) /
                      fusion_baseline_metrics['total_alerts'] * 100)

    print(f"Precision:        {fusion_baseline_metrics['precision']:.3f} → {evolved_metrics['precision']:.3f} "
          f"({'+'if precision_improvement > 0 else ''}{precision_improvement:.1f}%)")
    print(f"Recall:           {fusion_baseline_metrics['recall']:.3f} → {evolved_metrics['recall']:.3f} "
          f"({'+'if recall_change > 0 else ''}{recall_change:.1f}%)")
    print(f"F1 Score:         {fusion_baseline_metrics['f1_score']:.3f} → {evolved_metrics['f1_score']:.3f} "
          f"({'+'if f1_improvement > 0 else ''}{f1_improvement:.1f}%)")
    print(f"Alert Reduction:  {fusion_baseline_metrics['total_alerts']:,} → {evolved_metrics['total_alerts']:,} "
          f"({'-' if alert_reduction > 0 else '+'}{abs(alert_reduction):.1f}%)")

    # Save results
    print("\n5. Saving results...")
    os.makedirs(os.path.join(PROJECT_ROOT, 'outputs', 'tables'), exist_ok=True)

    results_df = pd.DataFrame([
        {'policy': 'Step 2 Fusion Baseline', 'threshold': 0.3, **fusion_baseline_metrics},
        {'policy': 'Step 3 Evolved', 'threshold': evolved_threshold, **evolved_metrics}
    ])

    results_df.to_csv(os.path.join(PROJECT_ROOT, 'outputs', 'tables', 'evolved_comparison.csv'), index=False)
    print("✓ Results saved to: outputs/tables/evolved_comparison.csv")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Evolved policy achieved {evolved_metrics['precision']:.1%} precision at {evolved_metrics['recall']:.1%} recall")
    print(f"✓ Prevented ¥{evolved_metrics['prevented_loss']/1e6:.1f}M fraud loss ({evolved_metrics['prevention_rate']:.1%} of total)")
    print(f"✓ Generated {evolved_metrics['total_alerts']:,} alerts ({evolved_metrics['alert_rate']:.1%} alert rate)")
    print(f"✓ Evolution optimized threshold from 0.300 to {evolved_threshold:.3f}")

    print("\n" + "=" * 80)
    print("✓ Step 3 evolved policy evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
