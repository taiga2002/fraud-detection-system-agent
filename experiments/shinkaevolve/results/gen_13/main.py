"""
Evolvable ML Pipeline for Japanese Banking Fraud Detection

This program is evolved by ShinkaEvolve. Code inside EVOLVE-BLOCK markers
can be mutated by the LLM. Everything outside is fixed infrastructure.

Outputs JSON metrics to stdout for the evaluator to parse.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import lightgbm as lgb

# ─── Fixed infrastructure: locate project root and load data ───
# Find project root by looking for data/processed/ directory
# Works regardless of where ShinkaEvolve copies this file
def _find_project_root():
    """Walk up from cwd and __file__ to find the project root with data/processed/."""
    candidates = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
    ]
    for start in candidates:
        path = os.path.abspath(start)
        for _ in range(10):
            if os.path.isdir(os.path.join(path, 'data', 'processed')):
                return path
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
    raise FileNotFoundError("Cannot find project root with data/processed/ directory")

PROJECT_ROOT = _find_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')


def load_data(seed=42):
    """Load and subsample transaction data for fast evaluation."""
    txn_path = os.path.join(DATA_DIR, 'transactions_with_features.csv')
    acc_path = os.path.join(DATA_DIR, 'accounts_with_risk.csv')

    txn_df = pd.read_csv(txn_path)
    acc_df = pd.read_csv(acc_path)

    # Subsample to 10K rows for speed (stratified by target)
    if len(txn_df) > 10000:
        txn_df, _ = train_test_split(
            txn_df, train_size=10000, random_state=seed,
            stratify=txn_df['is_laundering']
        )
        txn_df = txn_df.reset_index(drop=True)

    return txn_df, acc_df


# ─── Available columns in txn_df after feature engineering ───
# amount_jpy, amount_log, is_high_amount, is_very_high_amount,
# balance_drain_ratio, first_transfer,
# sender_account_age_days, sender_ib_enabled, sender_ib_days_since_enabled,
# sender_suspicious_device, sender_account_risk_score,
# receiver_account_age_days, receiver_is_new, receiver_is_very_new,
# txn_count_24h, txn_volume_24h, has_recent_activity,
# sender_mean_amount, sender_std_amount, sender_txn_count,
# amount_zscore, amount_above_mean, amount_much_above_mean,
# hour, day_of_week, is_weekend, is_business_hours, is_late_night


# EVOLVE-BLOCK-START
def engineer_evolved_features(df):
    """
    Create interaction features from the base transaction columns.

    Available numeric columns:
      amount_jpy, amount_log, balance_drain_ratio,
      sender_account_age_days, sender_ib_days_since_enabled,
      sender_account_risk_score, receiver_account_age_days,
      txn_count_24h, txn_volume_24h, sender_mean_amount,
      sender_std_amount, sender_txn_count, amount_zscore, hour

    Available binary columns:
      is_high_amount, is_very_high_amount, first_transfer,
      sender_ib_enabled, sender_suspicious_device,
      receiver_is_new, receiver_is_very_new, has_recent_activity,
      amount_above_mean, amount_much_above_mean,
      is_weekend, is_business_hours, is_late_night

    Returns a DataFrame with new feature columns only.
    """
    feats = pd.DataFrame(index=df.index)

    # Interaction between recent IB enablement and account age
    # Balance drain with recent IB enablement
    # New interaction features
    feats['balance_drain_recent_ib'] = (
        df['balance_drain_ratio'].fillna(0) *
        np.where(df['sender_ib_days_since_enabled'].fillna(9999) <= 30, 1, 0)).clip(lower=0)

    feats['recent_ib_age_ratio'] = (df['sender_ib_days_since_enabled'].fillna(9999) / (df['sender_account_age_days'] + 1)).clip(lower=0.01, upper=None)

    # Relative amount to mean transaction in recent window
    feats['amount_to_mean_ratio'] = df['amount_jpy'] / (df['sender_mean_amount'] + 1)

    # Velocity intensity: volume per transaction in recent window
    feats['avg_recent_txn_amount'] = (df['txn_volume_24h'] / (df['txn_count_24h'] + 1)).clip(lower=0.0, upper=None)

    # Interaction between new account and high amount
    feats['new_acct_high_amount'] = df['receiver_is_new'].fillna(0) * df['is_high_amount'].fillna(0)

    # Balance drain with first transfer interaction
    feats['drain_first_transfer_interaction'] = (df['balance_drain_ratio'].fillna(0) * df['first_transfer'].fillna(0)).clip(lower=0.0, upper=None)

    # Interaction between late night and high amount risk
    feats['late_night_high_amount'] = df['is_late_night'].fillna(0) * df['is_high_amount'].fillna(0)

    # Interaction between high velocity and suspicious device
    # Interaction between high velocity and suspicious device, with numeric component for weight
    feats['high_velocity_weighted_device'] = (df['txn_count_24h'].fillna(0) * df['sender_suspicious_device'].fillna(0)).clip(lower=0)

    return feats

# EVOLVE-BLOCK-END


# EVOLVE-BLOCK-START
def get_model_params():
    """
    Return LightGBM parameters for the fraud detection model.

    Key considerations:
    - Class imbalance: ~5% fraud vs 95% normal (use scale_pos_weight)
    - Overfitting risk: use regularization and random feature/subsampling
    - Speed: moderate num_leaves and max_depth for fast training

    Returns a dict of LightGBM parameters.
    """
    return {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        # Reduced complexity to improve regularization and training speed
'num_leaves': 25,
        # Smaller learning rate for better convergence with more regularized model
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.75,
        'bagging_freq': 6,
        'max_depth': -1,  # allow unlimited depth for flexibility
        'min_child_samples': 45,
        'scale_pos_weight': 15,  # higher weight to handle class imbalance
        'lambda_l1': 0.2,
        'lambda_l2': 1.0,
        'verbose': -1,
    }
# EVOLVE-BLOCK-END


# EVOLVE-BLOCK-START
def compute_fusion_score(transfer_risk_scores, account_risk_scores):
    """
    Combine transfer risk and account risk into a final suspicious-transfer score.

    Args:
        transfer_risk_scores: numpy array of P(fraud) from the LightGBM model
        account_risk_scores: numpy array of sender_account_risk_score

    Returns:
        fused_scores: numpy array of final risk scores (0-1)
        threshold: float, decision threshold for alerting
    """
    # Weighted fusion: transfer risk dominates, account risk amplifies non-linearly
    w_transfer = 0.85
    w_account = np.power(account_risk_scores + 0.1, 2) * 0.15

    fused = w_transfer * transfer_risk_scores + w_account

    # Adaptive threshold: use percentile-based to handle varying score ranges
    # Target ~5-8% alert rate (matching approximate fraud rate)
    threshold = float(np.percentile(fused, 92))

    return fused, threshold
# EVOLVE-BLOCK-END


# ─── Fixed infrastructure: training, evaluation, and output ───

BASE_FEATURE_COLS = [
    'amount_jpy', 'amount_log', 'is_high_amount', 'is_very_high_amount',
    'balance_drain_ratio', 'first_transfer',
    'sender_account_age_days', 'sender_ib_enabled', 'sender_ib_days_since_enabled',
    'sender_suspicious_device', 'sender_account_risk_score',
    'receiver_account_age_days', 'receiver_is_new', 'receiver_is_very_new',
    'txn_count_24h', 'txn_volume_24h', 'has_recent_activity',
    'sender_mean_amount', 'sender_std_amount', 'sender_txn_count',
    'amount_zscore', 'amount_above_mean', 'amount_much_above_mean',
    'hour', 'day_of_week', 'is_weekend', 'is_business_hours', 'is_late_night',
]


def evaluate(seed=42):
    """
    Full pipeline: load data, engineer features, train model, evaluate.
    Returns a dict of metrics.
    """
    txn_df, acc_df = load_data(seed=seed)

    # Engineer evolved features
    evolved_feats = engineer_evolved_features(txn_df)

    # Combine base + evolved features
    available_base = [c for c in BASE_FEATURE_COLS if c in txn_df.columns]
    X_base = txn_df[available_base].fillna(0)
    X = pd.concat([X_base, evolved_feats], axis=1)
    y = txn_df['is_laundering'].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Get model params from evolvable block
    params = get_model_params()

    # Train LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        valid_names=['val'],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)],
    )

    # Predict
    y_pred_proba = model.predict(X_test)

    # Get account risk scores for fusion
    account_risk_col = 'sender_account_risk_score'
    if account_risk_col in txn_df.columns:
        account_risk_test = txn_df.loc[X_test.index, account_risk_col].fillna(0).values
    else:
        account_risk_test = np.zeros(len(X_test))

    # Compute fusion score
    fused_scores, threshold = compute_fusion_score(y_pred_proba, account_risk_test)

    # Compute metrics
    y_pred_binary = (fused_scores >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_test, fused_scores))
    f1 = float(f1_score(y_test, y_pred_binary, zero_division=0))
    precision = float(precision_score(y_test, y_pred_binary, zero_division=0))
    recall = float(recall_score(y_test, y_pred_binary, zero_division=0))

    # Prevention rate: fraction of actual fraud that is caught
    n_fraud = int(y_test.sum())
    n_fraud_caught = int((y_pred_binary & y_test).sum()) if n_fraud > 0 else 0
    prevention_rate = n_fraud_caught / max(n_fraud, 1)

    # Alert quality: penalize degenerate alert rates
    alert_rate = float(y_pred_binary.mean())
    if alert_rate < 0.01 or alert_rate > 0.5:
        alert_quality = 0.0
    else:
        # Ideal alert rate 2-10%; penalize outside this range
        if 0.02 <= alert_rate <= 0.10:
            alert_quality = 1.0
        elif alert_rate < 0.02:
            alert_quality = alert_rate / 0.02
        else:
            alert_quality = max(0.0, 1.0 - (alert_rate - 0.10) / 0.40)

    # Composite fitness score
    combined_score = (
        0.40 * roc_auc +
        0.30 * f1 +
        0.20 * prevention_rate +
        0.10 * alert_quality
    )

    metrics = {
        'combined_score': combined_score,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'prevention_rate': prevention_rate,
        'alert_rate': alert_rate,
        'alert_quality': alert_quality,
        'n_fraud_total': n_fraud,
        'n_fraud_caught': n_fraud_caught,
        'n_evolved_features': len(evolved_feats.columns),
        'threshold': float(threshold),
    }

    return metrics


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    try:
        result = evaluate(seed=seed)
        print(json.dumps(result))
    except Exception as e:
        # Output zero metrics on failure
        print(json.dumps({
            'combined_score': 0.0,
            'error': str(e),
        }))
        sys.exit(1)