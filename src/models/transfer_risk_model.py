"""
Transfer Risk Model

Scores individual transactions for suspiciousness using ML + engineered features.
Combines with rule-based features from Step 1 baseline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import lightgbm as lgb
import matplotlib.pyplot as plt
import pickle
import os
import sys

# Add parent to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from features.transaction_features import TransactionFeatureEngineer


class TransferRiskModel:
    """
    LightGBM-based transfer risk classifier.

    Predicts P(transaction is laundering | transaction features, account features)
    """

    def __init__(self, model_params=None):
        """Initialize transfer risk model"""
        if model_params is None:
            # Hyperparameters evolved by ShinkaEvolve (gen 19, hardened run)
            model_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 45,
                'learning_rate': 0.03,
                'feature_fraction': 0.75,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': -1,
                'min_child_samples': 30,
                'scale_pos_weight': 25,
                'lambda_l1': 1.5,
                'lambda_l2': 1.5,
                'verbose': -1
            }

        self.model_params = model_params
        self.model = None
        self.feature_names = None
        self.feature_engineer = None


    def train(self, transactions_df, test_size=0.2, random_state=42):
        """
        Train the transfer risk model.

        Args:
            transactions_df: DataFrame with engineered transaction features
            test_size: Fraction for test set
            random_state: Random seed

        Returns:
            Dictionary with training metrics
        """
        # Get features (exclude ground truth for production-safe training)
        self.feature_engineer = TransactionFeatureEngineer()
        feature_cols = self.feature_engineer.get_feature_columns(exclude_ground_truth=True)
        # Filter to columns actually present in the data
        feature_cols = [c for c in feature_cols if c in transactions_df.columns]
        self.feature_names = feature_cols

        # Prepare X and y
        X = transactions_df[feature_cols].fillna(0)
        y = transactions_df['is_laundering'].astype(int)

        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # Train
        print("\nTraining LightGBM model...")
        self.model = lgb.train(
            self.model_params,
            train_data,
            num_boost_round=300,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(30)]
        )

        # Evaluate
        print("\nEvaluating model...")
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Laundering']))

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC AUC: {roc_auc:.4f}")

        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        print("\nTop 15 Most Important Features:")
        print(importance_df.head(15).to_string(index=False))

        return {
            'roc_auc': roc_auc,
            'feature_importance': importance_df,
            'test_predictions': y_pred_proba,
            'y_test': y_test
        }


    def predict_risk_score(self, transactions_df):
        """
        Predict risk scores for transactions.

        Args:
            transactions_df: DataFrame with transaction features

        Returns:
            Array of risk scores (0 to 1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        X = transactions_df[self.feature_names].fillna(0)
        risk_scores = self.model.predict(X)

        return risk_scores


    def save_model(self, filepath):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'model_params': self.model_params
            }, f)
        print(f"✓ Model saved to: {filepath}")


    @classmethod
    def load_model(cls, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        instance = cls(model_params=data['model_params'])
        instance.model = data['model']
        instance.feature_names = data['feature_names']
        instance.feature_engineer = TransactionFeatureEngineer()

        return instance


def plot_feature_importance(importance_df, output_path, top_n=20):
    """Plot feature importance"""
    plt.figure(figsize=(10, 10))
    top_features = importance_df.head(top_n)

    import seaborn as sns
    sns.barplot(data=top_features, y='feature', x='importance', hue='feature', palette='viridis', legend=False)
    plt.xlabel('Importance (Gain)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances\nTransfer Risk Model', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✓ Feature importance plot saved to: {output_path}")
    plt.close()


def main():
    """Train and evaluate transfer risk model"""
    print("=" * 80)
    print("TRANSFER RISK MODEL - TRAINING")
    print("=" * 80)

    # Resolve project root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Load enriched transaction data
    print("\n1. Loading transaction data with features...")
    transactions_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'transactions_with_features.csv'))
    print(f"   Loaded {len(transactions_df)} transactions")
    print(f"   Laundering: {transactions_df['is_laundering'].sum()}")
    print(f"   Normal: {(~transactions_df['is_laundering']).sum()}")

    # Train model
    print("\n2. Training transfer risk model...")
    model = TransferRiskModel()
    results = model.train(transactions_df, test_size=0.2, random_state=42)

    # Save model
    print("\n3. Saving model...")
    os.makedirs(os.path.join(PROJECT_ROOT, 'experiments', 'fusion'), exist_ok=True)
    model.save_model(os.path.join(PROJECT_ROOT, 'experiments', 'fusion', 'transfer_risk_model.pkl'))

    # Generate risk scores for all transactions
    print("\n4. Generating risk scores for all transactions...")
    risk_scores = model.predict_risk_score(transactions_df)
    transactions_df['transfer_risk_score'] = risk_scores

    # Save enriched transactions
    transactions_df.to_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'transactions_with_risk.csv'), index=False)
    print("✓ Enriched transactions saved to: data/processed/transactions_with_risk.csv")

    # Plot feature importance
    print("\n5. Generating visualizations...")
    plot_feature_importance(
        results['feature_importance'],
        os.path.join(PROJECT_ROOT, 'outputs', 'figures', 'transfer_risk_feature_importance.png'),
        top_n=20
    )

    # Summary
    print("\n" + "=" * 80)
    print("TRANSFER RISK MODEL - SUMMARY")
    print("=" * 80)
    print(f"ROC AUC:                 {results['roc_auc']:.4f}")
    print(f"Total transactions:      {len(transactions_df):,}")
    print(f"High-risk transfers:     {(risk_scores > 0.7).sum():,} ({(risk_scores > 0.7).mean()*100:.1f}%)")
    print(f"Medium-risk transfers:   {((risk_scores > 0.4) & (risk_scores <= 0.7)).sum():,}")
    print(f"Low-risk transfers:      {(risk_scores <= 0.4).sum():,} ({(risk_scores <= 0.4).mean()*100:.1f}%)")
    print("=" * 80)

    print("\n✓ Transfer risk model training complete!")


if __name__ == "__main__":
    main()
