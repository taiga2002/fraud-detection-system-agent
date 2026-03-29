"""
Account Opening Risk Model

Predicts the likelihood that a newly opened or IB-enabled account will be used for fraud.
Based on BAF-style features: account age, onboarding channel, device patterns, etc.

This model provides the "account risk prior" for the fusion layer.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os


class AccountRiskModel:
    """
    LightGBM-based account opening risk classifier.

    Predicts P(account is mule or victim | account features)
    """

    def __init__(self, model_params=None):
        """
        Initialize account risk model.

        Args:
            model_params: Optional LightGBM parameters dict
        """
        if model_params is None:
            # Default params optimized for imbalanced fraud detection
            model_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': 6,
                'min_child_samples': 20,
                'scale_pos_weight': 10,  # Handle class imbalance
                'verbose': -1
            }

        self.model_params = model_params
        self.model = None
        self.feature_names = None


    def engineer_features(self, accounts_df):
        """
        Engineer features from raw account data.

        Args:
            accounts_df: DataFrame with account information

        Returns:
            DataFrame with engineered features
        """
        df = accounts_df.copy()

        # Categorical encoding for onboarding channel
        channel_dummies = pd.get_dummies(df['onboarding_channel'], prefix='channel')
        df = pd.concat([df, channel_dummies], axis=1)

        # Customer type encoding
        df['is_business'] = (df['customer_type'] == 'business').astype(int)

        # Account age buckets
        df['account_very_new'] = (df['account_age_days'] < 30).astype(int)
        df['account_new'] = (df['account_age_days'] < 90).astype(int)
        df['account_young'] = (df['account_age_days'] < 180).astype(int)

        # IB enablement timing
        df['ib_recently_enabled'] = ((df['ib_enabled']) &
                                     (df['ib_days_since_enabled'] < 30)).astype(int)
        df['ib_very_recently_enabled'] = ((df['ib_enabled']) &
                                          (df['ib_days_since_enabled'] < 14)).astype(int)

        # IB enablement ratio (days since enabled / account age)
        df['ib_enablement_ratio'] = df['ib_days_since_enabled'] / (df['account_age_days'] + 1)
        df['ib_enablement_ratio'] = df['ib_enablement_ratio'].fillna(0)

        # Suspicious device flag
        df['has_suspicious_device'] = df['suspicious_device'].astype(int)

        # Device ID frequency (how many accounts share this device)
        device_counts = df['device_id'].value_counts()
        df['device_frequency'] = df['device_id'].map(device_counts)
        df['device_shared'] = (df['device_frequency'] > 1).astype(int)

        # Risk flag (target variable)
        df['is_risky'] = (df['is_mule'] | df['is_victim']).astype(int)

        return df


    def get_feature_columns(self):
        """Get list of feature column names for modeling"""
        return [
            'account_age_days',
            'ib_enabled',
            'ib_days_since_enabled',
            'is_business',
            'account_very_new',
            'account_new',
            'account_young',
            'ib_recently_enabled',
            'ib_very_recently_enabled',
            'ib_enablement_ratio',
            'has_suspicious_device',
            'device_frequency',
            'device_shared',
            'channel_branch',
            'channel_mobile_app',
            'channel_online',
            'channel_phone'
        ]


    def train(self, accounts_df, test_size=0.2, random_state=42):
        """
        Train the account risk model.

        Args:
            accounts_df: DataFrame with account data
            test_size: Fraction of data for test set
            random_state: Random seed

        Returns:
            Dictionary with training metrics and feature importance
        """
        # Engineer features
        print("Engineering features...")
        df = self.engineer_features(accounts_df)

        # Prepare features and target
        feature_cols = self.get_feature_columns()
        self.feature_names = feature_cols

        X = df[feature_cols].fillna(0)
        y = df['is_risky']

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

        # Train model
        print("\nTraining LightGBM model...")
        self.model = lgb.train(
            self.model_params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(20)]
        )

        # Evaluate
        print("\nEvaluating model...")
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Risky']))

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC AUC: {roc_auc:.4f}")

        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))

        return {
            'roc_auc': roc_auc,
            'feature_importance': importance_df,
            'test_predictions': y_pred_proba
        }


    def predict_risk_score(self, accounts_df):
        """
        Predict risk scores for accounts.

        Args:
            accounts_df: DataFrame with account data

        Returns:
            Array of risk scores (0 to 1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Engineer features
        df = self.engineer_features(accounts_df)
        X = df[self.feature_names].fillna(0)

        # Predict
        risk_scores = self.model.predict(X)

        return risk_scores


    def save_model(self, filepath):
        """Save trained model to disk"""
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
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        instance = cls(model_params=data['model_params'])
        instance.model = data['model']
        instance.feature_names = data['feature_names']

        return instance


def plot_feature_importance(importance_df, output_path, top_n=15):
    """Plot feature importance"""
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(top_n)

    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.xlabel('Importance (Gain)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances\nAccount Opening Risk Model', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✓ Feature importance plot saved to: {output_path}")
    plt.close()


def main():
    """Train and evaluate account opening risk model"""
    print("=" * 80)
    print("ACCOUNT OPENING RISK MODEL - TRAINING")
    print("=" * 80)

    # Resolve project root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Load account data
    print("\n1. Loading account data...")
    accounts_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'accounts.csv'))
    print(f"   Loaded {len(accounts_df)} accounts")
    print(f"   Mule accounts: {accounts_df['is_mule'].sum()}")
    print(f"   Victim accounts: {accounts_df['is_victim'].sum()}")
    print(f"   Normal accounts: {(~(accounts_df['is_mule'] | accounts_df['is_victim'])).sum()}")

    # Initialize and train model
    print("\n2. Training account risk model...")
    model = AccountRiskModel()
    results = model.train(accounts_df, test_size=0.2, random_state=42)

    # Save model
    print("\n3. Saving model...")
    os.makedirs(os.path.join(PROJECT_ROOT, 'experiments', 'fusion'), exist_ok=True)
    model.save_model(os.path.join(PROJECT_ROOT, 'experiments', 'fusion', 'account_risk_model.pkl'))

    # Generate all account risk scores
    print("\n4. Generating risk scores for all accounts...")
    risk_scores = model.predict_risk_score(accounts_df)
    accounts_df['account_risk_score'] = risk_scores

    # Save enriched accounts
    accounts_df.to_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'accounts_with_risk.csv'), index=False)
    print("✓ Enriched accounts saved to: data/processed/accounts_with_risk.csv")

    # Plot feature importance
    print("\n5. Generating visualizations...")
    os.makedirs(os.path.join(PROJECT_ROOT, 'outputs', 'figures'), exist_ok=True)
    plot_feature_importance(
        results['feature_importance'],
        os.path.join(PROJECT_ROOT, 'outputs', 'figures', 'account_risk_feature_importance.png'),
        top_n=15
    )

    # Summary
    print("\n" + "=" * 80)
    print("ACCOUNT RISK MODEL - SUMMARY")
    print("=" * 80)
    print(f"ROC AUC:                 {results['roc_auc']:.4f}")
    print(f"Total accounts:          {len(accounts_df):,}")
    print(f"High-risk accounts:      {(risk_scores > 0.7).sum():,} ({(risk_scores > 0.7).mean()*100:.1f}%)")
    print(f"Medium-risk accounts:    {((risk_scores > 0.4) & (risk_scores <= 0.7)).sum():,}")
    print(f"Low-risk accounts:       {(risk_scores <= 0.4).sum():,} ({(risk_scores <= 0.4).mean()*100:.1f}%)")
    print("=" * 80)

    print("\n✓ Account opening risk model training complete!")


if __name__ == "__main__":
    main()
