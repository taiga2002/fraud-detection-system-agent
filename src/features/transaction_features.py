"""
Transaction Feature Engineering

Creates behavioral features from transaction data:
- Velocity: transaction count and volume over time windows
- Deviation: how much transaction differs from account's normal behavior
- Temporal patterns: time-of-day, day-of-week
- Peer comparison: how account compares to similar accounts
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


class TransactionFeatureEngineer:
    """
    Engineer transaction-level features for fraud detection.
    """

    def __init__(self, lookback_hours=24):
        """
        Initialize feature engineer.

        Args:
            lookback_hours: How many hours to look back for velocity features
        """
        self.lookback_hours = lookback_hours


    def engineer_features(self, transactions_df, accounts_df):
        """
        Engineer all transaction features.

        Args:
            transactions_df: DataFrame with transactions
            accounts_df: DataFrame with account info (including risk scores)

        Returns:
            DataFrame with engineered features
        """
        df = transactions_df.copy()

        # Convert timestamp if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        print("Engineering transaction features...")

        # 1. Basic transaction features
        df = self._add_basic_features(df)

        # 2. Account-level features (join from accounts)
        df = self._add_account_features(df, accounts_df)

        # 3. Velocity features (transaction count and volume in recent window)
        df = self._add_velocity_features(df)

        # 4. Deviation features (how unusual is this transaction for this account)
        df = self._add_deviation_features(df)

        # 5. Temporal features
        df = self._add_temporal_features(df)

        # 6. Receiver risk features
        df = self._add_receiver_features(df, accounts_df)

        # 7. ShinkaEvolve-discovered features
        df = self._add_evolved_features(df)

        print(f"✓ Engineered {len(df.columns)} total features")

        return df


    def _add_basic_features(self, df):
        """Add basic transaction amount and flag features"""
        df['amount_log'] = np.log1p(df['amount_jpy'])
        df['is_high_amount'] = (df['amount_jpy'] > 1000000).astype(int)  # >¥1M
        df['is_very_high_amount'] = (df['amount_jpy'] > 5000000).astype(int)  # >¥5M

        return df


    def _add_account_features(self, df, accounts_df):
        """Join account-level features (production-safe: no ground truth)"""
        # Merge sender account info - EXCLUDE ground truth (is_mule, is_victim)
        sender_cols = ['account_id', 'account_age_days', 'ib_enabled',
                      'ib_days_since_enabled', 'suspicious_device']

        if 'account_risk_score' in accounts_df.columns:
            sender_cols.append('account_risk_score')

        sender_info = accounts_df[sender_cols].copy()
        sender_info.columns = ['sender_' + col if col != 'account_id' else 'sender'
                               for col in sender_info.columns]

        df = df.merge(sender_info, on='sender', how='left')

        # Merge receiver account info - EXCLUDE ground truth (is_mule)
        receiver_info = accounts_df[['account_id', 'account_age_days']].copy()
        receiver_info.columns = ['receiver', 'receiver_account_age_days']

        df = df.merge(receiver_info, on='receiver', how='left')

        return df


    def _add_velocity_features(self, df):
        """
        Add velocity features: transaction count and volume in recent time window
        """
        # Sort by account and timestamp
        df_sorted = df.sort_values(['sender', 'timestamp']).copy()

        # For each transaction, count how many transactions this account had in last 24h
        velocity_features = []

        for idx, row in df_sorted.iterrows():
            sender = row['sender']
            timestamp = row['timestamp']

            # Get all transactions for this sender before this one
            prior_txns = df_sorted[
                (df_sorted['sender'] == sender) &
                (df_sorted['timestamp'] < timestamp)
            ]

            # Count transactions in last 24 hours
            recent_window = timestamp - timedelta(hours=self.lookback_hours)
            recent_txns = prior_txns[prior_txns['timestamp'] >= recent_window]

            velocity_features.append({
                'txn_count_24h': len(recent_txns),
                'txn_volume_24h': recent_txns['amount_jpy'].sum() if len(recent_txns) > 0 else 0,
                'has_recent_activity': int(len(recent_txns) > 0)
            })

        velocity_df = pd.DataFrame(velocity_features, index=df_sorted.index)
        df = df_sorted.join(velocity_df)

        # Reset index to match original
        df = df.sort_index()

        return df


    def _add_deviation_features(self, df):
        """
        Add features measuring how this transaction deviates from account's normal behavior
        """
        # For each account, calculate historical mean and std
        account_stats = df.groupby('sender')['amount_jpy'].agg(['mean', 'std', 'count'])
        account_stats.columns = ['sender_mean_amount', 'sender_std_amount', 'sender_txn_count']

        df = df.merge(account_stats, left_on='sender', right_index=True, how='left')

        # Deviation from mean (z-score)
        df['amount_zscore'] = (
            (df['amount_jpy'] - df['sender_mean_amount']) /
            (df['sender_std_amount'] + 1)  # +1 to avoid division by zero
        )
        df['amount_zscore'] = df['amount_zscore'].fillna(0)

        # Is this amount higher than normal?
        df['amount_above_mean'] = (df['amount_jpy'] > df['sender_mean_amount']).astype(int)
        df['amount_much_above_mean'] = (df['amount_jpy'] > df['sender_mean_amount'] * 3).astype(int)

        return df


    def _add_temporal_features(self, df):
        """Add time-based features"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Business hours (9-17)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17)).astype(int)

        # Late night (22-6)
        df['is_late_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)

        return df


    def _add_receiver_features(self, df, accounts_df):
        """Add features about the receiving account (production-safe: no ground truth)"""
        # Receiver is new account
        df['receiver_is_new'] = (df['receiver_account_age_days'] < 90).astype(int)

        # Receiver is very new
        df['receiver_is_very_new'] = (df['receiver_account_age_days'] < 30).astype(int)

        # NOTE: receiver_is_mule REMOVED - ground truth not available in production

        return df


    def _add_evolved_features(self, df):
        """Add features discovered by ShinkaEvolve evolutionary search"""
        import sys
        features_dir = os.path.dirname(os.path.abspath(__file__))
        if features_dir not in sys.path:
            sys.path.insert(0, features_dir)
        from evolved_features import engineer_evolved_features
        evolved = engineer_evolved_features(df)
        df = pd.concat([df, evolved], axis=1)
        return df


    def get_feature_columns(self, exclude_ground_truth=True):
        """
        Get list of engineered feature columns for modeling.

        Args:
            exclude_ground_truth: If True, exclude features not available in production
                                 (e.g., receiver_is_mule which is ground truth)

        Returns:
            List of feature column names
        """
        features = [
            # Amount features
            'amount_jpy',
            'amount_log',
            'is_high_amount',
            'is_very_high_amount',
            # NOTE: balance_drain_ratio EXCLUDED (corr=0.82 with target — leaky proxy)

            # Account features
            'sender_account_age_days',
            'sender_ib_enabled',
            'sender_ib_days_since_enabled',
            'sender_suspicious_device',
            # NOTE: sender_account_risk_score EXCLUDED (corr=0.70 — trained on ground truth)

            # Velocity features
            'txn_count_24h',
            'txn_volume_24h',
            'has_recent_activity',

            # Deviation features
            # NOTE: sender_txn_count EXCLUDED (corr=0.90 — near-perfect proxy)
            # NOTE: sender_mean_amount, sender_std_amount EXCLUDED (corr>0.70)
            'amount_zscore',
            'amount_above_mean',
            'amount_much_above_mean',

            # Temporal features
            'hour',
            'day_of_week',
            'is_weekend',
            'is_business_hours',
            'is_late_night',

            # Receiver features
            'receiver_account_age_days',
            'receiver_is_new',
            'receiver_is_very_new',

            # Original flags
            'first_transfer',

            # ShinkaEvolve-discovered features
            'evolved_log_amount',
            'evolved_ib_enable_ratio',
            'evolved_avg_recent_txn',
            'evolved_new_acct_high_amount',
            'evolved_susp_device_ib',
            'evolved_weekend_late_night_volume',
            'evolved_volume_susp_device',
        ]

        # Add ground truth features only if requested
        if not exclude_ground_truth:
            features.insert(-1, 'receiver_is_mule')  # Insert before 'first_transfer'

        return features


if __name__ == "__main__":
    import os
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Test feature engineering
    print("Testing transaction feature engineering...")

    # Load data
    transactions_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'transactions.csv'))
    accounts_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'accounts_with_risk.csv'))

    print(f"Loaded {len(transactions_df)} transactions")
    print(f"Loaded {len(accounts_df)} accounts")

    # Engineer features
    engineer = TransactionFeatureEngineer(lookback_hours=24)
    enriched_df = engineer.engineer_features(transactions_df, accounts_df)

    # Save
    enriched_df.to_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'transactions_with_features.csv'), index=False)
    print(f"\n✓ Saved enriched transactions to: data/processed/transactions_with_features.csv")
    print(f"  Total features: {len(enriched_df.columns)}")
    print(f"  Modeling features: {len(engineer.get_feature_columns())}")
