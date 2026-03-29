"""
Generate synthetic transaction data calibrated to Japanese online-banking scam patterns.

Based on FSA/NPA statistics:
- Average loss: ¥2.26M per case
- 70% use existing accounts with internet banking enabled
- Patterns: victim-opened accounts, rapid fan-out, device reuse

Includes realistic ambiguity:
- Overlapping amount distributions between fraud and normal
- Normal accounts with suspicious-looking behavior
- Account age overlap between mule and normal
- Stealthy fraud transactions that mimic normal patterns
- Normal accounts transacting with mule-like accounts
- Device fingerprint noise (shared devices among normal users)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_accounts(n_accounts=10000):
    """Generate synthetic account data"""
    accounts = []

    # Pre-generate a pool of shared device IDs for normal device-sharing (~2%)
    shared_device_pool = [f"DEV_{random.randint(0, 5000):05d}" for _ in range(50)]

    for i in range(n_accounts):
        account_id = f"ACC_{i:06d}"

        # Account types
        is_mule = random.random() < 0.02  # 2% mule accounts
        is_victim = random.random() < 0.01  # 1% victim accounts

        # --- Change 3: Account age overlap ---
        if is_mule:
            if random.random() < 0.25:
                # 25% of mules are compromised dormant accounts (older)
                account_age_days = int(np.random.exponential(180))
            else:
                account_age_days = int(np.random.exponential(30))  # New mules
        else:
            if random.random() < 0.10:
                # 10% of normal accounts are new (young customers, new to bank)
                account_age_days = int(np.random.exponential(60))
            else:
                account_age_days = int(np.random.exponential(365 * 3))  # Average 3 years

        # Internet banking status
        if is_mule or is_victim:
            ib_enabled = True
            ib_days_since_enabled = min(int(np.random.exponential(10)), account_age_days)
        else:
            ib_enabled = random.random() < 0.7  # 70% have IB enabled
            ib_days_since_enabled = int(account_age_days * random.random()) if ib_enabled else None

        # Customer type
        customer_type = 'individual' if random.random() < 0.85 else 'business'

        # Onboarding channel
        channel = random.choice(['online', 'branch', 'mobile_app', 'phone'])

        # Device fingerprint (simplified)
        device_id = f"DEV_{random.randint(0, 5000):05d}"

        # --- Change 6: Device fingerprint noise ---
        # ~2% of normal accounts share device fingerprints (family, shared computers)
        if not is_mule and not is_victim and random.random() < 0.02:
            device_id = random.choice(shared_device_pool)

        # Suspicious device (reused from known fraud)
        # Mules: 30% have suspicious devices (unchanged)
        # Normal: ~2% flagged as suspicious (shared devices at work/home)
        if is_mule:
            suspicious_device = random.random() < 0.3
        else:
            suspicious_device = random.random() < 0.02

        accounts.append({
            'account_id': account_id,
            'customer_type': customer_type,
            'account_age_days': account_age_days,
            'ib_enabled': ib_enabled,
            'ib_days_since_enabled': ib_days_since_enabled,
            'onboarding_channel': channel,
            'device_id': device_id,
            'suspicious_device': suspicious_device,
            'is_mule': is_mule,
            'is_victim': is_victim
        })

    return pd.DataFrame(accounts)


def generate_transactions(accounts_df, n_transactions=50000):
    """Generate synthetic transaction data with scam patterns and realistic ambiguity"""
    transactions = []
    base_date = datetime(2025, 1, 1)

    # Get account lists
    mule_accounts = accounts_df[accounts_df['is_mule']]['account_id'].tolist()
    victim_accounts = accounts_df[accounts_df['is_victim']]['account_id'].tolist()
    normal_accounts = accounts_df[~(accounts_df['is_mule'] | accounts_df['is_victim'])]['account_id'].tolist()

    # --- Change 5: Identify young/mule-like normal accounts for false-positive pressure ---
    young_normal = accounts_df[
        ~(accounts_df['is_mule'] | accounts_df['is_victim']) &
        (accounts_df['account_age_days'] < 90)
    ]['account_id'].tolist()

    for i in range(n_transactions):
        txn_id = f"TXN_{i:08d}"

        # Decide if this is a laundering transaction
        is_laundering = random.random() < 0.05  # 5% laundering rate

        if is_laundering and victim_accounts and mule_accounts:
            # --- Change 4: Stealthy vs obvious fraud ---
            is_stealthy = random.random() < 0.15  # 15% of fraud is stealthy

            if is_stealthy:
                # Stealthy fraud: mimics normal patterns
                sender = random.choice(victim_accounts)
                receiver = random.choice(mule_accounts)

                # Moderate amounts that look normal (¥100K-500K)
                amount = int(np.random.uniform(100_000, 500_000))

                # Normal-looking timing
                days_offset = int(np.random.exponential(20))

                # Lower balance drain (30-60%)
                balance_drain_ratio = random.uniform(0.3, 0.6)

                # Lower first-transfer rate to blend in
                first_transfer = random.random() < 0.15

            else:
                # Obvious scam pattern: victim -> mule account
                sender = random.choice(victim_accounts)
                receiver = random.choice(mule_accounts)

                # --- Change 1: Wider amount distribution ---
                # Higher amounts for scams, but with wider spread
                if random.random() < 0.1:
                    # 10% small test transfers (¥50K-200K)
                    amount = int(np.random.uniform(50_000, 200_000))
                else:
                    amount = int(np.random.lognormal(14.6, 1.2))  # Wider std (was 0.8)

                days_offset = int(np.random.exponential(5))
                balance_drain_ratio = random.uniform(0.5, 1.0)
                first_transfer = random.random() < 0.3

        elif is_laundering and mule_accounts:
            # Mule fan-out pattern
            sender = random.choice(mule_accounts)
            receiver = random.choice(normal_accounts + mule_accounts)

            # Medium amounts with wider spread
            amount = int(np.random.lognormal(13, 1.3))  # Wider std (was 1.0)
            days_offset = int(np.random.exponential(3))
            balance_drain_ratio = random.uniform(0.5, 1.0)
            first_transfer = random.random() < 0.3

        else:
            # Normal transaction
            sender = random.choice(normal_accounts)

            # --- Change 5: 3% of normal txns go to young/mule-like accounts ---
            if young_normal and random.random() < 0.03:
                receiver = random.choice(young_normal)
            else:
                receiver = random.choice(normal_accounts)

            # --- Change 1: Normal amounts with occasional large transactions ---
            if random.random() < 0.03:
                # 3% large legitimate transactions (down payments, cars, tuition)
                amount = int(np.random.uniform(500_000, 2_000_000))
            else:
                amount = int(np.random.lognormal(11, 1.8))  # Wider std (was 1.5)

            days_offset = int(np.random.exponential(30))

            # --- Change 2: Normal accounts with suspicious-looking behavior ---
            if random.random() < 0.05:
                # 5% of normal txns have high balance drain (large purchases)
                balance_drain_ratio = random.uniform(0.3, 0.7)
            else:
                balance_drain_ratio = random.random() * 0.3

            # 15% first transfers (people do send money to new people)
            first_transfer = random.random() < 0.15

        timestamp = base_date + timedelta(days=days_offset,
                                         hours=random.randint(0, 23),
                                         minutes=random.randint(0, 59))

        # Transaction features
        sender_info = accounts_df[accounts_df['account_id'] == sender].iloc[0]
        receiver_info = accounts_df[accounts_df['account_id'] == receiver].iloc[0]

        transactions.append({
            'txn_id': txn_id,
            'timestamp': timestamp,
            'sender': sender,
            'receiver': receiver,
            'amount_jpy': amount,
            'sender_device_id': sender_info['device_id'],
            'receiver_device_id': receiver_info['device_id'],
            'first_transfer': first_transfer,
            'balance_drain_ratio': balance_drain_ratio,
            'is_laundering': is_laundering
        })

    return pd.DataFrame(transactions)


def main():
    """Generate and save synthetic datasets"""
    print("Generating synthetic accounts...")
    accounts_df = generate_accounts(n_accounts=10000)

    print(f"Generated {len(accounts_df)} accounts")
    print(f"  - Mule accounts: {accounts_df['is_mule'].sum()}")
    print(f"  - Victim accounts: {accounts_df['is_victim'].sum()}")
    print(f"  - Normal accounts: {(~(accounts_df['is_mule'] | accounts_df['is_victim'])).sum()}")

    print("\nGenerating synthetic transactions...")
    transactions_df = generate_transactions(accounts_df, n_transactions=50000)

    print(f"Generated {len(transactions_df)} transactions")
    print(f"  - Laundering transactions: {transactions_df['is_laundering'].sum()}")
    print(f"  - Normal transactions: {(~transactions_df['is_laundering']).sum()}")

    # Save to processed directory
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    os.makedirs(data_dir, exist_ok=True)
    accounts_df.to_csv(os.path.join(data_dir, 'accounts.csv'), index=False)
    transactions_df.to_csv(os.path.join(data_dir, 'transactions.csv'), index=False)

    print("\n✓ Synthetic data saved to data/processed/")
    print("  - accounts.csv")
    print("  - transactions.csv")

    # Print summary statistics
    print("\n=== Transaction Amount Statistics (JPY) ===")
    print(transactions_df.groupby('is_laundering')['amount_jpy'].describe())

    # Print overlap diagnostics
    print("\n=== Ambiguity Diagnostics ===")
    fraud = transactions_df[transactions_df['is_laundering']]
    normal = transactions_df[~transactions_df['is_laundering']]
    print(f"  Fraud amount range: ¥{fraud['amount_jpy'].min():,.0f} - ¥{fraud['amount_jpy'].max():,.0f}")
    print(f"  Normal amount range: ¥{normal['amount_jpy'].min():,.0f} - ¥{normal['amount_jpy'].max():,.0f}")
    print(f"  Normal first_transfer rate: {normal['first_transfer'].mean():.1%}")
    print(f"  Normal high balance drain (>0.3): {(normal['balance_drain_ratio'] > 0.3).mean():.1%}")
    print(f"  Fraud low amount (<500K): {(fraud['amount_jpy'] < 500_000).mean():.1%}")
    print(f"  Normal suspicious_device rate: {accounts_df[~(accounts_df['is_mule'] | accounts_df['is_victim'])]['suspicious_device'].mean():.1%}")


if __name__ == "__main__":
    main()
