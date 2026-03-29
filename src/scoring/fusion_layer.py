"""
Fusion Decision Layer

Combines three risk scores into unified suspicious-transfer and mule-account scores:
1. Account opening risk (from account_risk_model)
2. Transfer risk (from transfer_risk_model)
3. Network/graph risk (from mule_network_detector)

Outputs comprehensive reason codes and action recommendations.
"""

import pandas as pd
import numpy as np
import pickle
import os
from dataclasses import dataclass
from typing import List
from enum import Enum


class Action(Enum):
    """Decision actions"""
    ALLOW = "allow"
    STEP_UP_VERIFICATION = "step_up_verification"
    HOLD_FOR_REVIEW = "hold_for_review"
    ESCALATE = "escalate"
    FREEZE_TRANSFER = "freeze_transfer"


class RiskLevel(Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FusionDecision:
    """Fusion layer decision output"""
    txn_id: str
    action: Action
    risk_level: RiskLevel

    # Individual scores
    account_risk_score: float
    transfer_risk_score: float
    network_risk_score: float

    # Fused scores
    suspicious_transfer_score: float
    mule_account_score: float

    # Reason codes from all layers
    reason_codes: List[str]
    explanation: str


class FusionLayer:
    """
    Fusion decision layer combining all three risk models.

    Uses weighted fusion strategy to combine:
    - Account opening risk
    - Transfer risk
    - Network/graph risk
    """

    def __init__(self, config=None):
        """
        Initialize fusion layer.

        Args:
            config: Optional configuration dict with fusion weights.
                    If None, attempts to load optimized weights from
                    experiments/shinka/evolved_config.pkl (produced by Optuna).
                    Falls back to hand-tuned defaults if no optimized config exists.
        """
        if config is None:
            config = self._load_config()

        self.config = config
        self.reason_descriptions = config.get('reason_codes', {})
        self.account_weight = config['account_weight']
        self.transfer_weight = config['transfer_weight']
        self.network_weight = config['network_weight']
        self.thresholds = config['thresholds']

    @staticmethod
    def _load_config():
        """Load optimized config from Optuna output, or fall back to defaults."""
        default_config = {
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

        # Try to load optimized weights from experiments/shinka/evolved_config.pkl
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        config_path = os.path.join(project_root, 'experiments', 'shinka', 'evolved_config.pkl')

        if os.path.exists(config_path):
            try:
                with open(config_path, 'rb') as f:
                    data = pickle.load(f)
                best = data['best_individual']
                return {
                    'account_weight': best['account_weight'],
                    'transfer_weight': best['transfer_weight'],
                    'network_weight': best['network_weight'],
                    'thresholds': default_config['thresholds']
                }
            except Exception:
                pass

        return default_config

    def fuse_scores(self, account_score, transfer_score, network_score):
        """
        Fuse three scores into unified suspicious-transfer score.

        Uses weighted average with non-linear boost discovered by ShinkaEvolve:
        when account risk is elevated (>0.3), transfer risk is amplified via
        power transform (^1.2) to catch high-risk combinations more aggressively.

        Args:
            account_score: Account opening risk (0-1)
            transfer_score: Transfer risk (0-1)
            network_score: Network/graph risk (0-1)

        Returns:
            Fused score (0-1)
        """
        base_fused = (
            self.account_weight * account_score +
            self.transfer_weight * transfer_score +
            self.network_weight * network_score
        )

        # ShinkaEvolve-discovered: non-linear boost when account risk is elevated
        # Why it works: Japanese fraud pattern is newly-enabled IB (high account_risk)
        # + large suspicious transfer (high transfer_risk). The power transform (^1.2)
        # amplifies this dangerous combination more than either signal alone.
        # Example: account=0.4, transfer=0.85 → boost increases fused score 0.76→0.79
        # See docs/shinkaevolve.md for full explanation.
        if account_score > 0.3:
            boosted_transfer = min(transfer_score ** 1.2, 1.0)
            fused = (
                self.account_weight * account_score +
                self.transfer_weight * boosted_transfer +
                self.network_weight * network_score
            )
            return min(fused, 1.0)

        return base_fused


    def determine_action(self, fused_score, reason_codes):
        """
        Determine action based on fused score and reason codes.

        Args:
            fused_score: Fused suspicious-transfer score
            reason_codes: List of triggered reason codes

        Returns:
            (RiskLevel, Action) tuple
        """
        # Critical risk
        if fused_score >= self.thresholds['critical']:
            return (RiskLevel.CRITICAL, Action.FREEZE_TRANSFER)

        # High risk
        if fused_score >= self.thresholds['high']:
            # Check if we should escalate or just hold
            has_mule_network = any(code.startswith('NR') for code in reason_codes)
            if has_mule_network:
                return (RiskLevel.HIGH, Action.ESCALATE)
            else:
                return (RiskLevel.HIGH, Action.HOLD_FOR_REVIEW)

        # Medium risk
        if fused_score >= self.thresholds['medium']:
            # Step-up verification for medium risk
            return (RiskLevel.MEDIUM, Action.STEP_UP_VERIFICATION)

        # Low risk
        return (RiskLevel.LOW, Action.ALLOW)


    def generate_reason_codes(self, account_score, transfer_score, network_score,
                             transaction_features, account_features):
        """
        Generate comprehensive reason codes from all three layers.

        Args:
            account_score: Account risk score
            transfer_score: Transfer risk score
            network_score: Network risk score
            transaction_features: Dict of transaction features
            account_features: Dict of account features

        Returns:
            List of reason codes
        """
        codes = []

        # Account layer codes
        if account_score > 0.7:
            codes.append('AR04')  # High account opening risk score

        if account_features.get('ib_days_since_enabled', 999) < 14:
            codes.append('AR02')  # IB recently enabled

        if account_features.get('account_age_days', 999) < 30:
            codes.append('AR01')  # New account

        if account_features.get('suspicious_device', False):
            codes.append('AR03')  # Suspicious device

        # Transfer layer codes
        if transaction_features.get('first_transfer', False):
            codes.append('TR01')  # First-time transfer

        if transaction_features.get('balance_drain_ratio', 0) > 0.5:
            codes.append('TR03')  # Balance drain >50%

        if transaction_features.get('amount_jpy', 0) > 1000000:
            codes.append('TR04')  # Amount >¥1M

        if transaction_features.get('amount_much_above_mean', False):
            codes.append('TR02')  # Amount exceeds norm

        if transaction_features.get('txn_count_24h', 0) > 5:
            codes.append('TR07')  # Rapid sequence of transfers

        # Network layer codes
        if network_score > 0.7:
            codes.append('NR02')  # In known mule cluster

        if account_features.get('is_mule_like', False):
            codes.append('NR01')  # Mule-like fan-out pattern

        if account_features.get('fan_out_ratio', 0) > 3:
            codes.append('NR05')  # High out-degree ratio

        if account_features.get('has_shared_device', False):
            codes.append('NR03')  # Shared device

        # Combined pattern codes
        if (account_score > 0.6 and transfer_score > 0.6 and network_score > 0.6):
            codes.append('CM01')  # High risk across all layers

        if (account_features.get('ib_days_since_enabled', 999) < 14 and
            transaction_features.get('first_transfer', False) and
            transaction_features.get('amount_jpy', 0) > 1000000):
            codes.append('CM02')  # Japan scam pattern (victim-opened IB + rapid transfer)

        return list(set(codes))  # Remove duplicates


    def generate_explanation(self, reason_codes, fused_score, amount_jpy, action):
        """Generate human-readable explanation"""
        if not reason_codes:
            return f"Transaction of ¥{amount_jpy:,.0f} appears normal (fused risk score: {fused_score:.2f}). No suspicious indicators."

        reasons_text = '; '.join([self.reason_descriptions.get(code, code) for code in reason_codes[:6]])

        explanation = (
            f"Transfer of ¥{amount_jpy:,.0f} flagged with fused risk score {fused_score:.2f}. "
            f"Suspicious indicators: {reasons_text}. "
            f"Recommended action: {action.value.replace('_', ' ')}."
        )

        return explanation


    def evaluate_transaction(self, transaction_row, account_row, receiver_row, graph_features):
        """
        Evaluate single transaction using fusion of all three models.

        Args:
            transaction_row: Series with transaction data and transfer_risk_score
            account_row: Series with account data and account_risk_score + network_risk_score
            receiver_row: Series with receiver account data
            graph_features: Dict with graph features for sender

        Returns:
            FusionDecision object
        """
        # Extract scores
        account_score = account_row.get('account_risk_score', 0.0)
        transfer_score = transaction_row.get('transfer_risk_score', 0.0)
        network_score = account_row.get('network_risk_score', 0.0)

        # Fuse scores
        suspicious_transfer_score = self.fuse_scores(account_score, transfer_score, network_score)

        # Mule account score (based primarily on network features)
        mule_account_score = (
            0.2 * account_score +
            0.1 * transfer_score +
            0.7 * network_score
        )

        # Prepare features for reason code generation
        transaction_features = {
            'amount_jpy': transaction_row.get('amount_jpy', 0),
            'first_transfer': transaction_row.get('first_transfer', False),
            'balance_drain_ratio': transaction_row.get('balance_drain_ratio', 0),
            'amount_much_above_mean': transaction_row.get('amount_much_above_mean', False),
            'txn_count_24h': transaction_row.get('txn_count_24h', 0)
        }

        account_features = {
            'account_age_days': account_row.get('account_age_days', 999),
            'ib_days_since_enabled': account_row.get('ib_days_since_enabled', 999),
            'suspicious_device': account_row.get('suspicious_device', False),
            'is_mule_like': graph_features.get('is_mule_like', False) if graph_features else False,
            'fan_out_ratio': graph_features.get('fan_out_ratio', 0) if graph_features else 0,
            'has_shared_device': graph_features.get('has_shared_device', False) if graph_features else False
        }

        # Generate reason codes
        reason_codes = self.generate_reason_codes(
            account_score, transfer_score, network_score,
            transaction_features, account_features
        )

        # Determine action
        risk_level, action = self.determine_action(suspicious_transfer_score, reason_codes)

        # Generate explanation
        explanation = self.generate_explanation(
            reason_codes, suspicious_transfer_score,
            transaction_features['amount_jpy'], action
        )

        return FusionDecision(
            txn_id=transaction_row.get('txn_id', 'unknown'),
            action=action,
            risk_level=risk_level,
            account_risk_score=account_score,
            transfer_risk_score=transfer_score,
            network_risk_score=network_score,
            suspicious_transfer_score=suspicious_transfer_score,
            mule_account_score=mule_account_score,
            reason_codes=reason_codes,
            explanation=explanation
        )


def main():
    """Test fusion layer"""
    import os
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    print("=" * 80)
    print("FUSION LAYER - DEMO")
    print("=" * 80)

    # Load all enriched data
    print("\nLoading enriched data...")
    transactions_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'transactions_with_risk.csv'))
    accounts_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'accounts_with_network_risk.csv'))

    print(f"Loaded {len(transactions_df)} transactions")
    print(f"Loaded {len(accounts_df)} accounts")

    # Initialize fusion layer
    fusion = FusionLayer()

    # Example: Evaluate a high-risk transaction
    print("\n" + "=" * 80)
    print("EXAMPLE HIGH-RISK TRANSACTION")
    print("=" * 80)

    # Find a laundering transaction
    high_risk_txn = transactions_df[transactions_df['is_laundering'] == True].iloc[0]
    sender_account = accounts_df[accounts_df['account_id'] == high_risk_txn['sender']].iloc[0]
    receiver_account = accounts_df[accounts_df['account_id'] == high_risk_txn['receiver']].iloc[0]

    # Evaluate
    decision = fusion.evaluate_transaction(
        high_risk_txn,
        sender_account,
        receiver_account,
        graph_features=None  # Would load from detector in full pipeline
    )

    print(f"\nTransaction ID: {decision.txn_id}")
    print(f"Amount: ¥{high_risk_txn['amount_jpy']:,.0f}")
    print(f"\nRisk Scores:")
    print(f"  Account Risk:    {decision.account_risk_score:.3f}")
    print(f"  Transfer Risk:   {decision.transfer_risk_score:.3f}")
    print(f"  Network Risk:    {decision.network_risk_score:.3f}")
    print(f"  ► Fused Score:   {decision.suspicious_transfer_score:.3f}")
    print(f"  ► Mule Score:    {decision.mule_account_score:.3f}")
    print(f"\nDecision:")
    print(f"  Risk Level:      {decision.risk_level.value.upper()}")
    print(f"  Action:          {decision.action.value.replace('_', ' ').title()}")
    print(f"  Reason Codes:    {', '.join(decision.reason_codes)}")
    print(f"\nExplanation:")
    print(f"  {decision.explanation}")

    print("\n" + "=" * 80)
    print("✓ Fusion layer demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
