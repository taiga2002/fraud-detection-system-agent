"""
Baseline Rule Engine for Online-Banking Scam Detection

Implements interpretable hand-written rules based on Japanese scam patterns.
This serves as the baseline against which evolved ShinkaEvolve policies will be compared.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from enum import Enum


class Action(Enum):
    """Decision actions for suspicious transfers"""
    ALLOW = "allow"
    STEP_UP_VERIFICATION = "step_up_verification"
    HOLD_FOR_REVIEW = "hold_for_review"
    ESCALATE = "escalate"
    FREEZE_TRANSFER = "freeze_transfer"


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertDecision:
    """Decision output from the rule engine"""
    action: Action
    risk_level: RiskLevel
    risk_score: float  # 0.0 to 1.0
    reason_codes: List[str]
    explanation: str


class BaselineRuleEngine:
    """
    Hand-written rule engine for detecting suspicious online-banking transactions.

    Based on Japan-specific scam patterns from FSA/NPA official statistics:
    - New beneficiary + high amount
    - Recently enabled internet banking
    - Rapid fan-out patterns (mule accounts)
    - Device/access environment mismatches
    - Balance drain behavior
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize rule engine with thresholds from japan_calibration.yaml

        Args:
            config: Configuration dictionary with detection thresholds
        """
        self.config = config
        self.thresholds = config.get('detection_thresholds', {})

        # Reason code descriptions from config
        self.reason_descriptions = config.get('reason_codes', {})

        # Extract key thresholds
        txn_thresholds = self.thresholds.get('transaction', {})
        self.high_amount_threshold = txn_thresholds.get('new_beneficiary_high_amount_jpy', 1000000)
        self.balance_drain_threshold = txn_thresholds.get('balance_drain_ratio_threshold', 0.5)
        self.amount_multiplier_threshold = txn_thresholds.get('first_transfer_amount_multiplier', 3.0)

        account_thresholds = self.thresholds.get('account_opening', {})
        self.high_account_risk_threshold = account_thresholds.get('high_risk_score', 0.7)
        self.medium_account_risk_threshold = account_thresholds.get('medium_risk_score', 0.4)


    def evaluate_transaction(self,
                           transaction: Dict[str, Any],
                           account_info: Dict[str, Any],
                           receiver_info: Dict[str, Any]) -> AlertDecision:
        """
        Evaluate a single transaction for suspicious behavior.

        Args:
            transaction: Transaction details (amount, timestamp, etc.)
            account_info: Sender account information
            receiver_info: Receiver account information (if available)

        Returns:
            AlertDecision with action, risk level, scores, and reason codes
        """
        reason_codes = []
        risk_factors = []

        # Extract transaction features
        amount_jpy = transaction.get('amount_jpy', 0)
        first_transfer = transaction.get('first_transfer', False)
        balance_drain_ratio = transaction.get('balance_drain_ratio', 0.0)
        sender_device_id = transaction.get('sender_device_id', None)

        # Extract account features
        account_age_days = account_info.get('account_age_days', 999)
        ib_enabled = account_info.get('ib_enabled', True)
        ib_days_since_enabled = account_info.get('ib_days_since_enabled', 999)
        account_risk_score = account_info.get('account_risk_score', 0.0)
        suspicious_device = account_info.get('suspicious_device', False)
        device_id = account_info.get('device_id', None)

        # ===================================================================
        # RULE 1: New Account with Internet Banking Recently Enabled
        # ===================================================================
        if ib_enabled and ib_days_since_enabled is not None and ib_days_since_enabled < 14:
            reason_codes.append('AR02')  # IB recently enabled (<14 days)
            risk_factors.append(0.3)

            if first_transfer and amount_jpy > self.high_amount_threshold:
                reason_codes.append('TR01')  # First-time transfer to new beneficiary
                reason_codes.append('TR04')  # Amount >¥1M to new beneficiary
                risk_factors.append(0.4)
                risk_factors.append(0.3)

        # ===================================================================
        # RULE 2: High Balance Drain Ratio
        # ===================================================================
        if balance_drain_ratio > self.balance_drain_threshold:
            reason_codes.append('TR03')  # Balance drain ratio >50%
            risk_factors.append(0.35)

            if first_transfer:
                reason_codes.append('TR01')
                risk_factors.append(0.25)

        # ===================================================================
        # RULE 3: Suspicious Device / Access Environment
        # ===================================================================
        if suspicious_device:
            reason_codes.append('AR03')  # Suspicious device/IP at onboarding
            reason_codes.append('NR03')  # Device fingerprint matches frozen account
            risk_factors.append(0.5)

        # ===================================================================
        # RULE 4: High Account Opening Risk Score
        # ===================================================================
        if account_risk_score > self.high_account_risk_threshold:
            reason_codes.append('AR04')  # High account opening risk score (>0.7)
            risk_factors.append(0.4)
        elif account_risk_score > self.medium_account_risk_threshold:
            risk_factors.append(0.2)

        # ===================================================================
        # RULE 5: New Account (< 30 days)
        # ===================================================================
        if account_age_days < 30:
            reason_codes.append('AR01')  # New account (<30 days)
            risk_factors.append(0.25)

            if amount_jpy > self.high_amount_threshold:
                risk_factors.append(0.3)

        # ===================================================================
        # RULE 6: Amount Significantly Exceeds Normal
        # ===================================================================
        # This is a placeholder - would need account transaction history
        # For now, use high amounts as a proxy
        if amount_jpy > self.high_amount_threshold * 2:
            reason_codes.append('TR02')  # Amount significantly exceeds account norm
            risk_factors.append(0.3)

        # ===================================================================
        # RULE 7: Mule-Like Receiver Pattern (Observable Features Only)
        # NOTE: Uses observable characteristics, NOT ground truth is_mule
        # ===================================================================
        if receiver_info:
            receiver_age_days = receiver_info.get('account_age_days', 999)

            # New receiver account with high-value transfer (mule-like)
            if receiver_age_days < 90 and amount_jpy > 500000:
                reason_codes.append('NR01')  # Receiving account shows mule-like pattern
                risk_factors.append(0.3)

            # Very new receiver with any large transfer
            if receiver_age_days < 30 and amount_jpy > 200000:
                reason_codes.append('NR02')  # Very new receiving account
                risk_factors.append(0.25)

        # ===================================================================
        # COMBINED RISK SCORING
        # ===================================================================
        # Aggregate risk factors (capped at 1.0)
        raw_risk_score = sum(risk_factors)
        risk_score = min(raw_risk_score, 1.0)

        # Determine risk level and action
        risk_level, action = self._determine_action(risk_score, reason_codes)

        # Generate human-readable explanation
        explanation = self._generate_explanation(
            reason_codes, risk_score, action, amount_jpy
        )

        return AlertDecision(
            action=action,
            risk_level=risk_level,
            risk_score=risk_score,
            reason_codes=reason_codes,
            explanation=explanation
        )


    def _determine_action(self,
                         risk_score: float,
                         reason_codes: List[str]) -> Tuple[RiskLevel, Action]:
        """
        Determine action based on risk score and specific reason codes.

        Implements escalation policy:
        - Critical (>0.8): Freeze transfer
        - High (0.6-0.8): Escalate to analyst
        - Medium (0.4-0.6): Hold for review
        - Medium (0.3-0.4): Step-up verification
        - Low (<0.3): Allow

        Args:
            risk_score: Aggregated risk score (0.0 to 1.0)
            reason_codes: List of triggered reason codes

        Returns:
            (RiskLevel, Action) tuple
        """
        # Critical risk: Freeze transfer
        if risk_score >= 0.8:
            return (RiskLevel.CRITICAL, Action.FREEZE_TRANSFER)

        # High risk: Escalate to analyst
        if risk_score >= 0.6:
            return (RiskLevel.HIGH, Action.ESCALATE)

        # Medium-high risk: Hold for review
        if risk_score >= 0.4:
            # Exception: If only device mismatch, step-up instead
            if len(reason_codes) == 1 and reason_codes[0] in ['AR03', 'NR03']:
                return (RiskLevel.MEDIUM, Action.STEP_UP_VERIFICATION)
            return (RiskLevel.MEDIUM, Action.HOLD_FOR_REVIEW)

        # Medium risk: Step-up verification
        if risk_score >= 0.3:
            return (RiskLevel.MEDIUM, Action.STEP_UP_VERIFICATION)

        # Low risk: Allow
        return (RiskLevel.LOW, Action.ALLOW)


    def _generate_explanation(self,
                            reason_codes: List[str],
                            risk_score: float,
                            action: Action,
                            amount_jpy: float) -> str:
        """
        Generate human-readable explanation for the decision.

        Args:
            reason_codes: List of triggered reason codes
            risk_score: Final risk score
            action: Recommended action
            amount_jpy: Transaction amount

        Returns:
            Explanation string
        """
        if not reason_codes:
            return f"Transaction appears normal (risk score: {risk_score:.2f}). No suspicious indicators detected."

        # Build explanation
        reasons_text = '; '.join([
            self.reason_descriptions.get(code, code) for code in reason_codes[:5]  # Max 5 reasons
        ])

        explanation = (
            f"Transfer of ¥{amount_jpy:,.0f} flagged with risk score {risk_score:.2f}. "
            f"Suspicious indicators: {reasons_text}. "
            f"Recommended action: {action.value.replace('_', ' ')}."
        )

        return explanation


def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Load configuration from japan_calibration.yaml

    Args:
        yaml_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    import yaml
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Example usage
    import yaml

    # Load config
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'japan_calibration.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize rule engine
    engine = BaselineRuleEngine(config)

    # Example suspicious transaction (victim-opened IB + high amount + new beneficiary)
    suspicious_txn = {
        'txn_id': 'TXN_12345',
        'amount_jpy': 2500000,  # ¥2.5M
        'first_transfer': True,
        'balance_drain_ratio': 0.85,
        'sender_device_id': 'DEV_0123'
    }

    suspicious_account = {
        'account_id': 'ACC_98765',
        'account_age_days': 45,
        'ib_enabled': True,
        'ib_days_since_enabled': 7,  # Recently enabled!
        'account_risk_score': 0.75,  # High risk
        'suspicious_device': False,
        'device_id': 'DEV_0123'
    }

    receiver_account = {
        'account_id': 'ACC_11111',
        'account_age_days': 20,  # New account (mule-like pattern)
    }

    # Evaluate
    decision = engine.evaluate_transaction(
        suspicious_txn,
        suspicious_account,
        receiver_account
    )

    print("=" * 80)
    print("BASELINE RULE ENGINE - EXAMPLE DECISION")
    print("=" * 80)
    print(f"Action: {decision.action.value}")
    print(f"Risk Level: {decision.risk_level.value}")
    print(f"Risk Score: {decision.risk_score:.3f}")
    print(f"Reason Codes: {', '.join(decision.reason_codes)}")
    print(f"\nExplanation:\n{decision.explanation}")
    print("=" * 80)

    # Example normal transaction
    normal_txn = {
        'txn_id': 'TXN_67890',
        'amount_jpy': 35000,  # ¥35K
        'first_transfer': False,
        'balance_drain_ratio': 0.10,
        'sender_device_id': 'DEV_5555'
    }

    normal_account = {
        'account_id': 'ACC_54321',
        'account_age_days': 850,  # Old account
        'ib_enabled': True,
        'ib_days_since_enabled': 600,
        'account_risk_score': 0.15,  # Low risk
        'suspicious_device': False,
        'device_id': 'DEV_5555'
    }

    decision2 = engine.evaluate_transaction(
        normal_txn,
        normal_account,
        None  # No receiver info
    )

    print("\nNORMAL TRANSACTION:")
    print(f"Action: {decision2.action.value}")
    print(f"Risk Score: {decision2.risk_score:.3f}")
    print(f"Explanation: {decision2.explanation}")
