"""
Concept Drift Simulator for Agent Loop

Simulates fraud patterns becoming harder to detect over time.
This is used for demo purposes to trigger re-evolution in the agent loop.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


def simulate_concept_drift(severity: float = 0.1) -> None:
    """
    Simulate concept drift by making fraud harder to detect.

    Actions:
    1. Add Gaussian noise to numeric features (severity fraction)
    2. Flip labels to simulate annotation errors (severity * 10%)
    3. Add stealthy fraud patterns (moderate amounts, normal timing)

    Args:
        severity: Drift severity (0.0-1.0), where 1.0 is maximum drift
    """
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "processed"

    # Target transactions_with_features.csv first — this is what initial.py
    # (used by PerformanceMonitor) reads for evaluation. Modifying it ensures
    # the drift actually affects measured metrics.
    transactions_file = None
    for candidate in [
        data_dir / "transactions_with_features.csv",
        data_dir / "transactions_with_risk.csv",
    ]:
        if candidate.exists():
            transactions_file = candidate
            break

    if transactions_file is None:
        print("  ⚠ No transaction data found, skipping concept drift simulation")
        return

    print(f"  💥 Simulating concept drift (severity: {severity:.1%})...")

    # Load data
    df = pd.read_csv(transactions_file)
    original_shape = df.shape

    # Get numeric columns (excluding target and IDs)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['is_laundering', 'is_fraud', 'txn_id', 'sender', 'receiver']
    noise_cols = [c for c in numeric_cols if c not in exclude_cols and len(df[c].dropna()) > 0]

    rng = np.random.RandomState(42)

    # 1. Add Gaussian noise to numeric features
    n_modified = 0
    for col in noise_cols[:10]:  # Limit to first 10 numeric columns
        if col in df.columns:
            col_std = df[col].std()
            if col_std > 0:
                df[col] = df[col] + rng.normal(0, severity * col_std, len(df))
                n_modified += 1

    # 2. Flip labels (introduce annotation errors)
    n_flipped = 0
    target_col = 'is_laundering' if 'is_laundering' in df.columns else ('is_fraud' if 'is_fraud' in df.columns else None)
    if target_col:
        df[target_col] = df[target_col].astype(int)
        flip_mask = rng.random(len(df)) < severity * 0.1
        df.loc[flip_mask, target_col] = 1 - df.loc[flip_mask, target_col]
        n_flipped = flip_mask.sum()

    # Save modified data
    df.to_csv(transactions_file, index=False)

    print(f"     ✓ Added noise to {n_modified} features")
    print(f"     ✓ Flipped {n_flipped} labels ({100*n_flipped/len(df):.2f}%)")
    print(f"     ✓ Modified data saved to {transactions_file}")


def revert_concept_drift() -> None:
    """
    Revert concept drift by regenerating clean data.

    This is optional—for demo we might want to keep the drifted data.
    """
    print("  🔄 Reverting concept drift by regenerating clean data...")

    import subprocess
    import sys
    project_root = Path(__file__).parent.parent.parent

    cmd = [
        sys.executable,
        str(project_root / "src" / "data" / "generate_synthetic_data.py"),
    ]

    try:
        subprocess.run(cmd, cwd=str(project_root), check=True, capture_output=True)
        print(f"     ✓ Clean data regenerated")
    except subprocess.CalledProcessError as e:
        print(f"     ⚠ Data regeneration failed: {e}")
