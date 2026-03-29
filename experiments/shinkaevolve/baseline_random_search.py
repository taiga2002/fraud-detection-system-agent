"""
Random Search Baseline for ShinkaEvolve Comparison

This script runs uninformed random search on the same hardened fraud detection
task to establish a baseline. Comparing random search vs ShinkaEvolve proves
that LLM-guided mutations are genuinely better than random exploration.

Approach:
- Use the same hardened data and fitness function as ShinkaEvolve
- Generate random feature combinations from the available columns
- Evaluate 20 random configurations
- Compare best result to ShinkaEvolve's best (0.640)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Import the evaluation logic from initial.py
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'experiments' / 'shinkaevolve'))

# Import the base evaluation function
import importlib.util
spec = importlib.util.spec_from_file_location('initial', PROJECT_ROOT / 'experiments' / 'shinkaevolve' / 'initial.py')
initial_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(initial_module)

evaluate_fn = initial_module.evaluate
get_model_params = initial_module.get_model_params
compute_fusion_score = initial_module.compute_fusion_score
load_data = initial_module.load_data
BASE_FEATURE_COLS = initial_module.BASE_FEATURE_COLS


def generate_random_features(df, seed):
    """Generate random feature combinations (uninformed search)."""
    rng = np.random.RandomState(seed)
    feats = pd.DataFrame(index=df.index)

    # Available columns for random combinations
    numeric_cols = ['amount_jpy', 'sender_account_age_days', 'sender_ib_days_since_enabled',
                    'receiver_account_age_days', 'txn_count_24h', 'txn_volume_24h', 'hour']
    binary_cols = ['is_high_amount', 'is_very_high_amount', 'first_transfer',
                   'sender_ib_enabled', 'sender_suspicious_device', 'receiver_is_new',
                   'receiver_is_very_new', 'has_recent_activity', 'is_weekend',
                   'is_business_hours', 'is_late_night']

    # Generate 5-7 random features
    n_features = rng.randint(5, 8)

    for i in range(n_features):
        op = rng.choice(['ratio', 'product', 'sum', 'interaction'])

        if op == 'ratio':
            c1, c2 = rng.choice(numeric_cols, 2, replace=False)
            feats[f'random_ratio_{i}'] = df[c1].fillna(0) / (df[c2].fillna(0) + 1)

        elif op == 'product':
            c1 = rng.choice(numeric_cols)
            c2 = rng.choice(binary_cols)
            feats[f'random_prod_{i}'] = df[c1].fillna(0) * df[c2].fillna(0)

        elif op == 'sum':
            c1, c2 = rng.choice(numeric_cols, 2, replace=False)
            feats[f'random_sum_{i}'] = df[c1].fillna(0) + df[c2].fillna(0)

        elif op == 'interaction':
            b1, b2 = rng.choice(binary_cols, 2, replace=False)
            feats[f'random_inter_{i}'] = df[b1].fillna(0) * df[b2].fillna(0)

    return feats


def evaluate_random_config(seed):
    """Evaluate one random configuration."""
    # Monkey-patch the feature engineering function
    initial_module.engineer_evolved_features = lambda df: generate_random_features(df, seed)

    # Run evaluation
    result = evaluate_fn(seed=seed)
    return result


def main():
    print("=" * 80)
    print("RANDOM SEARCH BASELINE (20 iterations)")
    print("=" * 80)
    print("\nRunning uninformed random search on hardened fraud detection task...")
    print("This establishes a baseline to prove LLM-guided >> random mutations.\n")

    results = []
    best_score = 0.0
    best_seed = None

    for i in range(1, 21):
        print(f"Iteration {i}/20...", end=' ')
        seed = 1000 + i
        try:
            result = evaluate_random_config(seed)
            score = result['combined_score']
            results.append(result)

            if score > best_score:
                best_score = score
                best_seed = seed
                print(f"✓ score={score:.4f} ⭐ NEW BEST")
            else:
                print(f"✓ score={score:.4f}")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results.append({'combined_score': 0.0, 'error': str(e)})

    # Summary
    print("\n" + "=" * 80)
    print("RANDOM SEARCH SUMMARY")
    print("=" * 80)

    valid_scores = [r['combined_score'] for r in results if r['combined_score'] > 0]

    print(f"Valid results: {len(valid_scores)}/20")
    print(f"Best score: {best_score:.4f} (seed {best_seed})")
    print(f"Mean score: {np.mean(valid_scores):.4f}")
    print(f"Std score: {np.std(valid_scores):.4f}")

    # Compare to ShinkaEvolve
    shinkaevolve_hardened_best = 0.640
    shinkaevolve_baseline = 0.528

    print("\n" + "=" * 80)
    print("COMPARISON: RANDOM SEARCH vs SHINKAEVOLVE")
    print("=" * 80)
    print(f"{'Approach':<30} {'Baseline':<12} {'Best':<12} {'Improvement':<15}")
    print("-" * 80)
    print(f"{'Random Search (uninformed)':<30} {shinkaevolve_baseline:<12.3f} {best_score:<12.3f} {(best_score - shinkaevolve_baseline) / shinkaevolve_baseline * 100:>12.1f}%")
    print(f"{'ShinkaEvolve (LLM-guided)':<30} {shinkaevolve_baseline:<12.3f} {shinkaevolve_hardened_best:<12.3f} {(shinkaevolve_hardened_best - shinkaevolve_baseline) / shinkaevolve_baseline * 100:>12.1f}%")
    print(f"\n{'Advantage:':<30} ShinkaEvolve is {(shinkaevolve_hardened_best - shinkaevolve_baseline) / max(best_score - shinkaevolve_baseline, 0.001):.1f}x better")

    # Save results
    output_file = 'random_search_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'best_score': best_score,
            'best_seed': best_seed,
            'mean_score': float(np.mean(valid_scores)),
            'std_score': float(np.std(valid_scores)),
            'valid_count': len(valid_scores),
            'shinkaevolve_comparison': {
                'shinkaevolve_best': shinkaevolve_hardened_best,
                'random_search_best': best_score,
                'advantage': (shinkaevolve_hardened_best - shinkaevolve_baseline) / max(best_score - shinkaevolve_baseline, 0.001),
            }
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
