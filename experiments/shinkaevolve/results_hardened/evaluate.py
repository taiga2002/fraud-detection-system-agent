"""
ShinkaEvolve Evaluation Script for Fraud Detection Pipeline

Uses shinka's run_shinka_eval wrapper to evaluate evolved programs.
Each program is expected to have an `evaluate(seed=...)` function
that returns a dict with 'combined_score' and other metrics.
"""

import argparse
import numpy as np
from shinka.core.wrap_eval import run_shinka_eval


def validate_result(result):
    """Validate a single evaluation run result."""
    if not isinstance(result, dict):
        return False, f"Expected dict, got {type(result).__name__}"

    if 'combined_score' not in result:
        return False, "Missing 'combined_score' in result"

    score = result['combined_score']
    if not isinstance(score, (int, float)):
        return False, f"combined_score is not numeric: {type(score).__name__}"

    if score < 0.0 or score > 1.0:
        return False, f"combined_score out of range [0, 1]: {score}"

    if 'error' in result and score == 0.0:
        return False, f"Program errored: {result['error']}"

    return True, None


def aggregate_metrics(results):
    """Aggregate metrics across multiple evaluation runs."""
    scores = [r['combined_score'] for r in results]
    auc_scores = [r.get('roc_auc', 0.0) for r in results]
    f1_scores = [r.get('f1_score', 0.0) for r in results]
    prevention_rates = [r.get('prevention_rate', 0.0) for r in results]
    alert_qualities = [r.get('alert_quality', 0.0) for r in results]

    return {
        'combined_score': float(np.mean(scores)),
        'combined_score_std': float(np.std(scores)),
        'roc_auc': float(np.mean(auc_scores)),
        'roc_auc_std': float(np.std(auc_scores)),
        'f1_score': float(np.mean(f1_scores)),
        'prevention_rate': float(np.mean(prevention_rates)),
        'alert_quality': float(np.mean(alert_qualities)),
        'num_runs': len(results),
    }


def get_experiment_kwargs(run_index):
    """Provide different seeds for each run."""
    return {'seed': run_index + 1}


def main(program_path: str, results_dir: str):
    """Entry point called by ShinkaEvolve."""
    metrics, correct, error = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name='evaluate',
        num_runs=3,
        get_experiment_kwargs=get_experiment_kwargs,
        validate_fn=validate_result,
        aggregate_metrics_fn=aggregate_metrics,
        run_workers=1,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--program_path', required=True)
    parser.add_argument('--results_dir', required=True)
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
