"""
Performance Monitor for Agent Loop

Tracks fraud detection metrics and detects performance degradation.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, Tuple


class PerformanceMonitor:
    """Monitor system performance and detect degradation."""

    def __init__(self, log_dir: str = "experiments/agent_loop"):
        """Initialize performance monitor.

        Args:
            log_dir: Directory to store iteration logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv = self.log_dir / "iteration_log.csv"

    def load_current_metrics(self, seed: int = 42) -> Dict:
        """
        Load current system performance by running the ShinkaEvolve evaluate function.

        Returns:
            Dictionary with keys: precision, recall, f1_score, combined_score, etc.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        # Import the evaluate function from initial.py
        import importlib.util
        initial_path = Path(__file__).parent.parent.parent / "experiments/shinkaevolve/initial.py"
        spec = importlib.util.spec_from_file_location("initial", initial_path)
        initial_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(initial_module)

        # Run evaluation
        metrics = initial_module.evaluate(seed=seed)
        return metrics

    def detect_degradation(
        self,
        baseline_metrics: Dict,
        current_metrics: Dict,
        threshold: float = 0.05
    ) -> Tuple[bool, str]:
        """
        Detect if system performance has degraded.

        Checks for drops in precision and recall greater than threshold (5% by default).

        Args:
            baseline_metrics: Baseline metrics dict
            current_metrics: Current metrics dict
            threshold: Degradation threshold (fraction, 0.05 = 5%)

        Returns:
            Tuple of (degraded: bool, reason: str)
        """
        # Handle cases where metrics might be missing
        baseline_precision = baseline_metrics.get('precision', 0.9)
        baseline_recall = baseline_metrics.get('recall', 0.9)
        baseline_f1 = baseline_metrics.get('f1_score', 0.9)

        current_precision = current_metrics.get('precision', 0.9)
        current_recall = current_metrics.get('recall', 0.9)
        current_f1 = current_metrics.get('f1_score', 0.9)

        # Calculate relative drops
        precision_drop = max(0, (baseline_precision - current_precision) / max(baseline_precision, 0.001))
        recall_drop = max(0, (baseline_recall - current_recall) / max(baseline_recall, 0.001))
        f1_drop = max(0, (baseline_f1 - current_f1) / max(baseline_f1, 0.001))

        # Detect degradation
        degraded = precision_drop > threshold or recall_drop > threshold or f1_drop > threshold

        reason = (
            f"Precision: {precision_drop:.1%}, Recall: {recall_drop:.1%}, F1: {f1_drop:.1%}"
        )

        return degraded, reason

    def log_iteration(
        self,
        iteration: int,
        metrics: Dict,
        agent_action: str = "monitor"
    ) -> None:
        """
        Log iteration results to CSV.

        Args:
            iteration: Iteration number
            metrics: Performance metrics dict
            agent_action: What the agent did ("monitor", "evolve", etc.)
        """
        # Extract key metrics
        row = {
            'iteration': iteration,
            'action': agent_action,
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0),
            'combined_score': metrics.get('combined_score', 0),
            'roc_auc': metrics.get('roc_auc', 0),
            'prevention_rate': metrics.get('prevention_rate', 0),
            'alert_rate': metrics.get('alert_rate', 0),
            'alert_quality': metrics.get('alert_quality', 0),
        }

        # Write header if file doesn't exist
        if not self.log_csv.exists():
            with open(self.log_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()

        # Append row
        with open(self.log_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

        print(f"  → Logged iteration {iteration} to {self.log_csv}")

    def get_metrics_summary(self, metrics: Dict) -> str:
        """Format metrics for display."""
        return (
            f"P={metrics.get('precision', 0):.3f}, "
            f"R={metrics.get('recall', 0):.3f}, "
            f"F1={metrics.get('f1_score', 0):.3f}, "
            f"AUC={metrics.get('roc_auc', 0):.3f}"
        )
