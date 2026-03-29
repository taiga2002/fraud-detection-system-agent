"""
Evolution Trigger for Agent Loop

Launches ShinkaEvolve to discover new evolved features.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def trigger_evolution(iteration: int, num_generations: int = 10) -> str:
    """
    Launch ShinkaEvolve to evolve fraud detection features.

    Args:
        iteration: Agent loop iteration number
        num_generations: Number of generations to evolve

    Returns:
        Path to results directory

    Raises:
        RuntimeError: If evolution process fails
    """
    project_root = Path(__file__).parent.parent.parent
    shinkaevolve_dir = project_root / "experiments" / "shinkaevolve"
    results_dir = shinkaevolve_dir / f"results_iteration_{iteration}"

    print(f"\n  🔄 Triggering ShinkaEvolve evolution (iteration {iteration})...")
    print(f"     Generations: {num_generations}")
    print(f"     Results dir: {results_dir}")

    cmd = [
        sys.executable,
        str(shinkaevolve_dir / "run_experiment.py"),
        "--num-generations", str(num_generations),
        "--results-dir", str(results_dir),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=False,
            check=True,
            text=True
        )
        print(f"  ✓ ShinkaEvolve completed")
        return str(results_dir)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ShinkaEvolve evolution failed: {e}")


def get_best_evolved_program(results_dir: str) -> Path:
    """
    Find the best evolved program from ShinkaEvolve results.

    Looks for best/main.py in the results directory.

    Args:
        results_dir: Path to ShinkaEvolve results directory

    Returns:
        Path to best program file

    Raises:
        FileNotFoundError: If best program not found
    """
    results_path = Path(results_dir)
    best_program = results_path / "best" / "main.py"

    if not best_program.exists():
        raise FileNotFoundError(f"Best program not found at {best_program}")

    print(f"  ✓ Found best evolved program: {best_program}")
    return best_program


def get_evolution_metrics(results_dir: str) -> Optional[Dict]:
    """
    Extract evolution metrics from ShinkaEvolve results.

    Looks for metrics.json in the results directory.

    Args:
        results_dir: Path to ShinkaEvolve results directory

    Returns:
        Dictionary with evolution metrics, or None if not found
    """
    results_path = Path(results_dir)
    metrics_file = results_path / "metrics.json"

    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)

    return None
