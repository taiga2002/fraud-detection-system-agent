#!/usr/bin/env python3
"""
Wrapper to launch ShinkaEvolve experiment for fraud detection.

Usage:
    cd experiments/shinkaevolve
    python run_experiment.py [--num-generations 20] [--results-dir results]
"""

import argparse
import os
import sys
from pathlib import Path

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
from shinka.cli.run_config import load_optional_yaml_config


def main():
    parser = argparse.ArgumentParser(description="Run ShinkaEvolve fraud detection experiment")
    parser.add_argument("--num-generations", type=int, default=20)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    task_dir = Path(__file__).parent.resolve()
    results_dir = (task_dir / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    initial_path = task_dir / "initial.py"
    evaluate_path = task_dir / "evaluate.py"
    system_prompt_path = task_dir / "system_prompt.txt"

    # Load system prompt
    task_sys_msg = system_prompt_path.read_text(encoding="utf-8")

    # Load YAML config
    from dataclasses import fields as dc_fields
    allowed_types = {
        "evo": {f.name: f.type for f in dc_fields(EvolutionConfig)},
        "db": {f.name: f.type for f in dc_fields(DatabaseConfig)},
        "job": {f.name: f.type for f in dc_fields(LocalJobConfig)},
    }
    file_overrides, runner_config = load_optional_yaml_config(
        task_dir=task_dir,
        config_fname=args.config,
        allowed_field_types=allowed_types,
    )

    # Build EvolutionConfig
    evo_kwargs = {
        "num_generations": args.num_generations,
        "job_type": "local",
        "language": "python",
        "init_program_path": str(initial_path),
        "results_dir": str(results_dir),
        "task_sys_msg": task_sys_msg,
    }
    evo_kwargs.update(file_overrides["evo"])
    # Ensure our authoritative values win
    evo_kwargs["num_generations"] = args.num_generations
    evo_kwargs["results_dir"] = str(results_dir)
    evo_kwargs["task_sys_msg"] = task_sys_msg

    evo_config = EvolutionConfig(**evo_kwargs)

    # Build DatabaseConfig
    db_kwargs = {}
    db_kwargs.update(file_overrides["db"])
    db_config = DatabaseConfig(**db_kwargs)

    # Build LocalJobConfig
    job_kwargs = {
        "eval_program_path": str(evaluate_path),
    }
    job_kwargs.update(file_overrides["job"])
    # Set activate_script to the project venv
    project_root = task_dir.parent.parent
    venv_activate = project_root / "venv" / "bin" / "activate"
    if venv_activate.exists():
        job_kwargs["activate_script"] = str(venv_activate)

    job_config = LocalJobConfig(**job_kwargs)

    # Read source files
    init_program_str = initial_path.read_text(encoding="utf-8")
    evaluate_str = evaluate_path.read_text(encoding="utf-8")

    # Build and run
    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=args.verbose,
        init_program_str=init_program_str,
        evaluate_str=evaluate_str,
        max_evaluation_jobs=runner_config.get("max_evaluation_jobs", 1),
        max_proposal_jobs=runner_config.get("max_proposal_jobs", 1),
        max_db_workers=runner_config.get("max_db_workers", 4),
    )

    print(f"Starting ShinkaEvolve experiment")
    print(f"  Task dir:       {task_dir}")
    print(f"  Results dir:    {results_dir}")
    print(f"  Generations:    {args.num_generations}")
    print(f"  LLM models:     {evo_config.llm_models}")
    print(f"  Islands:        {db_config.num_islands}")
    print(f"  Archive size:   {db_config.archive_size}")
    print()

    runner.run()


if __name__ == "__main__":
    main()
