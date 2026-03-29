"""
Plot ShinkaEvolve Evolution Progress — Easy vs Hardened Task Comparison

Generates the key visualization showing why task difficulty calibration
matters for evolutionary search effectiveness.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_generation_scores(results_dir):
    """Load combined_score for each generation from results directory."""
    scores = []
    gen = 0
    while True:
        metrics_path = os.path.join(results_dir, f'gen_{gen}', 'results', 'metrics.json')
        correct_path = os.path.join(results_dir, f'gen_{gen}', 'results', 'correct.json')
        if not os.path.exists(metrics_path):
            break
        with open(metrics_path) as f:
            metrics = json.load(f)
        with open(correct_path) as f:
            correct = json.load(f)
        scores.append({
            'gen': gen,
            'score': metrics.get('combined_score', 0.0),
            'auc': metrics.get('roc_auc', 0.0),
            'f1': metrics.get('f1_score', 0.0),
            'correct': correct.get('correct', False),
        })
        gen += 1
    return scores


def plot_evolution_comparison(easy_scores, hard_scores, output_path):
    """Plot side-by-side comparison of easy vs hardened evolution runs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel 1: Combined Score ---
    ax = axes[0]

    # Easy run
    easy_gens = [s['gen'] for s in easy_scores]
    easy_vals = [s['score'] for s in easy_scores]
    easy_correct = [s['correct'] for s in easy_scores]
    easy_best = [max(easy_vals[:i+1]) for i in range(len(easy_vals))]

    ax.scatter([g for g, c in zip(easy_gens, easy_correct) if c],
               [v for v, c in zip(easy_vals, easy_correct) if c],
               color='#2E86AB', alpha=0.5, s=30, zorder=3)
    ax.scatter([g for g, c in zip(easy_gens, easy_correct) if not c],
               [v for v, c in zip(easy_vals, easy_correct) if not c],
               color='#2E86AB', alpha=0.2, s=20, marker='x', zorder=3)
    ax.plot(easy_gens, easy_best, color='#2E86AB', linewidth=2, label='Easy task (best so far)')

    # Hard run
    hard_gens = [s['gen'] for s in hard_scores]
    hard_vals = [s['score'] for s in hard_scores]
    hard_correct = [s['correct'] for s in hard_scores]
    hard_best = [max(hard_vals[:i+1]) for i in range(len(hard_vals))]

    ax.scatter([g for g, c in zip(hard_gens, hard_correct) if c],
               [v for v, c in zip(hard_vals, hard_correct) if c],
               color='#E8553D', alpha=0.5, s=30, zorder=3)
    ax.scatter([g for g, c in zip(hard_gens, hard_correct) if not c],
               [v for v, c in zip(hard_vals, hard_correct) if not c],
               color='#E8553D', alpha=0.2, s=20, marker='x', zorder=3)
    ax.plot(hard_gens, hard_best, color='#E8553D', linewidth=2, label='Hardened task (best so far)')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Combined Score', fontsize=12)
    ax.set_title('Evolution Progress: Combined Score', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add improvement annotations
    easy_improvement = (max(easy_vals) - easy_vals[0]) / easy_vals[0] * 100
    hard_improvement = (max(hard_vals) - hard_vals[0]) / hard_vals[0] * 100
    ax.annotate(f'+{easy_improvement:.1f}%', xy=(easy_gens[-1], max(easy_vals)),
                fontsize=10, color='#2E86AB', fontweight='bold',
                ha='right', va='bottom')
    ax.annotate(f'+{hard_improvement:.1f}%', xy=(hard_gens[-1], max(hard_vals)),
                fontsize=10, color='#E8553D', fontweight='bold',
                ha='right', va='bottom')

    # --- Panel 2: ROC AUC ---
    ax = axes[1]

    easy_auc = [s['auc'] for s in easy_scores if s['correct']]
    easy_auc_gens = [s['gen'] for s in easy_scores if s['correct']]
    hard_auc = [s['auc'] for s in hard_scores if s['correct']]
    hard_auc_gens = [s['gen'] for s in hard_scores if s['correct']]

    ax.plot(easy_auc_gens, easy_auc, 'o-', color='#2E86AB', linewidth=1.5,
            markersize=5, alpha=0.7, label='Easy task')
    ax.plot(hard_auc_gens, hard_auc, 's-', color='#E8553D', linewidth=1.5,
            markersize=5, alpha=0.7, label='Hardened task')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('ROC AUC', fontsize=12)
    ax.set_title('ROC AUC Over Generations', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: F1 Score ---
    ax = axes[2]

    easy_f1 = [s['f1'] for s in easy_scores if s['correct']]
    easy_f1_gens = [s['gen'] for s in easy_scores if s['correct']]
    hard_f1 = [s['f1'] for s in hard_scores if s['correct']]
    hard_f1_gens = [s['gen'] for s in hard_scores if s['correct']]

    ax.plot(easy_f1_gens, easy_f1, 'o-', color='#2E86AB', linewidth=1.5,
            markersize=5, alpha=0.7, label='Easy task')
    ax.plot(hard_f1_gens, hard_f1, 's-', color='#E8553D', linewidth=1.5,
            markersize=5, alpha=0.7, label='Hardened task')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score Over Generations', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('ShinkaEvolve: Task Difficulty Determines Evolution Effectiveness',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary_table(easy_scores, hard_scores, output_path):
    """Create a summary comparison table as an image."""
    easy_correct = [s for s in easy_scores if s['correct']]
    hard_correct = [s for s in hard_scores if s['correct']]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    table_data = [
        ['Metric', 'Easy Task', 'Hardened Task', 'Difference'],
        ['Baseline score', f"{easy_scores[0]['score']:.3f}", f"{hard_scores[0]['score']:.3f}", ''],
        ['Best evolved score', f"{max(s['score'] for s in easy_scores):.3f}",
         f"{max(s['score'] for s in hard_scores):.3f}", ''],
        ['Improvement', f"+{(max(s['score'] for s in easy_scores) - easy_scores[0]['score']) / easy_scores[0]['score'] * 100:.1f}%",
         f"+{(max(s['score'] for s in hard_scores) - hard_scores[0]['score']) / hard_scores[0]['score'] * 100:.1f}%",
         f"{(max(s['score'] for s in hard_scores) - hard_scores[0]['score']) / hard_scores[0]['score'] * 100 / ((max(s['score'] for s in easy_scores) - easy_scores[0]['score']) / easy_scores[0]['score'] * 100):.0f}x more"],
        ['Best AUC', f"{max(s['auc'] for s in easy_correct):.3f}",
         f"{max(s['auc'] for s in hard_correct):.3f}", ''],
        ['Best F1', f"{max(s['f1'] for s in easy_correct):.3f}",
         f"{max(s['f1'] for s in hard_correct):.3f}", ''],
        ['Correct programs', f"{len(easy_correct)}/{len(easy_scores)} ({len(easy_correct)/len(easy_scores)*100:.0f}%)",
         f"{len(hard_correct)}/{len(hard_scores)} ({len(hard_correct)/len(hard_scores)*100:.0f}%)", ''],
        ['Still improving at gen 20?', 'No (plateau gen 2)', 'Yes (new best gen 19)', ''],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(4):
        table[0, j].set_facecolor('#2E86AB')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight improvement row
    for j in range(4):
        table[3, j].set_facecolor('#FFF3CD')

    plt.title('ShinkaEvolve Results: Easy vs Hardened Task', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    easy_dir = os.path.join(PROJECT_ROOT, 'experiments', 'shinkaevolve', 'results')
    hard_dir = os.path.join(PROJECT_ROOT, 'experiments', 'shinkaevolve', 'results_hardened')
    output_dir = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    print("Loading ShinkaEvolve results...")

    easy_scores = load_generation_scores(easy_dir)
    hard_scores = load_generation_scores(hard_dir)

    if not easy_scores or not hard_scores:
        print("ERROR: Missing results. Run both experiments first.")
        print(f"  Easy: {easy_dir} ({'found' if easy_scores else 'MISSING'})")
        print(f"  Hard: {hard_dir} ({'found' if hard_scores else 'MISSING'})")
        return

    print(f"  Easy task: {len(easy_scores)} generations")
    print(f"  Hardened task: {len(hard_scores)} generations")

    plot_evolution_comparison(
        easy_scores, hard_scores,
        os.path.join(output_dir, 'shinkaevolve_evolution_comparison.png')
    )

    plot_summary_table(
        easy_scores, hard_scores,
        os.path.join(output_dir, 'shinkaevolve_summary_table.png')
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
