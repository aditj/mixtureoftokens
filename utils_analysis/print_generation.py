#!/usr/bin/env python3

from __future__ import annotations

import re
import json
import glob
import argparse
import os
from typing import Dict, List, Tuple
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

FILENAME_RE = re.compile(r"T_e_(\d+)_k_(\d+)")

def extract_te_k(filename: str) -> Tuple[int, int] | None:
    """Return (T_e, k) parsed from the filename or None if pattern not found."""
    m = FILENAME_RE.search(filename)

    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def load_results(directory: str) -> List[Dict]:
    """Load stats for each experiment JSON found in *directory*."""
    results = []
    for json_path in glob.glob(os.path.join(directory, "*.json")):
        parsed = extract_te_k(os.path.basename(json_path))
        if parsed is None:
            continue  # skip files that don't match pattern
        T_e, k = parsed
        with open(json_path, "r") as f:
            data = json.load(f)
        stats = data.get("embedding_mixture", {}).get("stats", {})
        accuracy = data.get("embedding_mixture", {}).get("accuracy")
        avg_tokens = stats.get("avg_tokens")

        # ------------------------------------------------------------------
        # Bootstrapped confidence interval on accuracy (over 200 examples)
        # ------------------------------------------------------------------
        result_entries = data.get("embedding_mixture", {}).get("results", [])
        is_correct_list = [1 if r.get("is_correct") else 0 for r in result_entries]
        token_counts = data.get("embedding_mixture", {}).get("token_counts", [])
        if accuracy is None or avg_tokens is None or not is_correct_list or not token_counts:
            continue  # incomplete record
        # Bootstrapping (95% CI) for accuracy
        n_samples = len(is_correct_list)
        n_boot = 10000
        boot_acc = []
        for _ in range(n_boot):
            sample = np.random.choice(is_correct_list, size=n_samples, replace=True)
            boot_acc.append(np.mean(sample))
        
        acc_ci_lower, acc_ci_upper = np.percentile(boot_acc, [2.5, 97.5])

        # Bootstrapping (95% CI) for average tokens
        n_tok_samples = len(token_counts)
        boot_tok = []
        for _ in range(n_boot):
            sample_tok = np.random.choice(token_counts, size=n_tok_samples, replace=True)
            boot_tok.append(np.mean(sample_tok))

        tok_ci_lower, tok_ci_upper = np.percentile(boot_tok, [2.5, 97.5])

        results.append({
            "T_e": T_e,
            "k": k,
            "accuracy": accuracy,
            "avg_tokens": avg_tokens,
            "acc_ci_lower": acc_ci_lower,
            "acc_ci_upper": acc_ci_upper,
            "tok_ci_lower": tok_ci_lower,
            "tok_ci_upper": tok_ci_upper,
        })
    return results


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def create_plots(results: List[Dict], output_file: str = None):
    """Create 1x2 plots: accuracy and avg tokens with different T_e values as different colors."""
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Group results by T_e
    grouped = {}
    for entry in results:
        T_e = entry["T_e"]
        if T_e not in grouped:
            grouped[T_e] = []
        grouped[T_e].append(entry)
    
    # Sort T_e values
    T_e_values = sorted(grouped.keys())
    
    if len(T_e_values) == 0:
        print("No data to plot")
        return
    
    # Create figure with 1 row and 2 columns
    fig, (ax_acc, ax_tok) = plt.subplots(1, 2, figsize=(14, 6))
    
    fig.suptitle('GSM8K Generation Comparison Results', fontsize=16, fontweight='bold')
    
    # Get colors for different T_e values
    colors = sns.color_palette("husl", len(T_e_values))
    
    for i, T_e in enumerate(T_e_values):
        data = grouped[T_e]
        # Sort by k for consistent plotting
        data = sorted(data, key=lambda x: x["k"])
        
        k_values = [entry["k"] for entry in data]
        accuracies = [entry["accuracy"] for entry in data]
        acc_ci_lower = [entry["acc_ci_lower"] for entry in data]
        acc_ci_upper = [entry["acc_ci_upper"] for entry in data]
        
        avg_tokens = [entry["avg_tokens"] for entry in data]
        tok_ci_lower = [entry["tok_ci_lower"] for entry in data]
        tok_ci_upper = [entry["tok_ci_upper"] for entry in data]
        
        color = colors[i]
        
        # Accuracy plot
        ax_acc.errorbar(k_values, accuracies, 
                       yerr=[np.array(accuracies) - np.array(acc_ci_lower),
                             np.array(acc_ci_upper) - np.array(accuracies)],
                       marker='o', linewidth=2.5, markersize=8, capsize=5, capthick=2,
                       color=color, label=f'T_e = {T_e}')
        
        # Token count plot
        ax_tok.errorbar(k_values, avg_tokens,
                       yerr=[np.array(avg_tokens) - np.array(tok_ci_lower),
                             np.array(tok_ci_upper) - np.array(avg_tokens)],
                       marker='s', linewidth=2.5, markersize=8, capsize=5, capthick=2,
                       color=color, label=f'T_e = {T_e}')
    
    # Customize accuracy plot
    ax_acc.set_title('Accuracy vs Number of Paths', fontweight='bold', fontsize=14)
    ax_acc.set_xlabel('Number of paths averaged over', fontsize=12)
    ax_acc.set_ylabel('Accuracy', fontsize=12)
    ax_acc.set_ylim(0, 1)
    ax_acc.legend(title='Temperature', title_fontsize=11, fontsize=10)
    
    # Customize token count plot
    ax_tok.set_title('Average Tokens vs Number of Paths', fontweight='bold', fontsize=14)
    ax_tok.set_xlabel('Number of paths averaged over', fontsize=12)
    ax_tok.set_ylabel('Average Tokens', fontsize=12)
    ax_tok.legend(title='Exploration Period', title_fontsize=11, fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    exp_name = "answer_directly"
    T_total = 500   
    N_examples = 100
    temperature = 0.6
    parser = argparse.ArgumentParser(description="Plot accuracy vs avg tokens for GSM8K experiments")
    parser.add_argument("--dir", default=f"generation_comparison/{exp_name}/{T_total}_{N_examples}_{temperature}/", help="Directory with JSON logs")
    parser.add_argument("--out", default=f"generation_comparison/{exp_name}/{T_total}_{N_examples}_{temperature}/scatter.png", help="Optional output filename (e.g. scatter.png)")
    args = parser.parse_args()

    res = load_results(args.dir)
    if not res:
        print(f"No valid JSON logs found in {args.dir}")
        return
    print("\n" + "="*80)
    print("GSM8K Generation Comparison Results")
    print("="*80)
    
    # Group results by T_e and k for better organization
    grouped = {}
    for entry in res:
        T_e = entry["T_e"]
        k = entry["k"]
        if T_e not in grouped:
            grouped[T_e] = {}
        if k not in grouped[T_e]:
            grouped[T_e][k] = []
        grouped[T_e][k].append(entry)
    
    # Print organized table
    for T_e in sorted(grouped.keys()):
        print(f"\nT_e = {T_e}:")
        print("-" * 60)
        print(f"{'k':<4} {'Accuracy':<10} {'Acc 95% CI':<17} {'Avg Tokens':<12} {'Tok 95% CI':<17}")
        print("-" * 60)
        
        for k in sorted(grouped[T_e].keys()):
            for entry in grouped[T_e][k]:
                acc_ci = f"[{entry['acc_ci_lower']:.3f}, {entry['acc_ci_upper']:.3f}]"
                tok_ci = f"[{entry['tok_ci_lower']:.1f}, {entry['tok_ci_upper']:.1f}]"
                print(f"{k:<4} {entry['accuracy']:<10.3f} {acc_ci:<17} {entry['avg_tokens']:<12.1f} {tok_ci:<17}")
    
    print("\n" + "="*80)
    
    # Create plots
    create_plots(res, args.out)

if __name__ == "__main__":
    main() 