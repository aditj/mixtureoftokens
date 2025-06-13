#!/usr/bin/env python3
"""
Problem Length Distribution Analysis and Visualization Script

This script loads problem length data saved by simple.py and generates comprehensive
visualizations and statistical analysis of how many tokens each problem required.
"""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

def load_problem_length_data(data_prefix="gsm8k_problem_lengths"):
    """Load problem length data from saved files."""
    raw_file = f"{data_prefix}_raw.pkl"
    summary_file = f"{data_prefix}_summary.json"
    
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Problem length data file '{raw_file}' not found. Run simple.py first to generate data.")
    
    # Load raw problem length data
    with open(raw_file, 'rb') as f:
        length_data = pickle.load(f)
    
    # Load summary data
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    
    return length_data, summary_data

def analyze_problem_lengths(length_data, summary_data):
    """Analyze problem length distribution and compute statistics."""
    token_counts = length_data['problem_token_counts']
    problem_results = length_data['problem_results']
    stats = length_data['stats']
    
    print("==================== Problem Length Distribution Analysis ====================")
    print(f"Total problems analyzed: {len(token_counts)}")
    print(f"Total tokens generated: {length_data['total_tokens']:,}")
    print(f"Average tokens per problem: {stats['mean']:.1f}")
    print(f"Median tokens per problem: {stats['median']:.1f}")
    print(f"Standard deviation: {stats['std']:.1f}")
    print(f"Min tokens: {stats['min']}")
    print(f"Max tokens: {stats['max']}")
    print(f"25th percentile: {stats['q25']:.1f}")
    print(f"75th percentile: {stats['q75']:.1f}")
    
    # Calculate additional statistics
    print(f"\nDistribution characteristics:")
    print(f"  Range: {stats['max'] - stats['min']} tokens")
    print(f"  Interquartile range (IQR): {stats['q75'] - stats['q25']:.1f} tokens")
    print(f"  Coefficient of variation: {(stats['std'] / stats['mean']):.2f}")
    
    # Accuracy analysis
    correct_problems = [result for result in problem_results if result['is_correct']]
    incorrect_problems = [result for result in problem_results if not result['is_correct']]
    
    if correct_problems and incorrect_problems:
        correct_tokens = [result['token_count'] for result in correct_problems]
        incorrect_tokens = [result['token_count'] for result in incorrect_problems]
        
        print(f"\nAccuracy vs. Length Analysis:")
        print(f"  Correct answers: {len(correct_problems)} problems")
        print(f"    Average tokens: {np.mean(correct_tokens):.1f}")
        print(f"    Median tokens: {np.median(correct_tokens):.1f}")
        print(f"  Incorrect answers: {len(incorrect_problems)} problems")
        print(f"    Average tokens: {np.mean(incorrect_tokens):.1f}")
        print(f"    Median tokens: {np.median(incorrect_tokens):.1f}")
    
    # Percentile analysis
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentile analysis:")
    for p in percentiles:
        value = np.percentile(token_counts, p)
        print(f"  {p}th percentile: {value:.1f} tokens")
    
    return {
        'token_counts': token_counts,
        'problem_results': problem_results,
        'stats': stats,
        'correct_tokens': [result['token_count'] for result in problem_results if result['is_correct']],
        'incorrect_tokens': [result['token_count'] for result in problem_results if not result['is_correct']]
    }

def create_visualizations(analysis_results, output_dir="problem_length_analysis_output"):
    """Create comprehensive visualizations of problem length distribution."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    token_counts = analysis_results['token_counts']
    correct_tokens = analysis_results['correct_tokens']
    incorrect_tokens = analysis_results['incorrect_tokens']
    stats = analysis_results['stats']
    
    # Create the main analysis plot
    plt.figure(figsize=(20, 12))
    
    # Subplot 1: Main histogram of token counts
    plt.subplot(2, 3, 1)
    n_bins = min(50, len(set(token_counts)))  # Adaptive binning
    plt.hist(token_counts, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.1f}')
    plt.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["median"]:.1f}')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Number of Problems')
    plt.title('Distribution of Token Counts per Problem')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Box plot
    plt.subplot(2, 3, 2)
    plt.boxplot(token_counts, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    plt.ylabel('Number of Tokens')
    plt.title('Box Plot of Token Counts')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Cumulative distribution
    plt.subplot(2, 3, 3)
    sorted_counts = np.sort(token_counts)
    cumulative_prob = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    plt.plot(sorted_counts, cumulative_prob, linewidth=2, color='purple')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function')
    plt.grid(True, alpha=0.3)
    
    # Add percentile lines
    for p, label in [(0.25, '25%'), (0.5, '50%'), (0.75, '75%'), (0.9, '90%')]:
        value = np.percentile(token_counts, p * 100)
        plt.axvline(value, linestyle=':', alpha=0.7, label=f'{label}: {value:.0f}')
    plt.legend()
    
    # Subplot 4: Comparison of correct vs incorrect
    plt.subplot(2, 3, 4)
    if correct_tokens and incorrect_tokens:
        bins = np.linspace(min(token_counts), max(token_counts), 30)
        plt.hist(correct_tokens, bins=bins, alpha=0.6, label=f'Correct ({len(correct_tokens)})', color='green')
        plt.hist(incorrect_tokens, bins=bins, alpha=0.6, label=f'Incorrect ({len(incorrect_tokens)})', color='red')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Number of Problems')
        plt.title('Token Distribution: Correct vs Incorrect')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No accuracy data available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Token Distribution: Correct vs Incorrect')
    
    # Subplot 5: Density plot (smoothed histogram)
    plt.subplot(2, 3, 5)
    plt.hist(token_counts, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Density')
    plt.title('Probability Density of Token Counts')
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Statistics summary as text
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    stats_text = f"""
    Problem Length Statistics
    
    Total Problems: {len(token_counts)}
    Mean: {stats['mean']:.1f} tokens
    Median: {stats['median']:.1f} tokens
    Std Dev: {stats['std']:.1f} tokens
    Min: {stats['min']} tokens
    Max: {stats['max']} tokens
    
    Quartiles:
    Q1 (25%): {stats['q25']:.1f} tokens
    Q3 (75%): {stats['q75']:.1f} tokens
    IQR: {stats['q75'] - stats['q25']:.1f} tokens
    
    Coefficient of Variation: {(stats['std'] / stats['mean']):.3f}
    """
    
    if correct_tokens and incorrect_tokens:
        stats_text += f"""
    
    Accuracy Analysis:
    Correct: {len(correct_tokens)} ({len(correct_tokens)/len(token_counts)*100:.1f}%)
    Avg tokens (correct): {np.mean(correct_tokens):.1f}
    Avg tokens (incorrect): {np.mean(incorrect_tokens):.1f}
    """
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    main_plot_path = f"{output_dir}/problem_length_analysis.png"
    plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
    print(f"Main analysis plot saved to '{main_plot_path}'")
    plt.close()
    
    # Create additional specialized plots
    create_detailed_analysis_plots(analysis_results, output_dir)
    
    return main_plot_path

def create_detailed_analysis_plots(analysis_results, output_dir):
    """Create additional detailed analysis plots."""
    
    token_counts = analysis_results['token_counts']
    problem_results = analysis_results['problem_results']
    stats = analysis_results['stats']
    
    # 1. Detailed percentile analysis
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    percentiles = np.arange(0, 101, 1)
    percentile_values = np.percentile(token_counts, percentiles)
    plt.plot(percentiles, percentile_values, linewidth=2, color='blue')
    plt.xlabel('Percentile')
    plt.ylabel('Number of Tokens')
    plt.title('Percentile Plot (P-P Plot)')
    plt.grid(True, alpha=0.3)
    
    # Mark important percentiles
    for p in [25, 50, 75, 90, 95]:
        value = np.percentile(token_counts, p)
        plt.plot(p, value, 'ro', markersize=8)
        plt.annotate(f'{p}%: {value:.0f}', (p, value), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    plt.subplot(2, 2, 2)
    # Log-scale histogram for better view of distribution tails
    plt.hist(token_counts, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Number of Problems (log scale)')
    plt.yscale('log')
    plt.title('Token Distribution (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Q-Q plot against normal distribution
    from scipy import stats as scipy_stats
    scipy_stats.probplot(token_counts, dist="norm", plot=plt)
    plt.title('Q-Q Plot vs Normal Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Distribution by problem index (to see if there are patterns)
    problem_indices = list(range(len(token_counts)))
    plt.scatter(problem_indices, token_counts, alpha=0.6, s=20)
    plt.xlabel('Problem Index')
    plt.ylabel('Number of Tokens')
    plt.title('Token Count vs Problem Order')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(problem_indices, token_counts, 1)
    p = np.poly1d(z)
    plt.plot(problem_indices, p(problem_indices), "r--", alpha=0.8, 
             label=f'Trend: slope={z[0]:.3f}')
    plt.legend()
    
    plt.tight_layout()
    detailed_plot_path = f"{output_dir}/detailed_length_analysis.png"
    plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
    print(f"Detailed analysis plot saved to '{detailed_plot_path}'")
    plt.close()

def save_detailed_report(analysis_results, output_dir="problem_length_analysis_output"):
    """Save a detailed text report of the analysis."""
    
    report_path = f"{output_dir}/problem_length_report.txt"
    
    token_counts = analysis_results['token_counts']
    problem_results = analysis_results['problem_results']
    stats = analysis_results['stats']
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PROBLEM LENGTH DISTRIBUTION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Basic statistics
        f.write("BASIC STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total problems analyzed: {len(token_counts)}\n")
        f.write(f"Total tokens generated: {sum(token_counts):,}\n")
        f.write(f"Average tokens per problem: {stats['mean']:.2f}\n")
        f.write(f"Median tokens per problem: {stats['median']:.2f}\n")
        f.write(f"Standard deviation: {stats['std']:.2f}\n")
        f.write(f"Minimum tokens: {stats['min']}\n")
        f.write(f"Maximum tokens: {stats['max']}\n")
        f.write(f"Range: {stats['max'] - stats['min']}\n")
        f.write(f"Coefficient of variation: {(stats['std'] / stats['mean']):.3f}\n\n")
        
        # Percentile analysis
        f.write("PERCENTILE ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(token_counts, p)
            f.write(f"{p:2d}th percentile: {value:6.1f} tokens\n")
        f.write(f"\nInterquartile range (Q3-Q1): {stats['q75'] - stats['q25']:.1f} tokens\n\n")
        
        # Accuracy analysis
        if problem_results:
            correct_results = [r for r in problem_results if r['is_correct']]
            incorrect_results = [r for r in problem_results if not r['is_correct']]
            
            f.write("ACCURACY ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total correct answers: {len(correct_results)} ({len(correct_results)/len(problem_results)*100:.1f}%)\n")
            f.write(f"Total incorrect answers: {len(incorrect_results)} ({len(incorrect_results)/len(problem_results)*100:.1f}%)\n\n")
            
            if correct_results:
                correct_tokens = [r['token_count'] for r in correct_results]
                f.write(f"Correct answers - token statistics:\n")
                f.write(f"  Average: {np.mean(correct_tokens):.1f} tokens\n")
                f.write(f"  Median: {np.median(correct_tokens):.1f} tokens\n")
                f.write(f"  Std dev: {np.std(correct_tokens):.1f} tokens\n")
                f.write(f"  Min: {min(correct_tokens)} tokens\n")
                f.write(f"  Max: {max(correct_tokens)} tokens\n\n")
            
            if incorrect_results:
                incorrect_tokens = [r['token_count'] for r in incorrect_results]
                f.write(f"Incorrect answers - token statistics:\n")
                f.write(f"  Average: {np.mean(incorrect_tokens):.1f} tokens\n")
                f.write(f"  Median: {np.median(incorrect_tokens):.1f} tokens\n")
                f.write(f"  Std dev: {np.std(incorrect_tokens):.1f} tokens\n")
                f.write(f"  Min: {min(incorrect_tokens)} tokens\n")
                f.write(f"  Max: {max(incorrect_tokens)} tokens\n\n")
        
        # Extreme cases
        f.write("EXTREME CASES:\n")
        f.write("-" * 40 + "\n")
        
        # Shortest problems
        shortest_indices = np.argsort(token_counts)[:5]
        f.write("5 shortest responses:\n")
        for i, idx in enumerate(shortest_indices, 1):
            result = problem_results[idx]
            f.write(f"{i}. Problem {idx}: {result['token_count']} tokens (correct: {result['is_correct']})\n")
        
        f.write("\n")
        
        # Longest problems
        longest_indices = np.argsort(token_counts)[-5:][::-1]
        f.write("5 longest responses:\n")
        for i, idx in enumerate(longest_indices, 1):
            result = problem_results[idx]
            f.write(f"{i}. Problem {idx}: {result['token_count']} tokens (correct: {result['is_correct']})\n")
    
    print(f"Detailed report saved to '{report_path}'")

def main():
    parser = argparse.ArgumentParser(description="Analyze problem length distribution from saved data")
    parser.add_argument("--data_prefix", default="gsm8k_problem_lengths", 
                       help="Prefix for data files (default: gsm8k_problem_lengths)")
    parser.add_argument("--output_dir", default="problem_length_analysis_output",
                       help="Output directory for plots and reports (default: problem_length_analysis_output)")
    
    args = parser.parse_args()
    
    try:
        # Load data
        print("Loading problem length data...")
        length_data, summary_data = load_problem_length_data(args.data_prefix)
        
        # Analyze distribution
        print("Analyzing problem length distribution...")
        analysis_results = analyze_problem_lengths(length_data, summary_data)
        
        # Create output directory
        Path(args.output_dir).mkdir(exist_ok=True)
        
        # Generate visualizations
        print("Creating visualizations...")
        create_visualizations(analysis_results, args.output_dir)
        
        # Save detailed report
        print("Generating detailed report...")
        save_detailed_report(analysis_results, args.output_dir)
        
        print(f"\nAnalysis complete! Check the '{args.output_dir}' directory for results.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run simple.py first to generate problem length data.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 