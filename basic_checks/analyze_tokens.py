#!/usr/bin/env python3
"""
Token Distribution Analysis and Visualization Script

This script loads token data saved by simple.py and generates comprehensive
visualizations and statistical analysis of token usage patterns.
"""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import os
from pathlib import Path

def load_token_data(data_prefix="gsm8k_token_data"):
    """Load token data from saved files."""
    raw_file = f"{data_prefix}_raw.pkl"
    mappings_file = f"{data_prefix}_mappings.json"
    
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Token data file '{raw_file}' not found. Run simple.py first to generate data.")
    
    # Load raw token data
    with open(raw_file, 'rb') as f:
        token_data = pickle.load(f)
    
    # Load token mappings
    with open(mappings_file, 'r') as f:
        token_mappings = json.load(f)
    
    return token_data, token_mappings

def analyze_token_distribution(token_data, token_mappings):
    """Analyze token usage distribution and compute statistics."""
    all_token_ids = token_data['all_token_ids']
    token_counter = Counter(token_data['token_counter'])
    total_tokens = token_data['total_tokens']
    unique_tokens = token_data['unique_tokens']
    vocab_size = token_data['vocab_size']
    
    print("==================== Token Distribution Analysis ====================")
    print(f"Total tokens generated: {total_tokens:,}")
    print(f"Unique tokens used: {unique_tokens:,}")
    print(f"Vocabulary coverage: {token_data['vocab_coverage']:.2%}")
    
    # Get most common tokens
    most_common = token_counter.most_common(20)
    print(f"\nTop 20 most frequent tokens:")
    print("Rank | Token ID | Count | Frequency | Token Text")
    print("-" * 70)
    
    for i, (token_id, count) in enumerate(most_common, 1):
        frequency = count / total_tokens
        token_text = repr(token_mappings.get(str(token_id), f"<ID:{token_id}>"))
        print(f"{i:4d} | {token_id:8d} | {count:5d} | {frequency:8.3%} | {token_text}")
    
    # Analyze frequency distribution
    frequencies = list(token_counter.values())
    frequencies.sort(reverse=True)
    
    print(f"\nFrequency statistics:")
    print(f"  Mean frequency: {np.mean(frequencies):.2f}")
    print(f"  Median frequency: {np.median(frequencies):.2f}")
    print(f"  Max frequency: {max(frequencies)}")
    print(f"  Min frequency: {min(frequencies)}")
    print(f"  Std deviation: {np.std(frequencies):.2f}")
    
    # Calculate concentration metrics
    top_10_percent = int(0.1 * unique_tokens)
    top_10_percent_count = sum(frequencies[:top_10_percent])
    concentration_ratio = top_10_percent_count / total_tokens
    print(f"  Top 10% of tokens account for {concentration_ratio:.1%} of all usage")
    
    return {
        'frequencies': frequencies,
        'most_common': most_common,
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'vocab_size': vocab_size,
        'concentration_ratio': concentration_ratio,
        'stats': {
            'mean': np.mean(frequencies),
            'median': np.median(frequencies),
            'std': np.std(frequencies),
            'max': max(frequencies),
            'min': min(frequencies)
        }
    }

def create_visualizations(analysis_results, token_mappings, output_dir="token_analysis_output"):
    """Create comprehensive visualizations of token distribution."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    frequencies = analysis_results['frequencies']
    most_common = analysis_results['most_common']
    total_tokens = analysis_results['total_tokens']
    
    # Create the main analysis plot
    plt.figure(figsize=(20, 15))
    
    # Subplot 1: Top 50 most frequent tokens
    plt.subplot(2, 3, 1)
    top_50 = most_common[:50]
    top_50_counts = [count for _, count in top_50]
    
    bars = plt.bar(range(len(top_50_counts)), top_50_counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Token Rank')
    plt.ylabel('Frequency')
    plt.title('Top 50 Most Frequent Tokens')
    plt.xticks(range(0, 50, 5))
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Log-log plot of frequency distribution (Zipf's Law)
    plt.subplot(2, 3, 2)
    ranks = np.arange(1, len(frequencies) + 1)
    plt.loglog(ranks, frequencies, 'b-', alpha=0.7, linewidth=2)
    plt.xlabel('Token Rank (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title('Token Frequency Distribution (Zipf\'s Law)')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Cumulative distribution
    plt.subplot(2, 3, 3)
    cumulative_freq = np.cumsum(frequencies) / total_tokens
    plt.plot(ranks, cumulative_freq, 'g-', linewidth=2)
    plt.xlabel('Token Rank')
    plt.ylabel('Cumulative Frequency')
    plt.title('Cumulative Token Frequency Distribution')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Histogram of frequencies
    plt.subplot(2, 3, 4)
    plt.hist(frequencies, bins=50, alpha=0.7, edgecolor='black', color='lightcoral')
    plt.xlabel('Token Frequency')
    plt.ylabel('Number of Tokens')
    plt.title('Distribution of Token Frequencies')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Top tokens with text labels
    plt.subplot(2, 3, 5)
    top_20 = most_common[:20]
    top_20_tokens = [token_mappings.get(str(tid), f"<ID:{tid}>")[:15] for tid, _ in top_20]
    top_20_counts = [count for _, count in top_20]
    
    bars = plt.barh(range(len(top_20_counts)), top_20_counts, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    plt.yticks(range(len(top_20_tokens)), top_20_tokens)
    plt.xlabel('Frequency')
    plt.title('Top 20 Tokens (Horizontal)')
    plt.gca().invert_yaxis()
    
    # Subplot 6: Vocabulary coverage analysis
    plt.subplot(2, 3, 6)
    coverage_points = [0.5, 0.8, 0.9, 0.95, 0.99]
    tokens_needed = []
    
    for coverage in coverage_points:
        cumulative = np.cumsum(frequencies)
        target_count = coverage * total_tokens
        tokens_for_coverage = np.searchsorted(cumulative, target_count) + 1
        tokens_needed.append(tokens_for_coverage)
    
    plt.bar([f"{c*100:.0f}%" for c in coverage_points], tokens_needed, 
            color='mediumpurple', edgecolor='purple', alpha=0.7)
    plt.xlabel('Coverage Percentage')
    plt.ylabel('Number of Unique Tokens Needed')
    plt.title('Tokens Needed for Coverage Levels')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    main_plot_path = f"{output_dir}/token_distribution_analysis.png"
    plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
    print(f"Main analysis plot saved to '{main_plot_path}'")
    plt.close()
    
    return main_plot_path

def save_detailed_report(analysis_results, token_mappings, output_dir="token_analysis_output"):
    """Save a detailed text report of the analysis."""
    
    report_path = f"{output_dir}/token_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TOKEN DISTRIBUTION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Basic statistics
        f.write("BASIC STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total tokens generated: {analysis_results['total_tokens']:,}\n")
        f.write(f"Unique tokens used: {analysis_results['unique_tokens']:,}\n")
        f.write(f"Vocabulary size: {analysis_results['vocab_size']:,}\n")
        f.write(f"Vocabulary coverage: {(analysis_results['unique_tokens'] / analysis_results['vocab_size']) * 100:.2f}%\n")
        f.write(f"Concentration ratio (top 10%): {analysis_results['concentration_ratio'] * 100:.1f}%\n\n")
        
        # Top 50 tokens
        f.write("TOP 50 MOST FREQUENT TOKENS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Rank':<6} {'Token ID':<10} {'Count':<8} {'Freq %':<8} {'Token Text'}\n")
        f.write("-" * 70 + "\n")
        
        for i, (token_id, count) in enumerate(analysis_results['most_common'][:50], 1):
            frequency = (count / analysis_results['total_tokens']) * 100
            token_text = token_mappings.get(str(token_id), f"<ID:{token_id}>")
            f.write(f"{i:<6} {token_id:<10} {count:<8} {frequency:<7.2f}% {repr(token_text)}\n")
    
    print(f"Detailed report saved to '{report_path}'")

def main():
    parser = argparse.ArgumentParser(description="Analyze token distribution from saved data")
    parser.add_argument("--data_prefix", default="gsm8k_token_data", 
                       help="Prefix for data files (default: gsm8k_token_data)")
    parser.add_argument("--output_dir", default="token_analysis_output",
                       help="Output directory for plots and reports (default: token_analysis_output)")
    
    args = parser.parse_args()
    
    try:
        # Load data
        print("Loading token data...")
        token_data, token_mappings = load_token_data(args.data_prefix)
        
        # Analyze distribution
        print("Analyzing token distribution...")
        analysis_results = analyze_token_distribution(token_data, token_mappings)
        
        # Create output directory
        Path(args.output_dir).mkdir(exist_ok=True)
        
        # Generate visualizations
        print("Creating visualizations...")
        create_visualizations(analysis_results, token_mappings, args.output_dir)
        
        # Save detailed report
        print("Generating detailed report...")
        save_detailed_report(analysis_results, token_mappings, args.output_dir)
        
        print(f"\nAnalysis complete! Check the '{args.output_dir}' directory for results.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run simple.py first to generate token data.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 