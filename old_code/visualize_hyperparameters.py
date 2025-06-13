#!/usr/bin/env python3
"""
Hyperparameter Visualization Script for GSM8K Grid Search Results

This script loads and visualizes the results from the hyperparameter grid search
performed by basicreasoning.py. It creates various plots and tables to help
understand the impact of different hyperparameters on model performance.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_grid_search_results(summary_file_path):
    """Load the grid search summary JSON file."""
    print(f"Loading results from: {summary_file_path}")
    
    with open(summary_file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['all_combinations'])} hyperparameter combinations")
    return data

def create_results_dataframe(data):
    """Convert grid search results to a pandas DataFrame for easier analysis."""
    
    rows = []
    for combo in data['all_combinations']:
        row = {
            # Hyperparameters
            'temperature': combo['hyperparameters']['temperature'],
            'num_paths': combo['hyperparameters']['num_paths'],
            'num_samples': combo['hyperparameters']['num_samples'],
            'tau_threshold': combo['hyperparameters']['tau_threshold'],
            'step_limit': combo['hyperparameters']['step_limit'],
            
            # Metrics
            'accuracy': combo['metrics']['accuracy'],
            'correct_count': combo['metrics']['correct_count'],
            'total_examples': combo['metrics']['total_examples'],
            'avg_time_per_question': combo['metrics']['avg_time_per_question'],
            'total_time': combo['metrics']['total_time'],
            
            # Derived metrics
            'accuracy_pct': combo['metrics']['accuracy'] * 100,
            'time_efficiency': combo['metrics']['accuracy'] / combo['metrics']['avg_time_per_question'] if combo['metrics']['avg_time_per_question'] > 0 else 0
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Create categorical variables for better plotting
    df['tau_threshold_str'] = df['tau_threshold'].astype(str)
    df['num_paths_str'] = df['num_paths'].astype(str)
    df['step_limit_str'] = df['step_limit'].astype(str)
    
    return df

def create_summary_table(df):
    """Create a summary table of all hyperparameter combinations."""
    
    print("\n" + "="*80)
    print("HYPERPARAMETER GRID SEARCH RESULTS SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Total combinations tested: {len(df)}")
    print(f"  Accuracy range: {df['accuracy_pct'].min():.1f}% - {df['accuracy_pct'].max():.1f}%")
    print(f"  Average accuracy: {df['accuracy_pct'].mean():.1f}%")
    print(f"  Average time per question: {df['avg_time_per_question'].mean():.1f}s")
    
    # Best performing combination
    best_idx = df['accuracy'].idxmax()
    best_combo = df.loc[best_idx]
    
    print(f"\nBest Performing Combination:")
    print(f"  Accuracy: {best_combo['accuracy_pct']:.1f}%")
    print(f"  Temperature: {best_combo['temperature']}")
    print(f"  Num paths: {best_combo['num_paths']}")
    print(f"  Num samples: {best_combo['num_samples']}")
    print(f"  Tau threshold: {best_combo['tau_threshold']}")
    print(f"  Step limit: {best_combo['step_limit']}")
    print(f"  Avg time per question: {best_combo['avg_time_per_question']:.1f}s")
    
    # Most efficient combination (highest accuracy per time)
    most_efficient_idx = df['time_efficiency'].idxmax()
    most_efficient = df.loc[most_efficient_idx]
    
    print(f"\nMost Time-Efficient Combination:")
    print(f"  Time efficiency: {most_efficient['time_efficiency']:.4f} (accuracy/time)")
    print(f"  Accuracy: {most_efficient['accuracy_pct']:.1f}%")
    print(f"  Temperature: {most_efficient['temperature']}")
    print(f"  Num paths: {most_efficient['num_paths']}")
    print(f"  Num samples: {most_efficient['num_samples']}")
    print(f"  Tau threshold: {most_efficient['tau_threshold']}")
    print(f"  Step limit: {most_efficient['step_limit']}")
    print(f"  Avg time per question: {most_efficient['avg_time_per_question']:.1f}s")
    
    return best_combo, most_efficient

def plot_hyperparameter_effects(df, save_path="plots"):
    """Create visualizations showing the effect of each hyperparameter."""
    
    # Create directory for plots
    Path(save_path).mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Main effects plot - how each hyperparameter affects accuracy
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Effects of Hyperparameters on Accuracy', fontsize=16, fontweight='bold')
    
    # Effect of num_paths
    axes[0, 0].boxplot([df[df['num_paths'] == val]['accuracy_pct'] for val in sorted(df['num_paths'].unique())], 
                       labels=sorted(df['num_paths'].unique()))
    axes[0, 0].set_title('Number of Paths vs Accuracy')
    axes[0, 0].set_xlabel('Number of Paths')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Effect of tau_threshold
    tau_values = sorted(df['tau_threshold'].unique())
    axes[0, 1].boxplot([df[df['tau_threshold'] == val]['accuracy_pct'] for val in tau_values], 
                       labels=[f'{val:.2f}' for val in tau_values])
    axes[0, 1].set_title('Tau Threshold vs Accuracy')
    axes[0, 1].set_xlabel('Tau Threshold')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Effect of step_limit
    step_values = sorted(df['step_limit'].unique())
    axes[1, 0].boxplot([df[df['step_limit'] == val]['accuracy_pct'] for val in step_values], 
                       labels=step_values)
    axes[1, 0].set_title('Step Limit vs Accuracy')
    axes[1, 0].set_xlabel('Step Limit')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy vs Time tradeoff
    scatter = axes[1, 1].scatter(df['avg_time_per_question'], df['accuracy_pct'], 
                                c=df['num_paths'], s=60, alpha=0.7, cmap='viridis')
    axes[1, 1].set_title('Accuracy vs Time Tradeoff')
    axes[1, 1].set_xlabel('Average Time per Question (s)')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Number of Paths')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/hyperparameter_effects.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Detailed heatmap for num_paths vs tau_threshold
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy heatmap
    pivot_acc = df.pivot_table(values='accuracy_pct', index='num_paths', columns='tau_threshold', aggfunc='mean')
    sns.heatmap(pivot_acc, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=axes[0])
    axes[0].set_title('Accuracy (%) by Num Paths vs Tau Threshold')
    axes[0].set_xlabel('Tau Threshold')
    axes[0].set_ylabel('Number of Paths')
    
    # Time heatmap
    pivot_time = df.pivot_table(values='avg_time_per_question', index='num_paths', columns='tau_threshold', aggfunc='mean')
    sns.heatmap(pivot_time, annot=True, fmt='.1f', cmap='RdYlBu', ax=axes[1])
    axes[1].set_title('Avg Time per Question (s) by Num Paths Tau Threshold')
    axes[1].set_xlabel('Tau Threshold')
    axes[1].set_ylabel('Number of Paths')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/heatmaps_paths_tau.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Step limit analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Step limit vs accuracy for different num_paths
    for num_paths in sorted(df['num_paths'].unique()):
        subset = df[df['num_paths'] == num_paths]
        grouped = subset.groupby('step_limit')['accuracy_pct'].mean()
        axes[0].plot(grouped.index, grouped.values, marker='o', label=f'{num_paths} paths')
    
    axes[0].set_title('Step Limit vs Accuracy by Number of Paths')
    axes[0].set_xlabel('Step Limit')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Step limit vs time for different num_paths
    for num_paths in sorted(df['num_paths'].unique()):
        subset = df[df['num_paths'] == num_paths]
        grouped = subset.groupby('step_limit')['avg_time_per_question'].mean()
        axes[1].plot(grouped.index, grouped.values, marker='o', label=f'{num_paths} paths')
    
    axes[1].set_title('Step Limit vs Time by Number of Paths')
    axes[1].set_xlabel('Step Limit')
    axes[1].set_ylabel('Average Time per Question (s)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/step_limit_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Efficiency analysis
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create bubble plot showing accuracy vs time with efficiency as bubble size
    bubble_sizes = (df['time_efficiency'] * 1000).clip(10, 200)  # Scale for visibility
    scatter = ax.scatter(df['avg_time_per_question'], df['accuracy_pct'], 
                        s=bubble_sizes, c=df['num_paths'], alpha=0.6, cmap='viridis')
    
    # Add labels for best performing points
    top_performers = df.nlargest(5, 'accuracy_pct')
    for _, row in top_performers.iterrows():
        ax.annotate(f"P:{int(row['num_paths'])}, τ:{row['tau_threshold']}, S:{int(row['step_limit'])}", 
                   (row['avg_time_per_question'], row['accuracy_pct']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    
    ax.set_xlabel('Average Time per Question (s)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Efficiency Analysis: Accuracy vs Time\n(Bubble size = Time Efficiency)')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of Paths')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Tau = 0.01 specific analysis: Step Limit vs Accuracy by Number of Paths
    tau_001_data = df[df['tau_threshold'] == 1e-2]
    
    if not tau_001_data.empty:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create scatter plot with different colors for different num_paths
        for num_paths in sorted(tau_001_data['num_paths'].unique()):
            subset = tau_001_data[tau_001_data['num_paths'] == num_paths]
            ax.scatter(subset['step_limit'], subset['accuracy_pct'], 
                      label=f'{num_paths} paths', s=80, alpha=0.7)
        
        ax.set_xlabel('Step Limit')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Step Limit vs Accuracy for Tau = 1e-2\n(colored by Number of Paths)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/tau_001_step_limit_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("Warning: No data found for tau_threshold = 1e-2")
        

def create_detailed_table(df, save_path="plots"):
    """Create a detailed results table."""
    
    # Sort by accuracy descending
    df_sorted = df.sort_values('accuracy_pct', ascending=False)
    
    # Create a formatted table
    table_data = df_sorted[['temperature', 'num_paths', 'num_samples', 'tau_threshold', 
                           'step_limit', 'accuracy_pct', 'avg_time_per_question', 'time_efficiency']].copy()
    
    # Format columns
    table_data['accuracy_pct'] = table_data['accuracy_pct'].round(1)
    table_data['avg_time_per_question'] = table_data['avg_time_per_question'].round(1)
    table_data['time_efficiency'] = table_data['time_efficiency'].round(4)
    
    # Rename columns for display
    table_data.columns = ['Temp', 'Paths', 'Samples', 'Tau', 'Step Limit', 
                         'Accuracy (%)', 'Time (s)', 'Efficiency']
    
    print("\n" + "="*120)
    print("DETAILED RESULTS TABLE (sorted by accuracy)")
    print("="*120)
    print(table_data.to_string(index=False))
    
    # Save to CSV
    table_data.to_csv(f'{save_path}/detailed_results.csv', index=False)
    print(f"\nDetailed results saved to {save_path}/detailed_results.csv")
    
    return table_data

def analyze_parameter_interactions(df):
    """Analyze interactions between different parameters."""
    
    print("\n" + "="*80)
    print("PARAMETER INTERACTION ANALYSIS")
    print("="*80)
    
    # Correlation matrix
    params = ['num_paths', 'tau_threshold', 'step_limit', 'accuracy_pct', 'avg_time_per_question']
    corr_matrix = df[params].corr()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    # Best combinations for each number of paths
    print("\nBest Accuracy for Each Number of Paths:")
    for num_paths in sorted(df['num_paths'].unique()):
        subset = df[df['num_paths'] == num_paths]
        best = subset.loc[subset['accuracy_pct'].idxmax()]
        print(f"  {num_paths} paths: {best['accuracy_pct']:.1f}% (τ={best['tau_threshold']}, steps={int(best['step_limit'])}, time={best['avg_time_per_question']:.1f}s)")
    
    # Best combinations for each tau threshold
    print("\nBest Accuracy for Each Tau Threshold:")
    for tau in sorted(df['tau_threshold'].unique()):
        subset = df[df['tau_threshold'] == tau]
        best = subset.loc[subset['accuracy_pct'].idxmax()]
        print(f"  τ={tau}: {best['accuracy_pct']:.1f}% (paths={int(best['num_paths'])}, steps={int(best['step_limit'])}, time={best['avg_time_per_question']:.1f}s)")
    
    # Best combinations for each step limit
    print("\nBest Accuracy for Each Step Limit:")
    for steps in sorted(df['step_limit'].unique()):
        subset = df[df['step_limit'] == steps]
        best = subset.loc[subset['accuracy_pct'].idxmax()]
        print(f"  {steps} steps: {best['accuracy_pct']:.1f}% (paths={int(best['num_paths'])}, τ={best['tau_threshold']}, time={best['avg_time_per_question']:.1f}s)")

def main():
    parser = argparse.ArgumentParser(description='Visualize hyperparameter grid search results')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-1.7B',
                       help='Model name')
    
    args = parser.parse_args()
    date = "2025-06-11"
    # Load and process data
    try:
        summary_file = f"logs/{args.model_name}/{date}/gsm8k_grid_search_summary_test_100.json"
        data = load_grid_search_results(summary_file)
        df = create_results_dataframe(data)
        save_path = f"plots/{args.model_name}"
        # Create output directory
        Path(save_path).mkdir(exist_ok=True)
        print(df.head())
        # Generate summary
        best_combo, most_efficient = create_summary_table(df)
        print(best_combo)
        print(most_efficient)
        # Analyze parameter interactions
        analyze_parameter_interactions(df)
        
        # Create detailed table
        table_data = create_detailed_table(df, save_path)
        
        # Create visualizations (unless skipped)
        
        print(f"\nGenerating visualizations...")
        plot_hyperparameter_effects(df, save_path)
        print(f"All plots saved to {save_path}/")
        
        print(f"\nAnalysis complete! Check {save_path}/ for all outputs.")
        
    except FileNotFoundError:
        print(f"Error: Could not find summary file at {summary_file}")
        print("Make sure you have run the grid search first using basicreasoning.py")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 