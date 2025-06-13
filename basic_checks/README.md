# Problem Length Distribution Analysis System

This system consists of two main scripts for analyzing how many tokens each problem requires to solve (problem length distribution):

## 1. Data Collection (`simple_backup.py`)
The main script that runs GSM8K math problems and collects token usage data.

### What it does:
- Runs the Qwen model on math problems
- Tracks how many tokens each problem requires to solve
- Saves problem length data and results for later analysis

### Files generated:
- `gsm8k_problem_lengths_raw.pkl` - Raw problem length data (pickled for fast loading)
- `gsm8k_problem_lengths_summary.json` - Human-readable summary with statistics

### Usage:
```bash
python simple_backup.py
```

## 2. Analysis and Visualization (`analyze_problem_lengths.py`)
Separate script that loads the saved problem length data and generates comprehensive analysis.

### What it provides:
- Detailed statistical analysis of problem length distribution
- Multiple visualization plots showing how many tokens problems require
- Comparison of token usage between correct and incorrect answers
- Text reports with distribution statistics

### Output files:
- `problem_length_analysis_output/problem_length_analysis.png` - Main visualization
- `problem_length_analysis_output/detailed_length_analysis.png` - Additional detailed plots
- `problem_length_analysis_output/problem_length_report.txt` - Detailed text report

### Usage:
```bash
# Basic usage (uses default file names)
python analyze_problem_lengths.py

# Custom data prefix and output directory
python analyze_problem_lengths.py --data_prefix my_problem_data --output_dir my_analysis_output
```

### Command line options:
- `--data_prefix`: Prefix for input data files (default: gsm8k_problem_lengths)
- `--output_dir`: Directory for output files (default: problem_length_analysis_output)

## Workflow

1. **Collect data**: Run `simple.py` to generate problem length data
2. **Analyze**: Run `analyze_problem_lengths.py` to create visualizations and reports

## Visualizations Generated

The analysis script creates comprehensive plots showing:

**Main Analysis Plot (6 subplots):**
1. **Histogram of Token Counts** - Distribution of how many tokens each problem required
2. **Box Plot** - Shows quartiles, median, and outliers of token counts
3. **Cumulative Distribution** - Shows percentiles of token usage
4. **Correct vs Incorrect** - Compares token usage for correct and incorrect answers
5. **Probability Density** - Smoothed distribution curve
6. **Statistics Summary** - Text summary of key statistics

**Detailed Analysis Plot (4 subplots):**
1. **Percentile Plot** - Detailed percentile analysis
2. **Log-Scale Histogram** - Better view of distribution tails
3. **Q-Q Plot** - Tests if distribution follows normal distribution
4. **Token Count vs Problem Order** - Shows if there are patterns over time

## Requirements

### For data collection (simple.py):
- torch
- transformers
- datasets
- tqdm

### For analysis (analyze_problem_lengths.py):
- numpy
- matplotlib
- scipy (for Q-Q plots)
- (Standard library: pickle, json, argparse, os, pathlib)

## Example Output

The system will show you:
- **Distribution of problem lengths**: How many tokens most problems require to solve
- **Statistical measures**: Mean, median, standard deviation, quartiles of token counts
- **Accuracy correlation**: Do longer responses tend to be more or less accurate?
- **Outlier analysis**: Which problems require unusually many or few tokens?
- **Percentile analysis**: What token count represents the 90th percentile of responses?
- **Distribution shape**: Is the distribution normal, skewed, or has multiple peaks?

This helps understand the model's "verbosity patterns" - whether it consistently uses similar amounts of reasoning for all problems, or if some problems require much more extensive reasoning than others. 