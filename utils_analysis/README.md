# Utils Analysis

A comprehensive analysis toolkit for visualizing and analyzing token generation patterns in mixture-of-experts models, particularly focused on embedding mixture experiments with GSM8K mathematical reasoning tasks.

## Overview

This toolkit provides multiple ways to analyze and visualize the results of token generation experiments:

- **Interactive Web Dashboard**: Real-time visualization of token generation phases with detailed analysis
- **Statistical Analysis**: Plotting accuracy vs token usage with confidence intervals
- **Token Dictionary Management**: Tools for creating and managing tokenizer vocabularies
- **Data Processing**: Utilities for loading and analyzing experiment results

## Components

### 1. Interactive Token Generation Dashboard

**Files**: `token_generation_dashboard.html`, `dashboard_server.py`, `token_dict.json`

A web-based dashboard for visualizing token generation in two-phase experiments:

#### Features:
- **Phase 1 Visualization**: Interactive table showing embedding mixture generation with:
  - Token-by-token generation across multiple rounds
  - Probability-based color coding (darker = higher probability)
  - Hover tooltips with token details (ID, text, probability)
  - Zoom controls for detailed inspection
  
- **Phase 2 Visualization**: Standard generation tokens displayed as a sequence
- **File Management**: Browse and load different experiment results
- **Question Navigation**: Select specific questions to analyze
- **Real-time Data Loading**: API-based data fetching with caching

#### Usage:
```bash
# Start the dashboard server
cd utils_analysis
python dashboard_server.py

# Open your browser and navigate to:
# http://localhost:8000/token_generation_dashboard.html
```

The dashboard will automatically:
- Scan for JSON files in `../generation_comparison/*/` directories
- Load the token dictionary for text conversion
- Provide an interface to browse experiments and questions

### 2. Statistical Analysis & Plotting

**File**: `print_generation.py`

Generate publication-ready plots comparing accuracy and token usage across different experimental parameters.

#### Features:
- **Bootstrapped Confidence Intervals**: 95% CI for both accuracy and token counts
- **Multi-parameter Analysis**: Compare different T_e (exploration temperature) and k (path count) values
- **Professional Visualization**: Publication-ready matplotlib/seaborn plots
- **Tabular Summary**: Organized results table with statistical metrics

#### Usage:
```bash
# Generate plots for default experiment
python print_generation.py

# Specify custom directory and output
python print_generation.py --dir path/to/results --out results_plot.png

# Example with specific experiment
python print_generation.py \
    --dir generation_comparison/answer_directly/500_100_0.6/ \
    --out analysis_results.png
```

#### Output:
- Side-by-side plots: Accuracy vs Number of Paths | Average Tokens vs Number of Paths
- Detailed statistical table with confidence intervals
- High-resolution PNG plots suitable for publications

### 3. Token Dictionary Management

**File**: `create_token_dict.py`

Create and manage token dictionaries for converting token IDs to human-readable text in the dashboard.

#### Features:
- **Multiple Model Support**: Works with any HuggingFace tokenizer
- **Format Options**: Export as JSON or JavaScript
- **Special Token Handling**: Proper display of whitespace and special characters
- **Batch Processing**: Efficient vocabulary extraction

#### Usage:
```bash
# Create token dictionary for default model (Qwen2.5-3B-Instruct)
python create_token_dict.py

# Use custom model
python create_token_dict.py --model microsoft/DialoGPT-large

# Export as JavaScript for direct HTML embedding
python create_token_dict.py --format js --output token_dict.js

# Show sample tokens for verification
python create_token_dict.py --sample
```

#### Special Character Handling:
- Space: `⎵` (visible space)
- Newline: `↵` (visible newline)
- Tab: `→` (visible tab)
- Invalid tokens: `[Invalid Token ID]`

## Data Format

The toolkit expects JSON files with the following structure:

```json
{
  "embedding_mixture": {
    "accuracy": 0.85,
    "stats": {
      "avg_tokens": 245.6
    },
    "results": [
      {
        "question": "What is 2+2?",
        "ground_truth": "4",
        "predicted_answer": "4",
        "is_correct": true,
        "phase_info": {
          "total_tokens": 250,
          "phase1_tokens": 150,
          "phase2_tokens": 100,
          "phase1_token_ids": [
            [[101, 102, 103], [0.8, 0.6, 0.9]],
            [[104, 105, 106], [0.7, 0.8, 0.5]]
          ],
          "phase2_token_ids": [107, 108, 109, 110]
        }
      }
    ],
    "token_counts": [250, 230, 280, ...]
  }
}
```

## File Structure Requirements

```
project_root/
├── utils_analysis/
│   ├── dashboard_server.py
│   ├── token_generation_dashboard.html
│   ├── create_token_dict.py
│   ├── token_dict.json
│   └── print_generation.py
└── generation_comparison/
    └── experiment_name/
        └── T_total_N_examples_temperature/
            ├── generation_comparison_T_e_50_k_4.json
            ├── generation_comparison_T_e_100_k_4.json
            └── ...
```

## Dependencies

Install required packages:

```bash
pip install transformers torch matplotlib seaborn numpy
```

## Quick Start

1. **Generate token dictionary**:
   ```bash
   python create_token_dict.py --sample
   ```

2. **Start the dashboard**:
   ```bash
   python dashboard_server.py
   ```

3. **Create statistical plots**:
   ```bash
   python print_generation.py --dir ../generation_comparison/your_experiment/
   ```

4. **Open dashboard in browser**:
   Navigate to `http://localhost:8000/token_generation_dashboard.html`

## Advanced Features

### Dashboard API Endpoints

- `GET /api/files` - List available experiment files
- `GET /api/questions?file=path` - Get questions for a specific file
- `GET /api/question-data?file=path&question=index` - Get detailed question data

### Customization

- **Color Schemes**: Modify probability-to-color mapping in the HTML dashboard
- **Zoom Levels**: Adjust zoom ranges in the dashboard controls
- **Plot Styling**: Customize matplotlib/seaborn themes in `print_generation.py`
- **Token Display**: Modify special character representations in `create_token_dict.py`

## Troubleshooting

### Common Issues:

1. **Token dictionary not loading**: Ensure `token_dict.json` is in the same directory as the HTML file
2. **No experiment files found**: Check that JSON files follow the naming pattern `generation_comparison_T_e_*_k_*.json`
3. **Dashboard not loading data**: Verify the server is running and CORS headers are properly set
4. **Plot generation fails**: Ensure all required dependencies are installed and data format is correct

### Debug Mode:

Enable verbose logging by checking browser console (F12) for the dashboard, or adding `--verbose` flag to Python scripts. 