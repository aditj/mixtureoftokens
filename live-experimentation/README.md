# Live Token Generation Experiment

Interactive web interface for experimenting with embedding mixture generation techniques in real-time.

## Features

- **Interactive Prompt Editor**: Modify system and user prompts directly in the web interface
- **Parameter Control**: Adjust generation parameters (k, T_e, T_total, temperature) on the fly
- **Experiment Strategies**: Choose from multiple sampling strategies:
  - Non-uniform (weighted mixture)
  - Element-wise max
  - Inverse probability weighting
  - Dirichlet sampling
  - Nucleus sampling
  - Answer directly + element-wise max
- **Real-time Visualization**: View Phase 1 (embedding mixture) and Phase 2 (standard generation) tokens with probability-based color coding
- **Live Generation**: Generate text and see results immediately without running scripts

## Setup

1. **Install Dependencies**:
   ```bash
   cd live-experimentation
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Access the Interface**:
   Open your browser and go to `http://localhost:5000`

## Usage

1. **Configure Prompts**: 
   - Enter your system prompt (e.g., "You are a helpful mathematical reasoning assistant")
   - Enter your user prompt (e.g., "What is 25 + 17?")

2. **Set Parameters**:
   - **k**: Number of top tokens to consider for mixture (default: 5)
   - **T_e**: Number of exploration rounds in Phase 1 (default: 50)
   - **T_total**: Total number of generation rounds (default: 250)
   - **Temperature**: Sampling temperature (default: 0.8)
   - **Experiment Strategy**: Choose sampling method

3. **Generate**: Click "Generate Text" to run the experiment

4. **View Results**:
   - Generated text appears in the results section
   - Phase 1 visualization shows token mixtures with probability-based coloring
   - Phase 2 visualization shows standard token-by-token generation
   - Hover over tokens to see detailed information

## Experiment Strategies

- **non_uniform**: Weighted mixture based on token probabilities
- **element_wise_max**: Takes element-wise maximum of top-k token embeddings
- **inverse_p**: Uses inverse probability weighting
- **dirichlet**: Samples from Dirichlet distribution
- **nucleus**: Uses nucleus (top-p) sampling for token selection
- **answer_directly_element_wise_max**: Skips thinking phase, goes directly to answer

## Architecture

- **Backend**: Flask server with integrated generation logic from `simple.py`
- **Frontend**: Interactive HTML interface with real-time visualization
- **Model**: Uses Qwen/Qwen2.5-3B-Instruct (configurable in `app.py`)

## Customization

- Modify `model_name` in `app.py` to use different models
- Adjust visualization parameters in the HTML template
- Add new experiment strategies by extending the generation function

## Troubleshooting

- Ensure you have sufficient GPU memory for the model
- Check that all dependencies are installed correctly
- Verify the model downloads successfully on first run 