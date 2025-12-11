# Llama 3.2 1B LoRA Fine-Tuning Project

This project implements a fine-tuning pipeline for `meta-llama/Llama-3.2-1B` on the Alpaca dataset using LoRA.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Hugging Face Access**:
    - Ensure you have access to `meta-llama/Llama-3.2-1B`.
    - The token is configured in `config.py`.

3.  **Data**:
    - The dataset `alpaca_data_cleaned.json` should be in the `data/` directory.

## Running the Project

### 1. Train the Model
Run the training script to load data, fine-tune the model, and save the adapter.
```bash
python train.py
```
- **Output**: Adapter weights saved to `outputs/best_model`.
- **Logs**: Training progress and GPU usage printed to console.

### 2. Evaluate
Evaluate the fine-tuned model on the test set (Perplexity & F1 Score).
```bash
python evaluate.py
```
- **Output**: `outputs/generations/test_set_evaluation.json`

### 3. Inference & Sampling
Generate responses using Greedy, Temperature, and Nucleus sampling on novel instructions.
```bash
python inference.py
```
- **Output**: `outputs/generations/sampling_comparison.json`

## Project Structure
- `config.py`: Hyperparameters and paths.
- `train.py`: Main training loop.
- `evaluate.py`: Metrics calculation.
- `inference.py`: Generation strategies.
- `to_report.md`: Notes for the final report (Theory & Decisions).
