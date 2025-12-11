# Project Report Notes

This document tracks technical decisions, challenges encountered, and parameter explanations to assist in writing the final report.

## Technical Decisions & Rationale

### Project Structure
- **Structure**: Followed the `project/` layout with separate scripts for training, evaluation, and inference.
- **Reasoning**: Modular design allows for easier debugging and testing of individual components (e.g., testing inference without retraining).

### Data Management
- **Dataset**: `alpaca_data_cleaned.json` (Stanford Alpaca).
- **Splitting Strategy**: 
    - Total: ~52k samples.
    - Selected Subset: 14,000 samples total.
    - Split: 10,000 (Train), 2,000 (Validation), 2,000 (Test).
    - **Why?**: The assignment explicitly requires this 5:1:1 ratio on a subset to ensure manageable training times while having enough data for evaluation.
    - **Reproducibility**: `SEED = 42` used for random sampling to ensure the exact same subset is selected every time.

### Model Configuration
- **Base Model**: `meta-llama/Llama-3.2-1B`.
- **LoRA (Low-Rank Adaptation)**:
    - **Rank (r)**: 16 (Hypothesis: Sufficient for learning instruction following without overfitting).
    - **Alpha**: 32 (Scaling factor, usually 2x rank).
    - **Target Modules**: `q_proj`, `v_proj` (Standard for Llama architectures to adapt attention mechanisms).
    - **Dropout**: 0.05 (To prevent overfitting).

### Training Hyperparameters
- **Epochs**: 3 (Standard for fine-tuning; enough to converge but prevents major catastrophic forgetting).
- **Batch Size**: 4 (Micro-batch) with Gradient Accumulation to simulate larger effective batch size if needed.
- **Learning Rate**: 2e-4 (Typical for LoRA).
- **Precision**: FP16 (Mixed precision) to reduce memory usage and speed up training on consumer GPUs.

## Challenges & Solutions
*(To be populated as we progress)*
- [ ] **Challenge**: 
    - **Solution**: 

## Parameter Explanations
- **Temperature (0.7)**: Controls randomness. Lower values (0.7) make the model more deterministic but still creative enough for instructions.
- **Top-p (0.9)**: Nucleus sampling. Restricts the token pool to the top 90% probability mass, preventing low-quality tail tokens.

## Part 1: Theoretical Questions Key Points

### 1. Masked Attention
- **Concept**: In causal language models (like GPT/Llama), the model must not see future tokens when predicting the current one.
- **Mechanism**: A mask matrix (usually upper triangular with -inf) is added to the attention scores before Softmax.
- **Effect**: Ensures $P(w_i | w_{<i})$ depends only on past tokens, preserving the autoregressive property.

### 2. Diffusion Noise Schedule
- **Concept**: Diffusion models learn to reverse a gradual noise addition process.
- **Schedule**: Defines how much noise $\beta_t$ is added at each timestep $t$.
- **Types**: Linear, Cosine, Sigmoid.
- **Impact**: A good schedule ensures the signal is destroyed gradually, allowing the model to learn the reverse mapping effectively at all noise levels.

### 3. RAG Retrieval Stages
- **Retrieval**: Fetching relevant documents from a vector database using semantic similarity (e.g., cosine similarity of embeddings).
- **Augmentation**: Inserting the retrieved context into the prompt alongside the user query.
### 3. RAG Retrieval Stages
- **Retrieval**: Fetching relevant documents from a vector database using semantic similarity (e.g., cosine similarity of embeddings).
- **Augmentation**: Inserting the retrieved context into the prompt alongside the user query.
- **Generation**: The LLM generates an answer based on the augmented prompt, grounding the response in the retrieved facts.

## Part 2: Implementation Analysis

### Training Results (Observed)
- **Final Training Loss**: ~1.38
- **Validation Loss**: ~1.40
- **Training Time**: 1047.86 seconds (~17.5 minutes)
- **Peak GPU Memory**: 8.19 GB
- **Hardware**: RTX 4090 (via Conda environment)

### Code Explanations

#### Novel Instructions (Inference)
The list of "Novel Instructions" in `inference.py` (e.g., Haikus, Riddles, Translations) are **Out-of-Distribution (OOD)** samples. 
- **Purpose**: To test the model's *generalization* capabilities. The model was trained on Alpaca (instruction-following data), but we want to see if it can handle *new* types of requests it hasn't explicitly seen, rather than just memorizing the training data.
- **Why these specific ones?**: They cover diverse tasks: Creativity (Haiku), Reasoning (Riddle), Knowledge (Tokyo guide), and Logic (Prime check).

#### Perplexity Calculation (Evaluation)
The Perplexity (PPL) logic in `evaluate.py` was updated to be **Conditional Perplexity**.
- **Standard PPL**: Calculates probability of the entire sequence (Instruction + Output).
- **Conditional PPL**: Calculates probability of the *Output* given the *Instruction*.
- **Implementation**: We feed the full sequence to the model but set the `labels` for the Instruction part to `-100`. This tells PyTorch's loss function to **ignore** the instruction tokens and only calculate loss (and thus perplexity) on the generated response. This gives a fairer measure of how well the model generates the *answer*.


