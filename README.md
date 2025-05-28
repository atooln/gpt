# GPT From Scratch

This project is a Python-based implementation of a character-level Generative Pre-trained Transformer (GPT) model, built from scratch using PyTorch. It aims to replicate core GPT-2 functionalities for educational purposes.

## Key Features

- **Character-Level Tokenization**: Processes text at the character level.
- **Transformer Architecture**: Implements the core components of a Transformer model:
  - Scaled Dot-Product Attention (Single Head)
  - Multi-Head Attention
  - Position-wise Feed-Forward Networks
  - Transformer Blocks (with pre-LayerNorm and residual connections)
- **Positional Embeddings**: Adds positional information to token embeddings.
- **Autoregressive Generation**: Generates new text character by character based on the learned context.
- **Training and Evaluation**: Includes a training loop with loss calculation (cross-entropy) and periodic evaluation on a validation set.
- **Device Agnostic**: Runs on CPU or CUDA GPU if available.

## How it Works (`gpt.py`)

The script `gpt.py` can be broken down into the following main parts:

1.  **Setup & Configuration**:

    - Defines global hyperparameters (see [Hyperparameters](#hyperparameters) below).
    - Sets a manual seed for reproducibility.
    - Determines the computation device (`CPU` or `GPU`).

2.  **Data Processing**:

    - **Load Data**: Reads raw text from `input.txt`.
    - **Tokenize**: Creates a vocabulary of unique characters and provides `encode` (string to int list) and `decode` (int list to string) functions.
    - **Split Data**: Divides the tokenized data into training (90%) and validation (10%) sets.
    - **Batching (`get_batch` function)**: Generates random batches of input sequences (`x`) and target sequences (`y`) for training or validation. Each sequence has a length of `BLOCK_SIZE`.

3.  **Model Definition (PyTorch `nn.Module` classes)**:

    - **`Head`**: Implements a single attention head with causal masking (to prevent attending to future tokens).
    - **`MultHead`**: Combines multiple `Head` instances for multi-head attention, followed by a linear projection.
    - **`FeedForward`**: A standard two-layer MLP with ReLU activation, used within each Transformer block.
    - **`Block`**: A single Transformer decoder block. It includes:
      - Multi-head self-attention (with pre-LayerNorm).
      - A feed-forward network (with pre-LayerNorm).
      - Residual connections around both sub-layers.
    - **`LM` (Language Model)**: The main model class.
      - Initializes token and positional embedding tables.
      - Stacks `N_LAYERS` of `Block` modules.
      - Adds a final linear layer (`lm_head`) to produce logits over the vocabulary.
      - `forward(idx, targets)`: Computes token and position embeddings, passes them through the Transformer blocks, and calculates the cross-entropy loss if targets are provided.
      - `generate(idx, max_new_tokens)`: Autoregressively generates text by repeatedly predicting the next token, sampling from the output distribution, and appending it to the input sequence.

4.  **Training Process**:

    - **Loss Estimation (`estimate_loss` function)**: Calculates the average loss on the training and validation sets without gradient computation.
    - **Training Loop**:
      - Initializes the `LM` model and the AdamW optimizer.
      - Iterates for `MAX_ITERS` steps.
      - Periodically (every `EVAL_INTERVAL`), evaluates and prints training and validation losses.
      - In each step:
        1.  Fetches a training batch.
        2.  Performs a forward pass to get logits and loss.
        3.  Clears gradients.
        4.  Performs backpropagation.
        5.  Updates model parameters.

5.  **Text Generation (Example)**:
    - The `if __name__ == "__main__":` block demonstrates how to use the trained model.
    - It starts with an initial context (a single token) and calls `model.generate()` to produce a specified number of new tokens.
    - The generated token indices are decoded back to a string and printed.

## Hyperparameters

The following hyperparameters are defined at the beginning of `gpt.py` and control the model architecture and training process:

- `BATCH_SIZE = 64`: Number of independent sequences to process in parallel.
- `BLOCK_SIZE = 256`: Maximum context length for predictions.
- `MAX_ITERS = 5000`: Total number of training iterations.
- `EVAL_INTERVAL = 500`: How often to evaluate the model on the validation set.
- `LEARNING_RATE = 3e-4`: Learning rate for the AdamW optimizer.
- `EVAL_ITERS = 200`: Number of batches to use for loss estimation during evaluation.
- `N_EMBED = 128`: Dimension of token and positional embeddings.
- `N_LAYERS = 4`: Number of Transformer blocks in the model.
- `N_HEADS = 4`: Number of attention heads in each multi-head attention layer.
- `DROPOUT = 0.2`: Dropout probability used for regularization in various layers.

## Prerequisites

- Python 3.x
- PyTorch
- Matplotlib (currently imported but not used for core model functionality in the provided script snippet)

## Running the Script

1.  **Prepare Data**: Ensure you have a text file named `input.txt` in the same directory as `gpt.py`. This file will be used as the training corpus.
2.  **Install Dependencies**: If you don't have PyTorch, install it by following the instructions on the [official PyTorch website](https://pytorch.org/).
3.  **Execute**: Run the script from your terminal:
    ```bash
    python gpt.py
    ```
