#!/bin/bash
set -e  # Exit on error

echo "🚀 [Start] Initializing Vessl AI Environment..."

# 0. Load .env file if exists
if [ -f .env ]; then
  echo "📄 Loading environment variables from .env..."
  export $(grep -v '^#' .env | xargs)
fi

# 1. Install Dependencies
# Vessl images usually have PyTorch, but we need specific libs
echo "📦 Installing Requirements..."
pip install --upgrade pip
pip install transformers datasets tokenizers wandb scipy sentencepiece ninja packaging wheel hf_transfer
# Install Transformer Engine if available (usually pre-installed in NVIDIA images, but just in case)
# pip install transformer-engine  <-- skipping to avoid compile time if not pre-built

# 2. Login to WandB (if key provided in env)
if [ -n "$WANDB_API_KEY" ]; then
    echo "🔑 Logging into WandB..."
    wandb login $WANDB_API_KEY
else
    echo "⚠️ WANDB_API_KEY not found. WandB runs will be offline or fail."
fi

# 3. Download Datasets
echo "⬇️  Downloading Datasets..."
python experiments/h100_moe_training/download_datasets.py

# 4. Train Tokenizer
echo "🔡 Training Custom Tokenizer..."
python experiments/h100_moe_training/train_tokenizer.py

# 5. Run Training
echo "🔥 Starting H100 Training..."
# Adjust batch size/workers based on actual hardware if needed via args
python experiments/h100_moe_training/train_h100_moe.py

echo "✅ [Done] All tasks completed successfully!"
