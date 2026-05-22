# config.py
# Central configuration for RAVEN-2 (100M parameter LLM)
# All hyperparameters are defined here. No hardcoded values in other files.

import os
import torch

# ── Model Architecture ──
vocab_size    = 16384       # 16K BPE tokenizer
block_size    = 1024        # Context window (tokens)
n_embd        = 768         # Embedding dimension (GPT-2 small scale)
n_head        = 12          # Query attention heads (64 dim per head)
n_kv_head     = 4           # Key/Value heads (Grouped Query Attention, GQA)
n_layer       = 12          # Transformer blocks (layers)
ffn_mult      = 4.5         # FFN hidden multiplier (used for SwiGLU: hidden_dim = round(n_embd * ffn_mult * 2 / 3 / 256) * 256)
dropout       = 0.1         # Dropout rate
weight_tying  = True        # Tie embedding and lm_head weights
bias          = False       # No bias in Linear/RMSNorm layers

# ── RoPE ──
rope_base     = 10000.0     # Rotary Position Embedding theta base

# ── Special Tokens ──
pad_id   = 0                # <|pad|> BPE special token ID
user_id  = 1                # <|user|> BPE special token ID
raven_id = 2                # <|raven|> BPE special token ID
eos_id   = 3                # <|eos|> BPE special token ID
sys_id   = 4                # <|system|> BPE special token ID

special_tokens = ["<|pad|>", "<|user|>", "<|raven|>", "<|eos|>", "<|system|>"]

# ── Pre-training ──
batch_size                  = 16      # Micro-batch size per GPU step
gradient_accumulation_steps = 4       # Effective batch size = batch_size * gradient_accumulation_steps = 64
max_iters                   = 50000   # Total training iterations
learning_rate               = 3e-4    # Peak learning rate
min_lr                      = 3e-5    # Cosine learning rate floor
warmup_iters                = 1000    # Linear warmup steps
weight_decay                = 0.1     # AdamW weight decay
grad_clip                   = 1.0     # Gradient clipping max norm
eval_interval               = 500     # Evaluate validation loss every N steps
eval_iters                  = 100     # Number of batches for validation estimation
use_amp                     = True    # Automatic Mixed Precision (float16)
gradient_checkpointing      = True    # Recompute activations on backward pass to save VRAM

# ── Instruction Tuning ──
instruct_max_iters          = 5000    # Total instruction tuning iterations
instruct_lr                 = 1e-4    # Peak learning rate
instruct_min_lr             = 1e-5    # Cosine learning rate floor for instruction tuning
instruct_warmup             = 200     # Warmup steps

# ── Personality Post-training ──
posttrain_max_iters         = 2000    # Total personality fine-tuning iterations
posttrain_lr                = 5e-5    # Peak learning rate
posttrain_min_lr            = 5e-6    # Cosine learning rate floor for personality fine-tuning
posttrain_warmup            = 100     # Warmup steps

# ── Paths ──
# Local paths inside workspace
tokenizer_path  = "checkpoints/raven2_tokenizer.json"
pretrain_ckpt   = "checkpoints/raven2_pretrain_best.pt"
instruct_ckpt   = "checkpoints/raven2_instruct_best.pt"
posttrain_ckpt  = "checkpoints/raven2_posttrain_best.pt"
train_bin       = "checkpoints/train.bin"
val_bin         = "checkpoints/val.bin"
instruct_bin    = "checkpoints/instruct.bin"
posttrain_bin   = "checkpoints/posttrain.bin"

# Google Drive sync path inside Colab
drive_backup_dir = "/content/drive/MyDrive/Raven2/checkpoints"

# ── Device & Dtype ──
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use float16 for CUDA by default to speed up and save VRAM, fallback to float32 on CPU
dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
dtype_str = "float16" if dtype == torch.float16 else "float32"

# Ensure checkpoints directory exists locally
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("eval", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)
