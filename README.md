# 🐦‍⬛ RAVEN-2: 100M Parameter Production-Grade AI Agent

RAVEN-2 is a complete, production-grade 100M parameter language model codebase built entirely from scratch in PyTorch. It features modern LLM architecture enhancements (RMSNorm, RoPE, SwiGLU, Grouped Query Attention, Flash Attention, and dynamic KV Caching) optimized for single-GPU training and Google Colab environments.

---

## 🎛️ Architecture Specifications

The architecture of RAVEN-2 is mathematically designed to match modern state-of-the-art models like LLaMA and Mistral:

### 1. Root Mean Square Normalization (RMSNorm)
RMSNorm replaces standard LayerNorm by normalizing inputs by their root mean square without subtracting the mean, improving computing speed and stability:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
Where $\gamma$ is a learnable scaling parameter. No bias parameters are used.

### 2. Rotary Position Embeddings (RoPE)
Positions are encoded by rotating adjacent pairs of dimension components of the Query ($Q$) and Key ($K$) tensors:
$$\begin{pmatrix} rx_{2i-1} \\ rx_{2i} \end{pmatrix} = \begin{pmatrix} \cos \theta_i & -\sin \theta_i \\ \sin \theta_i & \cos \theta_i \end{pmatrix} \begin{pmatrix} x_{2i-1} \\ x_{2i} \end{pmatrix}$$
Where $\theta_i = \text{base}^{-2(i-1)/d} \cdot t$ represents the frequency scaling for position index $t$.

### 3. SwiGLU Activation Function
The feed-forward network employs SwiGLU projections, scaling the hidden dimension based on the LLaMA-2 formula:
$$\text{SwiGLU}(x) = \text{Swish}(x W_{gate}) \cdot (x W_{up})$$
$$\text{hidden\_dim} = \text{round}\left(n_{\text{embd}} \times f_{\text{mult}} \times \frac{2}{3 \times 256}\right) \times 256$$
For $n_{\text{embd}} = 768$ and $f_{\text{mult}} = 4.5$, the resulting feed-forward hidden dimension is exactly **2304**.

### 4. Grouped Query Attention (GQA)
Implements grouped attention to balance memory throughput and capability. With $n_{\text{head}} = 12$ and $n_{\text{kv\_head}} = 4$, every Key/Value head is shared across exactly 3 Query heads (GQA group size = 3). Includes full compatibility with PyTorch's native FlashAttention (`F.scaled_dot_product_attention`).

---

## 📁 Repository Structure

```
Raven-2/
├── config.py                     # All architectural specs, hyperparameters & token IDs
├── model.py                      # RMSNorm, RoPE, SwiGLU, GQA, KV Cache & Generation
├── requirements.txt              # Project third-party python dependencies
├── tokenizer_train.py            # 16K BPE Byte-Level Tokenizer training script
├── train.py                      # Pre-training loop (Cosine Warmup, AMP, Drive, Ctrl+C)
├── instruct_tune.py              # Instruction tuning (Loss masking on System/User tokens)
├── posttrain.py                  # Personality tuning (Raven sarcastic, dry, direct voice)
├── inference.py                  # Interactive CLI chat client (Modes, 5-turn history)
├── app.py                        # Premium, dark monochrome Gradio Web UI block
├── .gitignore                    # Version control ignore files
├── README.md                     # Comprehensive architecture manual (this file)
├── data/
│   ├── build_dataset.py          # 40GB high-intelligence HF streaming corpus compiler
│   ├── build_personality_corpus.py # Programmatic 50K personality training set generator
│   ├── build_instruct_data.py    # Programmatic 50K high-quality QA dataset generator
│   ├── download_data.py          # Factual data downloader with filtering & deduplication
│   ├── combine_to_1gb.py         # Shuffling & chunking raw corpus tool
│   └── data_prep.py              # Streaming binary data packer (uint16 memmap & cache)
├── eval/
│   ├── knowledge_bench.py        # 100 fact-based Wikipedia QA checks
│   ├── reasoning_bench.py        # 50 mathematical/logic/analogy checks
│   └── run_full_eval.py          # Complete evaluation portal (Scores & Coherence)
└── notebooks/
    ├── colab_pretrain.ipynb      # Automated pre-training pipeline notebook
    ├── colab_instruct.ipynb      # Automated instruction fine-tuning notebook
    └── colab_posttrain.ipynb     # Automated personality post-training notebook
```

---

## 🚀 Execution & Pipeline Guide

### 1. Install Environment Dependencies
Verify that you are using Python 3.10+ and install dependencies:
```bash
pip3 install -r requirements.txt
```

### 2. Download and Compile Raw Text Corpus
Run the pre-training corpus compiler to stream and build maximum intelligence:
```bash
python3 data/build_dataset.py --target_gb 2     # Quick test with 2GB subset
python3 data/build_dataset.py --target_gb 40    # Full 40GB SmolLM-mix build
```

### 3. Train the Byte-Level BPE Tokenizer
Train a vocabulary of 16,384 tokens on your compiled corpus text:
```bash
python3 tokenizer_train.py
```
*This step creates `checkpoints/raven2_tokenizer.json` and verifies that `<|pad|>` (0), `<|user|>` (1), `<|raven|>` (2), `<|eos|>` (3), and `<|system|>` (4) are locked in.*

### 4. Build the Instruction Fine-Tuning Dataset
Compile 50,000 highly structured system, user, and raven dialogue pairs:
```bash
python3 data/build_instruct_data.py --dry-run
python3 data/build_instruct_data.py
```

### 5. Pack Binary Memory-Mapped Files
Convert textual strings into zero-copy, highly efficient binary uint16 representations for model training:
```bash
python3 data/data_prep.py --mode pretrain      # Packs checkpoints/train.bin & val.bin
python3 data/data_prep.py --mode instruct      # Packs checkpoints/instruct.bin & instruct_val.bin
```

### 6. Phase 1: Pre-training
Train the model autoregressively on base factual knowledge:
```bash
python3 train.py --dry-run                # Train for 20 quick test steps
python3 train.py                          # Full pre-training execution
```
*Gradients accumulate automatically. Checks validation loss, backs up to Google Drive if active in Colab, and saves checkpoints cleanly to `checkpoints/raven2_pretrain_best.pt`.*

### 7. Phase 2: Instruction Fine-Tuning
Execute fine-tuning with custom target loss masking (model only learns to predict the assistant's tokens):
```bash
python3 instruct_tune.py                  # Instruction fine-tuning
```

### 8. Phase 3: Personality Post-Training
Bake in Raven's dry, direct, and slightly sarcastic tone:
```bash
python3 posttrain.py                      # Personality post-training
```

### 9. Validation & Benchmarks
Test model accuracy against all evaluation sets:
```bash
python3 eval/run_full_eval.py
```
*This prints a scoreboard containing factual accuracy (100 prompts), logical reasoning (50 prompts), and loops/coherence checks.*

### 10. Start the Chat Clients
Interact with your fully trained RAVEN-2 model via command-line or Gradio web interfaces:
```bash
# Command-Line Client
python3 inference.py

# Gradio Web Portal (Developer UI)
python3 app.py
```

---

## 🎨 Interactive CLI Client Commands

In `inference.py`, the following slash commands are active mid-dialogue:
* `/mode [name]` - Switches active sampling parameter sets (choices: `precise`, `sharp`, `default`, `chaos`, `cold`).
* `/modes` - Lists the active temperature, top-k, top-p, and repetition penalty metrics for each mode.
* `/system [X]` - Updates the guiding instructions defining Raven's personality on the fly.
* `/stats` - Displays tokens generated, hardware utilization, context lengths, and token/sec speeds.
* `/reset` - Clears the current dialogue window (up to 5 history turns).
* `/save` - Exports full conversation logs to a timestamped local text file.
* `/clear` - Wipes the active terminal screen.
* `/exit` - Safely closes the chat portal.

---

## ☁️ Google Colab Setup (A100 / T4)

To train or fine-tune this model on Google Colab, open the corresponding notebooks in `/notebooks/`. Ensure your runtime includes GPU hardware acceleration. The codebase auto-detects `/content/drive` and synchronizes checkpoints every 2000 steps for maximum safety against runtime disconnections.

---

## 🎖️ Production-Grade Implementation Details

1. **Parameter Constraints Assertion**: `model.py` performs a strict unique parameter calculation at initialization, ensuring `90M <= parameter count <= 115M` holds true (tied embedding parameters are counted only once).
2. **Epistemic Honesty (Refusal Dataset)**: Specifically contains 2,000 conversational pairs where Raven declines to speculate on real-time news or events outside its scope.
3. **Loss Masking Guard**: Fine-tuning labels are selectively set to `-100`, which completely blocks gradient backpropagation on system headers and user prompts, forcing the model to learn only how to answer correctly.
4. **Ctrl+C Interrupt Handler**: Pressing `Ctrl+C` triggers an active signal trap that immediately compiles and saves current model, optimizer, scaler, and history states to a checkpoint file, allowing you to stop training at any time without losing progress.
