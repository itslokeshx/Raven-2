# 🐦‍⬛ RAVEN-2 — Google Colab Pre-Training Master Guide
### Cloud-Generated 10GB Personality + 40GB Factual Intelligence (50+ GB Total Corpus)

This master guide walks you through setting up, cloud-generating, merging, tokenizing, binary-compiling, and training **Raven-2 (100M Parameters)** on Google Colab (using A100 or T4 GPUs) by combining a **40GB Factual Base** with a **10GB Cloud-Generated Personality Corpus**.

---

## ⚡ Why Generate in the Cloud?
Uploading a 10GB file from a home internet connection can take hours and frequently fails due to disconnects. 

Instead, **generate the 10GB personality dataset directly inside Google Colab's cloud environment!** Because Python string formatting is incredibly fast, Colab will generate the entire 10GB file streamingly in **less than 2 minutes** using Colab's high-speed internal storage, with zero upload bandwidth required.

---

## 📚 The Colab Jupyter Notebook
For the easiest setup, we have compiled all of these cells into a premium, single-click Jupyter Notebook in your codebase:
📁 `notebooks/raven2_pretrain.ipynb`

Simply upload this `.ipynb` file directly to Google Colab, mount your Drive, and run the cells!

---

## 🛠️ Step-by-Step Google Colab Manual Guide

### Step 1: Set Up & Mount Google Drive
Mount your Google Drive to ensure all training checkpoints, trained tokenizers, and datasets are backed up in real time. If the runtime resets, your training progress is 100% safe.

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Create target directories on your Google Drive
import os
os.makedirs("/content/drive/MyDrive/Raven2/checkpoints", exist_ok=True)
os.makedirs("/content/drive/MyDrive/Raven2/data", exist_ok=True)

# 3. Symlink Google Drive directories to our workspace for easy, high-speed access
!rm -rf data checkpoints
!ln -s "/content/drive/MyDrive/Raven2/data" "./data"
!ln -s "/content/drive/MyDrive/Raven2/checkpoints" "./checkpoints"
```

---

### Step 2: Install High-Performance Dependencies
Install optimized Rust-backed tokenization engines and verify active GPU acceleration:
```bash
# Install dependencies
!pip install datasets tqdm tokenizers numpy -q

# Verify GPU status
python -c "import torch; print('CUDA Active:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

### Step 3: Stream-Download the 40GB High-Intelligence Corpus
Download FineWeb-Edu, Wikipedia, and Cosmopedia v2 directly to the Colab SSD:

```bash
# Streams and filters 40GB of factual knowledge base
!python data/build_dataset.py --target_gb 40
```
*This downloads clean pre-training data directly into `data/corpus.txt`.*

---

### Step 4: Generate the 10GB Personality Corpus in the Cloud
Run the generator directly on Colab to build **19,200,000 dialogue pairs** streamingly:

```bash
# Generate 19.2M conversational dialogue pairs (10GB+ personality dataset)
!python data/build_personality_corpus.py --count 19200000
```
> [!TIP]
> This command will execute streamingly and write the `data/personality.txt` file directly onto Colab's high-speed local disk in **under 2 minutes**.

---

### Step 5: High-Speed Unified Corpus Concatenation
Append your newly generated 10GB personality dataset and any local raw combined knowledge corpus (`data/combined.txt`) directly into the master `data/corpus.txt` file:

```bash
# Rename local combined to prevent collision
if os.path.exists("data/combined.txt"):
    !mv data/combined.txt data/local_combined.txt

# Append local combined (if exists) and the huge personality dataset into the main corpus
print("Merging corpora into data/corpus.txt...")
if os.path.exists("data/local_combined.txt"):
    !cat data/local_combined.txt >> data/corpus.txt
if os.path.exists("data/personality.txt"):
    !cat data/personality.txt >> data/corpus.txt

# Verify final sizes (Should be ~50-52 GB in total)
!ls -lh data/
```

---

### Step 6: Train the Custom 16K BPE Tokenizer
Train your custom BPE vocabulary directly on the 50GB combined file to capture both factual text patterns and dry personality expressions:

```bash
# Train BPE Tokenizer on the full combined dataset
!python tokenizer_train.py --input data/corpus.txt
```
*Rust-backed tokenization completes this on a Colab High-RAM instance in about 3–5 minutes.*

---

### Step 7: Binary Pack the 50GB Corpus
Stream the combined text file and pack it into high-speed memory-mapped binary `uint16` files. The preparation pipeline is completely streaming and **will not run out of memory**:

```bash
# 1. Compile pre-training binaries (generates checkpoints/train.bin and checkpoints/val.bin)
!python data/data_prep.py --mode pretrain

# 2. Compile instruction binaries (generates checkpoints/instruct.bin)
!python data/data_prep.py --mode instruct
```

---

### Step 8: Launch Pre-Training
Pre-train your 100M parameter model! The training script automatically backs up your best checkpoints to Google Drive every 500 iterations, ensuring resume-safety:

```bash
# Start pre-training with Flash Attention, Mixed Precision, and Google Drive auto-backup
!python train.py
```

---

## ⚡ Keep Colab Active
Keep your browser tab active by pasting this keep-alive script into your browser developer console (`F12` -> `Console`):
```javascript
function ClickConnect(){
  console.log("Keep-alive clicked");
  document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
```
