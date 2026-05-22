# data_prep.py
# Multi-source streaming data preparation pipeline
# Processes raw text or JSONL instruction datasets into binary uint16 memmap files.
# Uses metadata caching to avoid redundant rebuilds.

import os
import sys
import json
import hashlib
import argparse
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer
import config

def get_file_metadata(filepath):
    """Get file size and modification time to check for changes.
    """
    if not os.path.exists(filepath):
        return None
    stats = os.stat(filepath)
    return {
        "size": stats.st_size,
        "mtime": stats.st_mtime
    }

def get_metadata_cache_path(mode):
    return f"checkpoints/{mode}_data_prep_cache.json"

def is_cache_valid(input_files, mode):
    cache_path = get_metadata_cache_path(mode)
    if not os.path.exists(cache_path):
        return False
        
    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)
            
        # Check if tokenizer hasn't changed
        tok_meta = get_file_metadata(config.tokenizer_path)
        if not tok_meta or cache.get("tokenizer") != tok_meta:
            return False
            
        # Check if all input files are unchanged
        cached_files = cache.get("files", {})
        if len(cached_files) != len(input_files):
            return False
            
        for filepath in input_files:
            meta = get_file_metadata(filepath)
            if not meta or cached_files.get(filepath) != meta:
                return False
                
        # Check if output files exist
        train_bin_path = config.train_bin if mode == "pretrain" else config.instruct_bin
        val_bin_path = config.val_bin if mode == "pretrain" else config.posttrain_bin # Fallback split paths
        if mode == "instruct":
            train_bin_path = config.instruct_bin
            val_bin_path = "checkpoints/instruct_val.bin"
            
        if not os.path.exists(train_bin_path) or not os.path.exists(val_bin_path):
            return False
            
        return True
    except Exception:
        return False

def save_metadata_cache(input_files, mode):
    cache_path = get_metadata_cache_path(mode)
    cache = {
        "tokenizer": get_file_metadata(config.tokenizer_path),
        "files": {filepath: get_file_metadata(filepath) for filepath in input_files}
    }
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

def stream_samples_from_file(filepath, mode):
    """Stream samples from file.
    In pretrain mode: reads paragraph blocks separated by double-newlines.
    In instruct mode: reads JSONL lines, formatting them as:
    <|system|>{system}<|user|>{user}<|raven|>{raven}<|eos|>
    """
    if mode == "pretrain":
        # Read in chunks of 8MB to stay memory-efficient
        chunk_size = 8 * 1024 * 1024
        leftover = ""
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                text = leftover + chunk
                # Split by paragraph block
                paragraphs = text.split("\n\n")
                # The last paragraph might be cut off, save as leftover
                leftover = paragraphs.pop()
                for p in paragraphs:
                    p = p.strip()
                    if p:
                        yield p
            if leftover.strip():
                yield leftover.strip()
    elif mode == "instruct":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    system = data.get("system", "").strip()
                    user = data.get("user", "").strip()
                    raven = data.get("raven", "").strip()
                    
                    # Format standard instruction format
                    sample = f"<|system|>{system}<|user|>{user}<|raven|>{raven}<|eos|>"
                    yield sample
                except Exception:
                    # If not JSONL, fallback to treating the line as raw text
                    yield line

def main():
    parser = argparse.ArgumentParser(description="Prepare RAVEN-2 binary dataset.")
    parser.add_argument("--mode", type=str, required=True, choices=["pretrain", "instruct"],
                        help="Data preparation mode: pretrain or instruct")
    parser.add_argument("--inputs", nargs="+", default=None,
                        help="Input file paths (default: data/corpus.txt for pretrain, data/instruct.txt for instruct)")
    args = parser.parse_args()
    
    # Set default inputs
    if args.inputs is None:
        if args.mode == "pretrain":
            args.inputs = ["data/corpus.txt"]
        else:
            args.inputs = ["data/instruct.txt"]
            
    # Verify inputs exist
    for filepath in args.inputs:
        if not os.path.exists(filepath):
            print(f"❌ Input file not found: {filepath}")
            if args.mode == "pretrain" and filepath == "data/corpus.txt":
                print("Tokenizer training script generated data/corpus.txt. Did you run it?")
            elif args.mode == "instruct" and filepath == "data/instruct.txt":
                print("Run: python data/build_instruct_data.py first to create data/instruct.txt")
            sys.exit(1)
            
    if not os.path.exists(config.tokenizer_path):
        print(f"❌ Tokenizer not found at: {config.tokenizer_path}. Run tokenizer_train.py first.")
        sys.exit(1)
        
    # Check cache
    if is_cache_valid(args.inputs, args.mode):
        print(f"✨ Metadata cache matches for mode={args.mode}. Skipping data preparation!")
        return
        
    print(f"🚀 Preparing binary dataset for mode={args.mode}...")
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    
    # Load all samples into memory (each sample is relatively short, so this is RAM-safe)
    samples = []
    for filepath in args.inputs:
        print(f"Reading samples from {filepath}...")
        for sample in stream_samples_from_file(filepath, args.mode):
            samples.append(sample)
            
    total_samples = len(samples)
    print(f"Loaded {total_samples:,} total samples.")
    
    # Shuffle samples
    np.random.seed(42)
    np.random.shuffle(samples)
    
    # Split into train (95%) and val (5%)
    split_idx = int(0.95 * total_samples)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    # Tokenize and pack into uint16 binary files
    train_bin_path = config.train_bin if args.mode == "pretrain" else config.instruct_bin
    # For instruct, validation path is distinct to prevent conflict with pre-training
    val_bin_path = config.val_bin if args.mode == "pretrain" else "checkpoints/instruct_val.bin"
    
    # Batch size for tokenization
    batch_size = 4096
    
    # Track statistics
    total_tokens = 0
    unique_tokens = set()
    sample_lengths = []
    eos_missing_count = 0
    
    for split_name, split_samples, bin_path in [("train", train_samples, train_bin_path), ("val", val_samples, val_bin_path)]:
        print(f"\nProcessing {split_name} split ({len(split_samples):,} samples) -> {bin_path}")
        
        # Open binary file for writing
        with open(bin_path, "wb") as f:
            for i in tqdm(range(0, len(split_samples), batch_size), desc=f"Tokenizing {split_name}"):
                batch = split_samples[i : i + batch_size]
                
                # Tokenize batch
                encodings = tokenizer.encode_batch(batch)
                
                # Write to binary
                packed_ids = []
                for enc in encodings:
                    ids = enc.ids
                    
                    # For pretrain mode, ensure it ends with <|eos|>
                    if args.mode == "pretrain" and (len(ids) == 0 or ids[-1] != config.eos_id):
                        ids.append(config.eos_id)
                        
                    # Statistics tracking
                    total_tokens += len(ids)
                    sample_lengths.append(len(ids))
                    unique_tokens.update(ids)
                    
                    # Verify eos token presence
                    if config.eos_id not in ids:
                        eos_missing_count += 1
                        
                    packed_ids.extend(ids)
                    
                # Write batch as uint16
                arr = np.array(packed_ids, dtype=np.uint16)
                f.write(arr.tobytes())
                
    # Report stats
    avg_len = sum(sample_lengths) / len(sample_lengths) if sample_lengths else 0
    print("\n" + "=" * 40)
    print(" DATA PIPELINE STATS & REPORT")
    print("=" * 40)
    print(f"Total tokens processed: {total_tokens:,}")
    print(f"Unique tokens seen:     {len(unique_tokens):,}")
    print(f"Average sample length:  {avg_len:.2f} tokens")
    
    eos_missing_pct = (eos_missing_count / total_samples) * 100 if total_samples > 0 else 0
    print(f"Samples missing <|eos|>: {eos_missing_count:,} ({eos_missing_pct:.2f}%)")
    
    if eos_missing_pct > 1.0:
        print("⚠️ WARNING: More than 1% of samples lack an <|eos|> token. Ensure formatting is correct!")
    else:
        print("✅ Success! <|eos|> token validation passed.")
    print("=" * 40)
    
    # Save metadata cache
    save_metadata_cache(args.inputs, args.mode)
    print("✨ Metadata cache saved successfully.")

if __name__ == "__main__":
    main()
