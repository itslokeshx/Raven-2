# build_dataset.py
# High-performance, streaming-safe 40GB dataset compilation script for RAVEN-2.
# Streams FineWeb-Edu, Wikipedia, Cosmopedia, and Gutenberg directly from HuggingFace to disk.
# Applies NFKC normalization, boilerplate cleaners, alphanumeric/digit filters, MD5-based deduplication,
# and streaming write buffers with Google Drive and resume support.

import os
import re
import sys
import time
import json
import shutil
import hashlib
import argparse
import unicodedata
from tqdm import tqdm

# Add workspace directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Ratios of target size per source as research-backed by SmolLM / FineWeb-Edu best mix
SOURCE_RATIOS = {
    "fineweb-edu": 0.50,   # 50% - Best high-quality educational web corpus
    "wikipedia": 0.25,     # 25% - Structured fact-based encyclopedic knowledge
    "cosmopedia": 0.15,    # 15% - High-quality synthetic textbooks
    "gutenberg": 0.10      # 10% - Long-form literary coherence
}

STATE_FILE = "checkpoints/build_dataset_state.json"

def clean_unicode(text):
    """Normalize text representation to standard unicode NFKC form.
    """
    return unicodedata.normalize("NFKC", text)

def clean_boilerplate(text):
    """Remove bare URLs and common web boilerplate patterns.
    """
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Common boilerplate phrases
    boilerplate_patterns = [
        r'click here', r'read more', r'terms of use', r'privacy policy',
        r'cookie policy', r'all rights reserved', r'designed by',
        r'subscribe to', r'share on facebook', r'share on twitter',
        r'follow us on', r'email this post', r'back to top'
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text

def normalize_whitespace(text):
    """Collapse consecutive spaces and uniform blank lines.
    """
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def is_high_quality(text):
    """Apply strict quality filters to discard garbage, short clips, digit-heavy spam, or code dumps.
    """
    # 1. Minimum 200 characters
    if len(text) < 200:
        return False
        
    # 2. Minimum 50 words
    words = text.split()
    if len(words) < 50:
        return False
        
    # 3. 70%+ alphanumeric and space character ratio (filters out binary, heavy code, formulas, base64)
    alpha_count = sum(c.isalpha() or c.isspace() for c in text)
    if alpha_count / len(text) < 0.70:
        return False
        
    # 4. No digit-heavy garbage (less than 10% digits)
    digit_count = sum(c.isdigit() for c in text)
    if digit_count / len(text) > 0.10:
        return False
        
    return True

def get_prefix_hash(text):
    """Deduplication: Compute MD5 hash of first 1000 characters.
    """
    prefix = text[:1000].strip()
    return hashlib.md5(prefix.encode("utf-8")).hexdigest()

def rebuild_dedup_set(filepath):
    """Resume helper: Reads existing dataset from disk and extracts hashes to prevent duplicates.
    Runs streamingly to prevent loading gigabytes into RAM.
    """
    dedup_set = set()
    if not os.path.exists(filepath):
        return dedup_set
        
    file_size_gb = os.path.getsize(filepath) / (1024**3)
    print(f"🔄 Resuming build: Rebuilding deduplication database from existing corpus ({file_size_gb:.2f} GB)...")
    
    # 10MB chunk reads
    chunk_size = 10 * 1024 * 1024
    buffer = ""
    
    # Visual loading bar
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f, tqdm(
        total=os.path.getsize(filepath), unit='B', unit_scale=True, desc="Rebuilding database"
    ) as pbar:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            pbar.update(len(chunk.encode("utf-8")))
            buffer += chunk
            docs = buffer.split("<|eos|>")
            # The last element might be a cut-off document, hold it in buffer
            buffer = docs.pop()
            
            for doc in docs:
                clean_doc = doc.strip()
                if clean_doc:
                    dedup_set.add(get_prefix_hash(clean_doc))
                    
    print(f"✅ Deduplication database restored: {len(dedup_set):,} unique documents indexed.")
    return dedup_set

def backup_to_drive(filepath):
    """Google Colab helper: Automatic backup to Google Drive.
    """
    if os.path.exists("/content/drive"):
        try:
            drive_dest_dir = "/content/drive/MyDrive/Raven-2/data"
            os.makedirs(drive_dest_dir, exist_ok=True)
            drive_dest_path = os.path.join(drive_dest_dir, os.path.basename(filepath))
            print(f"\n☁️ Synchronizing dataset to Google Drive: {drive_dest_path}...")
            shutil.copy(filepath, drive_dest_path)
            print("✅ Backup complete.")
        except Exception as e:
            print(f"\n⚠️ Drive synchronization failed: {e}")

def get_dataset_stream(source):
    """Resilient, streaming loaders for all four designated data subsets.
    """
    from datasets import load_dataset
    
    if source == "fineweb-edu":
        # Stream high-quality educational web corpus subset
        return load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    elif source == "wikipedia":
        # Stream Wikipedia EN via the modern wikimedia/wikipedia dataset (Parquet format)
        return load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    elif source == "cosmopedia":
        # Stream Mixtral-generated textbook synthetic corpus
        return load_dataset("HuggingFaceTB/cosmopedia-v2", name="cosmopedia-v2", split="train", streaming=True)
    elif source == "gutenberg":
        # Stream Project Gutenberg books via the modern sedthh/gutenberg_english dataset (Parquet format)
        return load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
    else:
        raise ValueError(f"Unknown dataset source: {source}")

def iterate_with_retry(stream, retries=5, delay=5):
    """Iterator wrapper that recovers from connection drops or HuggingFace API rate limits.
    """
    iterator = iter(stream)
    while True:
        try:
            item = next(iterator)
            yield item
        except StopIteration:
            break
        except Exception as e:
            if retries <= 0:
                raise e
            print(f"\n⚠️ HF Connection drop: {e}. Re-connecting in {delay}s... ({retries} retries left)")
            time.sleep(delay)
            retries -= 1

def validate_dataset(filepath):
    """Validates the compiled binary-text corpus and prints standard quality control stats.
    """
    if not os.path.exists(filepath):
        print(f"❌ Corpus file not found at: {filepath}")
        return
        
    size_gb = os.path.getsize(filepath) / (1024**3)
    print(f"\n🔬 Performing validation scans on {filepath} ({size_gb:.2f} GB)...")
    
    total_docs = 0
    total_chars = 0
    min_len = float("inf")
    max_len = 0
    words_total = 0
    
    # 10MB streaming blocks
    chunk_size = 10 * 1024 * 1024
    buffer = ""
    
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f, tqdm(
        total=os.path.getsize(filepath), unit='B', unit_scale=True, desc="Analyzing documents"
    ) as pbar:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            pbar.update(len(chunk.encode("utf-8")))
            buffer += chunk
            docs = buffer.split("<|eos|>")
            buffer = docs.pop()
            
            for doc in docs:
                clean_doc = doc.strip()
                if clean_doc:
                    total_docs += 1
                    doc_len = len(clean_doc)
                    total_chars += doc_len
                    min_len = min(min_len, doc_len)
                    max_len = max(max_len, doc_len)
                    words_total += len(clean_doc.split())
                    
    avg_len = total_chars / total_docs if total_docs > 0 else 0
    avg_words = words_total / total_docs if total_docs > 0 else 0
    est_tokens = total_chars / 4.0 # approximate estimate
    
    print("\n" + "=" * 55)
    print(" 📊 RAVEN-2 CORPUS VALIDATION REPORT")
    print("=" * 55)
    print(f"  Corpus File:        {filepath}")
    print(f"  Total Size:         {size_gb:.4f} GB")
    print(f"  Document Count:     {total_docs:,} articles/books")
    print(f"  Average Doc Length: {avg_len:.1f} characters")
    print(f"  Average Word Count: {avg_words:.1f} words")
    print(f"  Min Doc Length:     {min_len if min_len != float('inf') else 0} chars")
    print(f"  Max Doc Length:     {max_len} chars")
    print(f"  Estimated Tokens:   {est_tokens:,.0f} tokens")
    print("=" * 55)
    print("✅ Quality scans complete! Ready for tokenization.\n")

def combine_local_corpora(dest_filepath):
    """Scan and streamingly merge 4 local corpus files into dest_filepath.
    Applies unicode cleaning, quality filtering, and deduplication.
    """
    # 1. Look for candidates
    candidates = []
    search_dirs = [".", "data"]
    variations = [
        "corpus_{}.txt", "corpus{}.txt", "corpus {}.txt", "corpus-{}.txt"
    ]
    
    for i in range(1, 5):
        found = False
        for sdir in search_dirs:
            for var in variations:
                path = os.path.join(sdir, var.format(i))
                if os.path.exists(path):
                    candidates.append(path)
                    found = True
                    break
            if found:
                break
                
    if not candidates:
        # Fallback search for any corpus-like files in current directory or data
        print("⚠️ Specific corpus_1 to corpus_4 files not found. Searching for any corpus*.txt files...")
        for sdir in search_dirs:
            if not os.path.exists(sdir):
                continue
            for f in os.listdir(sdir):
                if f.startswith("corpus") and f.endswith(".txt") and f != "corpus.txt":
                    candidates.append(os.path.join(sdir, f))
                    
    if not candidates:
        print("❌ No local corpus files (corpus_1 to corpus_4 or corpus*.txt) found to combine.")
        print("Please place your files in the workspace (e.g. data/corpus_1.txt) and try again.")
        sys.exit(1)
        
    # Sort files naturally
    candidates = sorted(list(set(candidates)))
    
    print("\n" + "=" * 60)
    print(" 🛠️  RAVEN-2 CORPUS MERGER & COMPILER")
    print(f" Found {len(candidates)} files to combine:")
    for c in candidates:
        sz_mb = os.path.getsize(c) / (1024**2)
        print(f"  - {c} ({sz_mb:.2f} MB)")
    print(f" Destination: {dest_filepath}")
    print("=" * 60 + "\n")
    
    # Rebuild dedup set
    dedup_hashes = rebuild_dedup_set(dest_filepath)
    
    # Open destination file in append mode
    with open(dest_filepath, "a", encoding="utf-8") as out_file:
        write_buffer = []
        buffer_size_chars = 0
        LIMIT_BUFFER_BYTES = 8 * 1024 * 1024 # 8MB
        
        for filepath in candidates:
            print(f"\nMerging {filepath} streamingly...")
            total_size = os.path.getsize(filepath)
            
            # Read in 10MB blocks
            chunk_size = 10 * 1024 * 1024
            buffer = ""
            
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=f"Processing {os.path.basename(filepath)}"
            ) as pbar:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    pbar.update(len(chunk.encode("utf-8")))
                    buffer += chunk
                    
                    # Split by <|eos|> first
                    if "<|eos|>" in buffer:
                        docs = buffer.split("<|eos|>")
                        buffer = docs.pop()
                    else:
                        # Fallback if no <|eos|>: split by double newlines (paragraphs)
                        # but keep last paragraph in buffer to avoid cut-off
                        docs = buffer.split("\n\n")
                        buffer = docs.pop()
                        
                    for doc in docs:
                        clean_doc = clean_unicode(doc)
                        clean_doc = clean_boilerplate(clean_doc)
                        clean_doc = normalize_whitespace(clean_doc)
                        
                        if not is_high_quality(clean_doc):
                            continue
                            
                        doc_hash = get_prefix_hash(clean_doc)
                        if doc_hash in dedup_hashes:
                            continue
                            
                        dedup_hashes.add(doc_hash)
                        
                        formatted_doc = f"{clean_doc}<|eos|>\n\n"
                        write_buffer.append(formatted_doc)
                        buffer_size_chars += len(formatted_doc)
                        
                        if buffer_size_chars >= LIMIT_BUFFER_BYTES:
                            out_file.write("".join(write_buffer))
                            out_file.flush()
                            os.fsync(out_file.fileno())
                            write_buffer = []
                            buffer_size_chars = 0
                            
                # Process remaining buffer content as the final document
                if buffer.strip():
                    clean_doc = clean_unicode(buffer)
                    clean_doc = clean_boilerplate(clean_doc)
                    clean_doc = normalize_whitespace(clean_doc)
                    if is_high_quality(clean_doc):
                        doc_hash = get_prefix_hash(clean_doc)
                        if doc_hash not in dedup_hashes:
                            dedup_hashes.add(doc_hash)
                            formatted_doc = f"{clean_doc}<|eos|>\n\n"
                            write_buffer.append(formatted_doc)
                            
            # Flush at the end of each file
            if write_buffer:
                out_file.write("".join(write_buffer))
                out_file.flush()
                os.fsync(out_file.fileno())
                write_buffer = []
                buffer_size_chars = 0
                
            print(f"✅ Finished merging {filepath}")
            
    print("\n🎉 LOCAL CORPORA COMBINATION COMPLETED SUCCESSFULLY!")
    validate_dataset(dest_filepath)

def main():
    parser = argparse.ArgumentParser(description="RAVEN-2 40GB High-Intelligence Dataset Compiler")
    parser.add_argument("--target_gb", type=float, default=40.0, help="Target total corpus size in GB (default: 40.0)")
    parser.add_argument("--source", type=str, default="all", choices=["all", "fineweb-edu", "wikipedia", "cosmopedia", "gutenberg"],
                        help="Train single source or download all combined (default: all)")
    parser.add_argument("--validate", action="store_true", help="Scan existing data/corpus.txt and output detailed stats")
    parser.add_argument("--combine", action="store_true", help="Merge local files corpus_1.txt to corpus_4.txt streamingly with cleaning")
    args = parser.parse_args()
    
    corpus_path = "data/corpus.txt"
    
    if args.validate:
        validate_dataset(corpus_path)
        return
        
    if args.combine:
        combine_local_corpora(corpus_path)
        return
        
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. Establish state and resume history
    state = {"target_gb": args.target_gb, "completed_sources": []}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as sf:
                saved_state = json.load(sf)
                if saved_state.get("target_gb") == args.target_gb:
                    state = saved_state
                    print(f"🔄 Recovered pipeline state. Completed sources: {state['completed_sources']}")
        except Exception:
            pass
            
    # Calculate target capacities for active download sources
    target_bytes = args.target_gb * (1024**3)
    
    active_sources = SOURCE_RATIOS.keys() if args.source == "all" else [args.source]
    
    # Scale targets if running a single source
    source_targets = {}
    if args.source == "all":
        for src, ratio in SOURCE_RATIOS.items():
            source_targets[src] = int(target_bytes * ratio)
    else:
        source_targets[args.source] = int(target_bytes)
        
    # 2. Rebuild deduplication index streamingly
    dedup_hashes = rebuild_dedup_set(corpus_path)
    
    print("\n" + "=" * 60)
    print(f" 🚀 RAVEN-2 CORPUS SYSTEM INITIALIZED")
    print(f"  Target File:      {corpus_path}")
    print(f"  Target Capacity:  {args.target_gb:.2f} GB")
    for src in active_sources:
        status = "COMPLETED" if src in state["completed_sources"] else "PENDING"
        print(f"  - {src:12s}: {source_targets[src]/(1024**3):.2f} GB ({status})")
    print("=" * 60 + "\n")
    
    # Open dataset file in append mode to support resumption
    with open(corpus_path, "a", encoding="utf-8") as out_file:
        write_buffer = []
        buffer_size_chars = 0
        LIMIT_BUFFER_BYTES = 8 * 1024 * 1024 # 8 MB write block buffer
        
        for source in active_sources:
            if source in state["completed_sources"]:
                print(f"✨ Skipping already completed source: {source}")
                continue
                
            print(f"📥 Loading stream for source: {source.upper()}...")
            stream = get_dataset_stream(source)
            
            source_written_bytes = 0
            source_target_bytes = source_targets[source]
            
            pbar = tqdm(total=source_target_bytes, unit='B', unit_scale=True, desc=f"Streaming {source}")
            
            # Streaming loop
            for doc in iterate_with_retry(stream):
                if source_written_bytes >= source_target_bytes:
                    break
                    
                # Extract raw text depending on schema standard
                if source == "gutenberg":
                    raw_text = doc.get("text", "")
                elif source == "wikipedia":
                    raw_text = doc.get("text", "")
                else:
                    raw_text = doc.get("text", "")
                    
                if not raw_text:
                    continue
                    
                # Clean pipelines
                text = clean_unicode(raw_text)
                text = clean_boilerplate(text)
                text = normalize_whitespace(text)
                
                # Check quality bar
                if not is_high_quality(text):
                    continue
                    
                # Deduplication check
                doc_hash = get_prefix_hash(text)
                if doc_hash in dedup_hashes:
                    continue
                    
                dedup_hashes.add(doc_hash)
                
                # Format document block
                formatted_doc = f"{text}<|eos|>\n\n"
                formatted_bytes = len(formatted_doc.encode("utf-8"))
                
                write_buffer.append(formatted_doc)
                buffer_size_chars += len(formatted_doc)
                
                # Update counters
                source_written_bytes += formatted_bytes
                pbar.update(formatted_bytes)
                
                # If write buffer meets capacity limit, flush directly to disk
                if buffer_size_chars >= LIMIT_BUFFER_BYTES:
                    out_file.write("".join(write_buffer))
                    out_file.flush()
                    os.fsync(out_file.fileno()) # Enforce actual disk flush
                    
                    write_buffer = []
                    buffer_size_chars = 0
                    
            pbar.close()
            
            # Flush remainder write buffer blocks
            if write_buffer:
                out_file.write("".join(write_buffer))
                out_file.flush()
                os.fsync(out_file.fileno())
                write_buffer = []
                buffer_size_chars = 0
                
            print(f"✅ Source {source.upper()} successfully compiled to disk.")
            
            # Mark source as completed in pipeline state
            state["completed_sources"].append(source)
            with open(STATE_FILE, "w") as sf:
                json.dump(state, sf, indent=2)
                
            # Perform automatic Drive backup sync
            backup_to_drive(corpus_path)
            
    print("\n🎉 RAVEN-2 CORPUS GENERATION COMPLETED SUCCESSFULLY!")
    validate_dataset(corpus_path)

if __name__ == "__main__":
    main()
