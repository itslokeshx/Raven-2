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
    Uses trust_remote_code for maximum compatibility with HuggingFace datasets.
    """
    from datasets import load_dataset
    
    kwargs = {"streaming": True, "trust_remote_code": True}
    
    if source == "fineweb-edu":
        # Stream high-quality educational web corpus subset
        print("  → Connecting to HuggingFace CDN for fineweb-edu (sample-10BT)...")
        return load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", **kwargs)
    elif source == "wikipedia":
        # Stream Wikipedia EN via the modern wikimedia/wikipedia dataset (Parquet format)
        print("  → Connecting to HuggingFace CDN for Wikipedia EN...")
        return load_dataset("wikimedia/wikipedia", "20231101.en", split="train", **kwargs)
    elif source == "cosmopedia":
        # Stream Mixtral-generated textbook synthetic corpus
        print("  → Connecting to HuggingFace CDN for Cosmopedia v2...")
        return load_dataset("HuggingFaceTB/cosmopedia-v2", name="cosmopedia-v2", split="train", **kwargs)
    elif source == "gutenberg":
        # Stream Project Gutenberg books via the modern sedthh/gutenberg_english dataset (Parquet format)
        print("  → Connecting to HuggingFace CDN for Project Gutenberg...")
        return load_dataset("sedthh/gutenberg_english", split="train", **kwargs)
    else:
        raise ValueError(f"Unknown dataset source: {source}")

def get_fast_dataset(source, target_gb):
    """High-speed parallel downloader for Colab multi-gigabit network.
    Queries the dataset repository, calculates the precise number of parquet files
    needed, and downloads them in parallel using multi-threaded downloads.
    """
    from huggingface_hub import HfApi
    from datasets import load_dataset
    import math
    
    api = HfApi()
    
    # Map sources to repo IDs, file folders, and average uncompressed text yields per file
    if source == "fineweb-edu":
        repo_id = "HuggingFaceFW/fineweb-edu"
        pattern = "sample/10BT/"
        name = "sample-10BT"
        mb_per_file = 42 # Average text yield per parquet file
    elif source == "wikipedia":
        repo_id = "wikimedia/wikipedia"
        pattern = "20231101.en/"
        name = "20231101.en"
        mb_per_file = 22
    elif source == "cosmopedia":
        repo_id = "HuggingFaceTB/cosmopedia-v2"
        pattern = "data/"
        name = "cosmopedia-v2"
        mb_per_file = 32
    elif source == "gutenberg":
        repo_id = "sedthh/gutenberg_english"
        pattern = "data/"
        name = "default"
        mb_per_file = 48
    else:
        raise ValueError(f"Unknown source: {source}")
        
    print(f"\n🔍 Resolving repository files for {source.upper()} from HuggingFace Hub...")
    try:
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        # Filter for parquet files matching the pattern
        parquet_files = [f for f in all_files if pattern in f and f.endswith(".parquet")]
        parquet_files.sort()
        
        if not parquet_files:
            print(f"⚠️ No matching files found under pattern '{pattern}'. Reverting to stream.")
            return None
            
        # Calculate files needed with 35% safety margin to guarantee target size is met
        target_mb = target_gb * 1024 * 1.35
        num_files_needed = max(1, math.ceil(target_mb / mb_per_file))
        selected_files = parquet_files[:num_files_needed]
        
        print(f"🚀 Selected {len(selected_files)}/{len(parquet_files)} parquet shards (~{len(selected_files)*mb_per_file:.1f} MB text equivalent).")
        print(f"📥 Initiating multi-threaded parallel download via HuggingFace parallel backend...")
        
        # This will download the files locally at max speed using Hugging Face's optimized parallel downloader
        dataset = load_dataset(repo_id, name=name, data_files=selected_files, split="train", trust_remote_code=True)
        print(f"✅ Local download complete! Dataset initialized with {len(dataset):,} rows.")
        return dataset
    except Exception as e:
        print(f"⚠️ Parallel download failed: {e}")
        print("🔄 Falling back to streaming mode...")
        return None

def iterate_with_retry(source_name, stream_factory, max_retries=10, delay=10):
    """Iterator wrapper that recovers from connection drops by re-creating the stream.
    stream_factory is a callable that returns a fresh dataset stream.
    """
    retries = 0
    skip_count = 0  # Track how many items to skip on reconnect
    
    while retries <= max_retries:
        try:
            stream = stream_factory()
            iterator = iter(stream)
            
            # Skip already-yielded items on reconnect
            for _ in range(skip_count):
                next(iterator)
            
            for item in iterator:
                skip_count += 1
                yield item
            
            # Clean exit — dataset exhausted
            return
            
        except StopIteration:
            return
        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"\n❌ Max retries ({max_retries}) exceeded. Last error: {e}")
                raise e
            print(f"\n⚠️ HF Connection drop on {source_name}: {e}")
            print(f"   Re-connecting in {delay}s... (retry {retries}/{max_retries}, will skip {skip_count} already-processed items)")
            time.sleep(delay)

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
    
    # Colab High-Speed Local SSD Cache Detection
    is_colab = os.path.exists("/content")
    active_corpus_path = corpus_path
    
    if is_colab:
        active_corpus_path = "/content/corpus_local.txt"
        if os.path.exists(corpus_path) and not os.path.exists(active_corpus_path):
            print(f"🔄 Colab detected. Restoring existing corpus from Google Drive to local SSD for fast streaming...")
            try:
                shutil.copy(corpus_path, active_corpus_path)
                print("✅ Restoration complete.")
            except Exception as e:
                print(f"⚠️ Restoration failed: {e}")
    
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
        
    # 2. Rebuild deduplication index streamingly on active local path
    dedup_hashes = rebuild_dedup_set(active_corpus_path)
    
    print("\n" + "=" * 60)
    print(f" 🚀 RAVEN-2 CORPUS SYSTEM INITIALIZED")
    print(f"  Target File (Active): {active_corpus_path}")
    print(f"  Target File (Drive):  {corpus_path}")
    print(f"  Target Capacity:      {args.target_gb:.2f} GB")
    for src in active_sources:
        status = "COMPLETED" if src in state["completed_sources"] else "PENDING"
        print(f"  - {src:12s}: {source_targets[src]/(1024**3):.2f} GB ({status})")
    print("=" * 60 + "\n")
    
    # Open active dataset file in append mode to support resumption
    with open(active_corpus_path, "a", encoding="utf-8") as out_file:
        write_buffer = []
        buffer_size_chars = 0
        LIMIT_BUFFER_BYTES = 8 * 1024 * 1024 # 8 MB write block buffer
        
        for source in active_sources:
            if source in state["completed_sources"]:
                print(f"✨ Skipping already completed source: {source}")
                continue
                
            source_target_bytes = source_targets[source]
            source_gb = source_target_bytes / (1024**3)
            
            # Try to get the high-speed local dataset first (resolves and downloads shards in parallel)
            dataset = get_fast_dataset(source, source_gb)
            
            source_written_bytes = 0
            
            # Live activity counters
            docs_scanned = 0
            docs_filtered = 0
            docs_deduped = 0
            docs_accepted = 0
            
            pbar = tqdm(total=source_target_bytes, unit='B', unit_scale=True, desc=f"Compiling {source}")
            
            # Use local fast iterator if available, otherwise fall back to resilient stream
            if dataset is not None:
                print(f"  ⚡ Local high-speed iteration enabled. Processing {len(dataset):,} rows...")
                iterator = dataset
            else:
                print(f"  📥 Loading stream for source: {source.upper()}...")
                print(f"  ⏳ Waiting for first documents from HuggingFace CDN (Streaming)...")
                def make_stream(src=source):
                    return get_dataset_stream(src)
                iterator = iterate_with_retry(source, make_stream)
            
            # Unified processing loop (runs at lightspeed for local iteration)
            for doc in iterator:
                if source_written_bytes >= source_target_bytes:
                    break
                
                docs_scanned += 1
                
                # Heartbeat: show activity every 1000 docs scanned even if none accepted yet
                if docs_scanned % 1000 == 0 and docs_accepted == 0:
                    print(f"  🔍 Scanned {docs_scanned:,} docs | Filtered: {docs_filtered:,} | Deduped: {docs_deduped:,} | Accepted: {docs_accepted:,}")
                elif docs_scanned % 50000 == 0:
                    print(f"  📊 Scanned {docs_scanned:,} | Accepted: {docs_accepted:,} | Filtered: {docs_filtered:,} | Deduped: {docs_deduped:,} | Written: {source_written_bytes/(1024**3):.2f}GB")
                    
                # Extract raw text
                raw_text = doc.get("text", "")
                    
                if not raw_text:
                    docs_filtered += 1
                    continue
                    
                # Clean pipelines
                text = clean_unicode(raw_text)
                text = clean_boilerplate(text)
                text = normalize_whitespace(text)
                
                # Check quality bar
                if not is_high_quality(text):
                    docs_filtered += 1
                    continue
                    
                # Deduplication check
                doc_hash = get_prefix_hash(text)
                if doc_hash in dedup_hashes:
                    docs_deduped += 1
                    continue
                    
                dedup_hashes.add(doc_hash)
                docs_accepted += 1
                
                # Print confirmation when first doc is accepted
                if docs_accepted == 1:
                    print(f"  ✅ First document accepted! (scanned {docs_scanned:,} docs to get here). Streaming is LIVE!")
                
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
            print(f"  📊 Final: Scanned {docs_scanned:,} | Accepted {docs_accepted:,} | Filtered {docs_filtered:,} | Deduped {docs_deduped:,}")
            
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
                
            # Perform automatic Drive backup sync of completed source chunk
            if is_colab:
                try:
                    print(f"\n☁️ Syncing completed progress to Google Drive ({corpus_path})...")
                    shutil.copy(active_corpus_path, corpus_path)
                    print("✅ Backup complete.")
                except Exception as e:
                    print(f"\n⚠️ Drive synchronization failed: {e}")
            else:
                backup_to_drive(corpus_path)
            
    print("\n🎉 RAVEN-2 CORPUS GENERATION COMPLETED SUCCESSFULLY!")
    validate_dataset(active_corpus_path)

if __name__ == "__main__":
    main()
