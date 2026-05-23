# tokenizer_train.py
# Trains a 16K byte-level BPE tokenizer on corpus text
# Ensures fixed special token IDs in the exact required order:
# <|pad|> -> 0, <|user|> -> 1, <|raven|> -> 2, <|eos|> -> 3, <|system|> -> 4

import os
import sys
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
import config

def train_tokenizer(input_file, output_path):
    print(f"Starting BPE tokenizer training on {input_file}...")
    
    # Initialize empty BPE model
    tokenizer = Tokenizer(BPE(unk_token=None))
    
    # Set pre-tokenizer to Byte-Level (pre-tokenizes into bytes to handle all unicode)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    # Set up BPE Trainer with target vocab size and special tokens
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=2,
        special_tokens=config.special_tokens
    )
    
    # Train from files
    tokenizer.train([input_file], trainer)
    
    # Set up post-processor to automatically handle formatting if needed
    # But usually we manually append special tokens in the datasets.
    # To keep tokenization clean, we just set the decoders to ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    tokenizer.decoder = ByteLevelDecoder()
    
    # Ensure correct parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the trained tokenizer
    tokenizer.save(output_path)
    print(f"Tokenizer trained successfully and saved to {output_path}")
    
    # Verification
    verify_tokenizer(output_path)

def verify_tokenizer(path):
    tokenizer = Tokenizer.from_file(path)
    print("\n" + "=" * 40)
    print(" TOKENIZER VERIFICATION")
    print("=" * 40)
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    
    # Verify special token IDs
    for token in config.special_tokens:
        token_id = tokenizer.token_to_id(token)
        print(f"Special token {token:10s} -> ID {token_id}")
        
    # Check that special token IDs are exactly 0-4
    assert tokenizer.token_to_id("<|pad|>") == config.pad_id, "pad ID mismatch"
    assert tokenizer.token_to_id("<|user|>") == config.user_id, "user ID mismatch"
    assert tokenizer.token_to_id("<|raven|>") == config.raven_id, "raven ID mismatch"
    assert tokenizer.token_to_id("<|eos|>") == config.eos_id, "eos ID mismatch"
    assert tokenizer.token_to_id("<|system|>") == config.sys_id, "system ID mismatch"
    
    print("\n✅ Verification successful! Special token IDs are exactly 0-4 in order.")
    print("=" * 40)
    
    # Sample encoding check
    sample_text = "Hello, Raven-2! Are we intelligent yet?"
    encoded = tokenizer.encode(sample_text)
    print(f"\nSample Text: '{sample_text}'")
    print(f"Tokens:      {encoded.tokens}")
    print(f"IDs:         {encoded.ids}")
    decoded = tokenizer.decode(encoded.ids)
    print(f"Decoded:     '{decoded}'")
    print("=" * 40)

def main(args=None):
    if args is None:
        import sys
        if any("ipykernel" in arg for arg in sys.argv):
            args = []
            
    parser = argparse.ArgumentParser(description="Train Byte-Level BPE Tokenizer for RAVEN-2")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/corpus.txt", 
        help="Path to training text corpus (default: data/corpus.txt)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=config.tokenizer_path, 
        help="Path to save tokenizer JSON (default from config)"
    )
    args = parser.parse_args(args)
    
    # For standalone test, if corpus doesn't exist, create a dummy corpus to verify
    if not os.path.exists(args.input):
        print(f"⚠️ Input corpus {args.input} not found.")
        print("Creating a small mock corpus to verify script...")
        os.makedirs(os.path.dirname(args.input), exist_ok=True)
        mock_text = (
            "Hello world! This is a simple English text corpus used for training our "
            "BPE tokenizer. We want to test that the tokenizer trains correctly, "
            "and all special tokens <|pad|>, <|user|>, <|raven|>, <|eos|>, <|system|> "
            "are assigned standard, predictable IDs from 0 to 4 in our 16K BPE dictionary.\n"
        ) * 1000
        with open(args.input, "w", encoding="utf-8") as f:
            f.write(mock_text)
            
    train_tokenizer(args.input, args.output)

if __name__ == "__main__":
    main()
