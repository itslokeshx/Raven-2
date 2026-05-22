# inference.py
# Interactive terminal chat client for RAVEN-2 (100M parameter LLM)
# Implements auto-checkpoint loading, conversation history (last 5 turns),
# slash commands, sampling modes, typing animations, and performance tracking.

import os
import sys
import time
import argparse
import torch
from tokenizers import Tokenizer

import config
from model import Raven2

# Sampling Modes Definition
SAMPLING_MODES = {
    "default": {"temp": 0.70, "top_k": 40, "top_p": 0.90, "rep_pen": 1.10, "desc": "Balanced, intelligent"},
    "sharp":   {"temp": 0.55, "top_k": 25, "top_p": 0.80, "rep_pen": 1.10, "desc": "Focused, precise"},
    "chaos":   {"temp": 0.90, "top_k": 60, "top_p": 0.95, "rep_pen": 1.05, "desc": "Wild, creative"},
    "cold":    {"temp": 0.40, "top_k": 15, "top_p": 0.70, "rep_pen": 1.08, "desc": "Dry, minimal"},
    "precise": {"temp": 0.30, "top_k": 10, "top_p": 0.65, "rep_pen": 1.12, "desc": "Maximum determinism"}
}

# Fun, dry fallbacks when generation is empty
WITTY_FALLBACKS = [
    "I have nothing to say to that. Literally.",
    "Silence is a valid response, right?",
    "That prompt did not trigger any neurons. Next.",
    "Error 404: Witty remark not found.",
    "Let's pretend you said something interesting instead.",
    "My parameters are silent on this matter.",
    "...",
]

def load_model():
    """Auto-detect and load the best available checkpoint.
    """
    model = Raven2(config)
    
    checkpoints = [
        ("posttrain", config.posttrain_ckpt),
        ("instruct", config.instruct_ckpt),
        ("pretrain", config.pretrain_ckpt)
    ]
    
    loaded_name = "random (scratch)"
    
    for name, path in checkpoints:
        if os.path.exists(path):
            print(f"🔄 Loading {name} weights from {path}...")
            state = torch.load(path, map_location=config.device)
            model.load_state_dict(state["model_state"])
            print(f"✅ Loaded checkpoint: {name}")
            loaded_name = name
            break
        # Also check Google Drive in Colab
        elif os.path.exists("/content/drive"):
            drive_path = os.path.join(config.drive_backup_dir, os.path.basename(path))
            if os.path.exists(drive_path):
                print(f"🔄 Loading {name} weights from Google Drive...")
                state = torch.load(drive_path, map_location=config.device)
                model.load_state_dict(state["model_state"])
                print(f"✅ Loaded Google Drive checkpoint: {name}")
                loaded_name = name
                break
                
    if loaded_name == "random (scratch)":
        print("⚠️ No checkpoints found. Initializing with random weights.")
        
    model.to(config.device)
    model.eval()
    return model, loaded_name

def clean_response(text):
    """Clean generated response by stripping leaked special tokens.
    """
    for tok in config.special_tokens:
        text = text.replace(tok, "")
    return text.strip()

def stream_print(text, delay=0.015):
    """Animate terminal typing effect for organic feel.
    """
    sys.stdout.write("Raven: ")
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")

def main():
    parser = argparse.ArgumentParser(description="RAVEN-2 Interactive Chat Client")
    parser.add_argument("--single", type=str, default=None, help="Run a single prompt and exit")
    args = parser.parse_args()
    
    # Verify tokenizer exists
    if not os.path.exists(config.tokenizer_path):
        print(f"❌ Tokenizer not found at: {config.tokenizer_path}. Please train it first.")
        sys.exit(1)
        
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    
    # Load model weights
    model, checkpoint_type = load_model()
    
    # Setup initial session variables
    current_mode = "default"
    system_prompt = "You are Raven. A highly advanced 100M parameter AI agent. Sarcastic, dry, direct, honest."
    history = []  # List of dicts: [{"user": "...", "raven": "..."}]
    
    # Tracking statistics
    tokens_generated = 0
    total_latency = 0.0
    
    # Single prompt run mode
    if args.single:
        # Generate single response and exit
        prompt = f"<|system|>{system_prompt}<|user|>{args.single}<|raven|>"
        idx = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=config.device)
        
        t0 = time.time()
        with torch.no_grad():
            gen_ids = model.generate(
                idx, 
                max_new_tokens=150, 
                temperature=0.7, 
                top_k=40, 
                top_p=0.9, 
                repetition_penalty=1.1,
                use_kv_cache=True
            )
        latency = time.time() - t0
        output = tokenizer.decode(gen_ids[0].tolist())
        output = clean_response(output)
        if not output:
            import random
            output = random.choice(WITTY_FALLBACKS)
        print(f"\nRaven: {output}")
        print(f"⚡ Latency: {latency:.2f}s | Speed: {len(gen_ids[0])/latency:.1f} tok/s")
        return

    # Print welcome header
    print("=" * 60)
    print(f" 🐦‍⬛  RAVEN-2  |  100M parameter LLM")
    print(f" Active Checkpoint: {checkpoint_type.upper()}")
    print(f" Hardware Device:  {config.device.upper()} | Context: {config.block_size} tokens")
    print("=" * 60)
    print(" Type your prompt below. Type /help to see all commands.")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
            
        if not user_input:
            continue
            
        # Handle Slash Commands
        if user_input.startswith("/"):
            parts = user_input.split(" ", 1)
            cmd = parts[0].lower()
            
            if cmd == "/exit":
                print("Bye.")
                break
            elif cmd == "/help":
                print("\n" + "─" * 40)
                print(" 🛠️  AVAILABLE SLASH COMMANDS")
                print("─" * 40)
                print("  /mode [name] - Switch sampling mode (precise, sharp, default, chaos, cold)")
                print("  /modes       - List detailed parameters for all sampling modes")
                print("  /system [X]  - Change or print the current system prompt")
                print("  /stats       - View model and performance stats")
                print("  /reset       - Clear conversation history")
                print("  /clear       - Clear terminal screen")
                print("  /save        - Save chat transcript to a text file")
                print("  /exit        - Exit the client")
                print("─" * 40 + "\n")
                continue
            elif cmd == "/clear":
                os.system("clear" if os.name == "posix" else "cls")
                continue
            elif cmd == "/reset":
                history = []
                print("✨ Conversation history cleared.\n")
                continue
            elif cmd == "/modes":
                print("\n" + "─" * 50)
                print(" 🎛️  SAMPLING MODES")
                print("─" * 50)
                for name, p in SAMPLING_MODES.items():
                    active = "⭐" if name == current_mode else "  "
                    print(f"{active} {name:8s} | Temp: {p['temp']:.2f} | Top-K: {p['top_k']:3d} | Top-P: {p['top_p']:.2f} | RepPen: {p['rep_pen']:.2f} | {p['desc']}")
                print("─" * 50 + "\n")
                continue
            elif cmd == "/mode":
                if len(parts) < 2:
                    print(f"Current mode is: {current_mode}\n")
                    continue
                mode_name = parts[1].strip().lower()
                if mode_name in SAMPLING_MODES:
                    current_mode = mode_name
                    print(f"🎛️ Switched sampling mode to: {current_mode.upper()} ({SAMPLING_MODES[current_mode]['desc']})\n")
                else:
                    print(f"❌ Unknown mode: {mode_name}. Choose from: {', '.join(SAMPLING_MODES.keys())}\n")
                continue
            elif cmd == "/system":
                if len(parts) < 2:
                    print(f"Current system prompt: '{system_prompt}'\n")
                    continue
                system_prompt = parts[1].strip()
                print(f"⚙️ System prompt updated to: '{system_prompt}'\n")
                continue
            elif cmd == "/stats":
                print("\n" + "─" * 40)
                print(" 📊 SYSTEM STATISTICS")
                print("─" * 40)
                print(f"  Checkpoint Type:   {checkpoint_type.upper()}")
                print(f"  Tokenizer Vocab:   {tokenizer.get_vocab_size():,}")
                print(f"  Device Name:       {config.device.upper()}")
                print(f"  Tokens Generated:  {tokens_generated:,}")
                if total_latency > 0:
                    print(f"  Average Latency:   {total_latency/len(history):.2f}s per turn")
                    print(f"  Average Speed:     {tokens_generated/total_latency:.1f} tok/s")
                print(f"  History Size:      {len(history)} / 5 turns")
                print("─" * 40 + "\n")
                continue
            elif cmd == "/save":
                filename = f"raven2_chat_{int(time.time())}.txt"
                try:
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(f"RAVEN-2 Chat Transcript\n")
                        f.write(f"System Prompt: {system_prompt}\n")
                        f.write("=" * 60 + "\n\n")
                        for turn in history:
                            f.write(f"You: {turn['user']}\n")
                            f.write(f"Raven: {turn['raven']}\n\n")
                    print(f"💾 Transcript saved to: {filename}\n")
                except Exception as e:
                    print(f"❌ Saving failed: {e}\n")
                continue
            else:
                print(f"❌ Unknown command: {cmd}. Type /help for assistance.\n")
                continue

        # 2. Build Prompt incorporating historical context (limit to last 5 turns)
        history_context = history[-5:]
        prompt = f"<|system|>{system_prompt}"
        for turn in history_context:
            prompt += f"<|user|>{turn['user']}<|raven|>{turn['raven']}<|eos|>"
        # Append current user prompt
        prompt += f"<|user|>{user_input}<|raven|>"
        
        # Tokenize prompt
        idx = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=config.device)
        
        # Display typing indicator animation
        sys.stdout.write("Raven: thinking...")
        sys.stdout.flush()
        
        # Get active mode parameters
        p = SAMPLING_MODES[current_mode]
        
        t0 = time.time()
        # Generate tokens
        with torch.no_grad():
            gen_ids = model.generate(
                idx, 
                max_new_tokens=100, 
                temperature=p["temp"], 
                top_k=p["top_k"], 
                top_p=p["top_p"], 
                repetition_penalty=p["rep_pen"],
                use_kv_cache=True
            )
        latency = time.time() - t0
        
        # Erase thinking message
        sys.stdout.write("\r" + " " * 20 + "\r")
        sys.stdout.flush()
        
        # Decode and clean response
        new_tokens_count = len(gen_ids[0])
        response = tokenizer.decode(gen_ids[0].tolist())
        response = clean_response(response)
        
        # Fallback if generation is blank
        if not response:
            import random
            response = random.choice(WITTY_FALLBACKS)
            
        # Display response with character-by-character typing effect
        stream_print(response)
        
        # Display token speed statistics in dim text
        tok_speed = new_tokens_count / latency if latency > 0 else 0
        print(f"\033[90m⚡ Speed: {tok_speed:.1f} tok/s | Latency: {latency:.2f}s | Tokens: {new_tokens_count}\033[0m\n")
        
        # Update metrics and conversation history
        tokens_generated += new_tokens_count
        total_latency += latency
        history.append({"user": user_input, "raven": response})

if __name__ == "__main__":
    main()
