# posttrain.py
# Personality Fine-Tuning loop for RAVEN-2 (100M parameter LLM)
# Implements the final phase to bake in Raven's dry, sarcastic, direct character.
# Uses low learning rates and short steps to refine conversational style without hurting knowledge.

import os
import sys
import time
import math
import json
import shutil
import argparse
import signal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from model import Raven2
from train import configure_optimizers, compute_lr, PrefetchDataLoader, estimate_loss, save_checkpoint, load_checkpoint
from instruct_tune import mask_targets, estimate_instruct_loss

# Global reference for interrupt handler
training_state = {
    "step": 0,
    "model": None,
    "optimizer": None,
    "scaler": None,
    "best_val_loss": float("inf"),
    "train_losses": [],
    "val_losses": [],
    "config_snapshot": {},
    "output_ckpt_path": config.posttrain_ckpt
}

def ctrl_c_handler(signum, frame):
    print("\n🛑 Clean interrupt received! Saving posttrain checkpoint before exit...")
    save_checkpoint(
        training_state["output_ckpt_path"],
        training_state["step"],
        training_state["model"],
        training_state["optimizer"],
        training_state["scaler"],
        training_state["best_val_loss"],
        training_state["train_losses"],
        training_state["val_losses"],
        training_state["config_snapshot"]
    )
    sys.exit(0)

# Register interrupt handler
signal.signal(signal.SIGINT, ctrl_c_handler)


@torch.no_grad()
def generate_personality_sample(model, tokenizer_path):
    """Generate personality samples during post-training to monitor Raven character development.
    """
    from tokenizers import Tokenizer
    if not os.path.exists(tokenizer_path):
        return
        
    model.eval()
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    prompts = [
        "what's up?",
        "I'm really bored",
        "make me laugh",
        "what is gravity?"
    ]
    
    print("\n" + "=" * 60)
    print(" 🐦‍⬛ PERSONALITY MONITOR (RAVEN-2 CHARACTER)")
    print("=" * 60)
    
    for prompt in prompts:
        formatted = f"<|system|>You are Raven. Sarcastic, dry, direct, honest.<|user|>{prompt}<|raven|>"
        idx = torch.tensor([tokenizer.encode(formatted).ids], dtype=torch.long, device=config.device)
        
        generated_ids = model.generate(
            idx, 
            max_new_tokens=50, 
            temperature=0.8, 
            top_k=40, 
            top_p=0.9, 
            repetition_penalty=1.1,
            use_kv_cache=True
        )
        
        response = tokenizer.decode(generated_ids[0].tolist())
        print(f"User : {prompt}")
        print(f"Raven: {response.strip()}")
        print("-" * 60)
        
    model.train()


def print_before_after(model, tokenizer_path, base_weights_exist):
    """Print personality responses before and after training to show the shift.
    """
    from tokenizers import Tokenizer
    if not os.path.exists(tokenizer_path):
        return
        
    tokenizer = Tokenizer.from_file(tokenizer_path)
    prompts = [
        "Who are you?",
        "Explain recursion",
        "I'm stressed about exams"
    ]
    
    print("\n" + "=" * 70)
    print(" 🌟 BEFORE/AFTER PERSONALITY SHIFT COMPILATION")
    print("=" * 70)
    
    # We assume model is currently in its fully trained post-trained state
    model.eval()
    
    for prompt in prompts:
        formatted = f"<|system|>You are Raven. Sarcastic, dry, direct, honest.<|user|>{prompt}<|raven|>"
        idx = torch.tensor([tokenizer.encode(formatted).ids], dtype=torch.long, device=config.device)
        
        # Current response
        gen_ids = model.generate(idx, max_new_tokens=60, temperature=0.7, top_k=40, top_p=0.9, use_kv_cache=True)
        trained_response = tokenizer.decode(gen_ids[0].tolist()).strip()
        
        print(f"Prompt: {prompt}")
        if base_weights_exist:
            print(f"  [Tuned Raven-2]: {trained_response}")
        else:
            print(f"  [Output]: {trained_response}")
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="RAVEN-2 Model Personality Post-training")
    parser.add_argument("--resume", action="store_true", help="Resume from latest posttrain checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Dry run with 20 steps only for testing")
    args = parser.parse_args()
    
    max_steps = config.posttrain_max_iters
    eval_interval = 250
    eval_iters = config.eval_iters
    
    if args.dry_run:
        print("🚨 DRY RUN MODE ACTIVE - Training for 20 steps only.")
        max_steps = 20
        eval_interval = 10
        eval_iters = 2
        
    # Set up post-train split paths
    posttrain_bin_path = config.posttrain_bin
    val_bin_path = "checkpoints/posttrain_val.bin"
    
    # Check if data exists, if not fallback to instruct data for demonstration
    if not os.path.exists(posttrain_bin_path):
        print(f"⚠️ Personality posttrain binary dataset not found at {posttrain_bin_path}.")
        print(f"Falling back to instruction tuning binary dataset {config.instruct_bin} for style training.")
        posttrain_bin_path = config.instruct_bin
        val_bin_path = "checkpoints/instruct_val.bin"
        
    if not os.path.exists(posttrain_bin_path):
        print("❌ No training data found. Please prepare data first.")
        sys.exit(1)
        
    print("Initialize RAVEN-2 model...")
    model = Raven2(config)
    model.to(config.device)
    
    # Load instruct base model weights
    base_weights_exist = False
    if not os.path.exists(config.instruct_ckpt) and not args.resume:
        print(f"⚠️ Base instruction checkpoint not found at {config.instruct_ckpt}.")
        if os.path.exists(config.pretrain_ckpt):
            print(f"Falling back to loading pre-trained weights from {config.pretrain_ckpt}...")
            ckpt = torch.load(config.pretrain_ckpt, map_location=config.device)
            model.load_state_dict(ckpt["model_state"])
            base_weights_exist = True
        else:
            print("No checkpoints found. Fine-tuning from scratch with random weights.")
    elif not args.resume:
        print(f"🔄 Loading instruction-tuned weights from {config.instruct_ckpt}...")
        ckpt = torch.load(config.instruct_ckpt, map_location=config.device)
        model.load_state_dict(ckpt["model_state"])
        print("✅ Base instruction weights loaded successfully.")
        base_weights_exist = True
        
    # Configure optimizer with very low learning rate
    optimizer = configure_optimizers(model, config.weight_decay, config.posttrain_lr)
    scaler = torch.amp.GradScaler(device_type="cuda", enabled=config.use_amp)
    
    # Resume safety
    start_step = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    
    config_snapshot = {
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_layer": config.n_layer,
        "ffn_mult": config.ffn_mult,
        "device": config.device,
        "dtype": config.dtype_str,
        "phase": "personality_posttraining"
    }
    
    training_state["model"] = model
    training_state["optimizer"] = optimizer
    training_state["scaler"] = scaler
    training_state["config_snapshot"] = config_snapshot
    
    latest_ckpt_path = "checkpoints/raven2_posttrain_latest.pt"
    if args.resume or os.path.exists(latest_ckpt_path):
        ckpt_to_load = latest_ckpt_path if os.path.exists(latest_ckpt_path) else config.posttrain_ckpt
        if os.path.exists(ckpt_to_load):
            start_step, best_val_loss, train_losses, val_losses = load_checkpoint(ckpt_to_load, model, optimizer, scaler)
            training_state["step"] = start_step
            training_state["best_val_loss"] = best_val_loss
            training_state["train_losses"] = train_losses
            training_state["val_losses"] = val_losses
            
    # Initialize Dataloaders
    print(f"Initializing streaming memory-mapped dataloaders for {posttrain_bin_path}...")
    train_loader = PrefetchDataLoader(posttrain_bin_path, config.block_size, config.batch_size)
    val_loader = PrefetchDataLoader(val_bin_path, config.block_size, config.batch_size)
    
    model.train()
    step = start_step
    
    print("\n🚀 Beginning RAVEN-2 Personality Post-training Loop...")
    print(f"Peak LR: {config.posttrain_lr:.2e} | Floor LR: {config.posttrain_min_lr:.2e} | Target Steps: {max_steps:,}")
    print("─" * 90)
    print(f"{'Step':8s} | {'Train Loss':10s} | {'Val Loss':10s} | {'LR':8s} | {'Tokens/s':10s} | {'Mem GB':8s} | {'ETA':10s}")
    print("─" * 90)
    
    t0 = time.time()
    
    while step < max_steps:
        lr = compute_lr(step, max_steps, config.posttrain_lr, config.posttrain_min_lr, config.posttrain_warmup)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
        loss_accum = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for micro_step in range(config.gradient_accumulation_steps):
            x, y = train_loader.get_batch()
            x, y = x.to(config.device), y.to(config.device)
            
            # Loss masking
            y_masked = mask_targets(y)
            
            with torch.amp.autocast(device_type="cuda", dtype=config.dtype, enabled=config.use_amp):
                logits, _, _ = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    y_masked.view(-1), 
                    ignore_index=-100
                )
                loss = loss / config.gradient_accumulation_steps
                
            loss_accum += loss.item()
            scaler.scale(loss).backward()
            
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        step += 1
        training_state["step"] = step
        train_losses.append(loss_accum)
        training_state["train_losses"] = train_losses
        
        # Evaluate validation loss and print personality samples
        if step % eval_interval == 0 or step == max_steps:
            val_loss = estimate_instruct_loss(model, val_loader, eval_iters)
            val_losses.append((step, val_loss))
            training_state["val_losses"] = val_losses
            
            # Print personality monitor
            generate_personality_sample(model, config.tokenizer_path)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                training_state["best_val_loss"] = best_val_loss
                save_checkpoint(
                    config.posttrain_ckpt,
                    step, model, optimizer, scaler,
                    best_val_loss, train_losses, val_losses, config_snapshot
                )
                
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            
            tokens_per_step = config.batch_size * config.gradient_accumulation_steps * config.block_size
            tok_per_sec = tokens_per_step * eval_interval / dt
            mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            
            steps_left = max_steps - step
            eta_seconds = (dt / eval_interval) * steps_left
            if eta_seconds > 3600:
                eta = f"{eta_seconds/3600:.1f} hrs"
            elif eta_seconds > 60:
                eta = f"{eta_seconds/60:.1f} mins"
            else:
                eta = f"{eta_seconds:.0f} s"
                
            print(f"{step:8d} | {loss_accum:10.4f} | {val_loss:10.4f} | {lr:.2e} | {tok_per_sec:10.0f} | {mem_gb:8.2f} | {eta:10s}")
            
        if step % 1000 == 0:
            save_checkpoint(
                latest_ckpt_path,
                step, model, optimizer, scaler,
                best_val_loss, train_losses, val_losses, config_snapshot
            )
            
    train_loader.close()
    val_loader.close()
    
    print("\n🎉 Personality Post-training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    save_checkpoint(
        config.posttrain_ckpt,
        step, model, optimizer, scaler,
        best_val_loss, train_losses, val_losses, config_snapshot
    )
    
    # Print before/after personality summary
    print_before_after(model, config.tokenizer_path, base_weights_exist)

if __name__ == "__main__":
    main()
