# instruct_tune.py
# Instruction Fine-Tuning loop for RAVEN-2 (100M parameter LLM)
# Implements CRITICAL loss masking where loss is only computed on raven response tokens.
# Prevents model from learning to predict system prompts and user inputs.

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
    "output_ckpt_path": config.instruct_ckpt
}

def ctrl_c_handler(signum, frame):
    print("\n🛑 Clean interrupt received! Saving instruct checkpoint before exit...")
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


def mask_targets(targets):
    """CRITICAL: Mask out system prompts and user inputs.
    Only compute loss on tokens between <|raven|> (inclusive) and <|eos|> (inclusive).
    Sets all other token targets to -100 so they are ignored by cross_entropy.
    """
    masked = targets.clone()
    batch, seq_len = targets.shape
    
    for b in range(batch):
        active = False
        for t in range(seq_len):
            tok = targets[b, t].item()
            if tok == config.raven_id:
                active = True
                
            # If not active, mask it
            if not active:
                masked[b, t] = -100
                
            # Stop active state AFTER the eos token is processed
            if tok == config.eos_id:
                active = False
                
    return masked


@torch.no_grad()
def generate_sample_qna(model, tokenizer_path):
    """Generate sample responses during instruction tuning to visually monitor quality.
    """
    from tokenizers import Tokenizer
    if not os.path.exists(tokenizer_path):
        return
        
    model.eval()
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    prompts = [
        ("What is photosynthesis?", "Explain scientifically but clearly."),
        ("Write a simple python script to reverse a list.", "Code only."),
        ("I'm feeling very tired today.", "Provide simple encouragement.")
    ]
    
    print("\n" + "=" * 60)
    print(" 🧪 GENERATION MONITOR (INSTRUCTION TUNING)")
    print("=" * 60)
    
    for user_query, sys_prompt in prompts:
        # Construct template
        prompt = f"<|system|>{sys_prompt}<|user|>{user_query}<|raven|>"
        idx = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=config.device)
        
        # Generate using KV cache
        generated_ids = model.generate(
            idx, 
            max_new_tokens=100, 
            temperature=0.7, 
            top_k=40, 
            top_p=0.9, 
            repetition_penalty=1.1,
            use_kv_cache=True
        )
        
        response = tokenizer.decode(generated_ids[0].tolist())
        print(f"System: {sys_prompt}")
        print(f"User  : {user_query}")
        print(f"Raven : {response.strip()}")
        print("-" * 60)
        
    model.train()


@torch.no_grad()
def estimate_instruct_loss(model, dataloader, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = dataloader.get_batch()
        x, y = x.to(config.device), y.to(config.device)
        
        # Mask targets
        y_masked = mask_targets(y)
        
        with torch.amp.autocast(device_type="cuda", dtype=config.dtype, enabled=config.use_amp):
            # Model forward with raw x, but we compute loss manually to handle masked labels
            logits, _, _ = model(x)
            # Compute cross entropy ignoring padding and masked system/user tokens
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                y_masked.view(-1), 
                ignore_index=-100
            )
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    parser = argparse.ArgumentParser(description="RAVEN-2 Model Instruction Fine-tuning")
    parser.add_argument("--resume", action="store_true", help="Resume from latest instruct checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Dry run with 20 steps only for testing")
    args = parser.parse_args()
    
    max_steps = config.instruct_max_iters
    eval_interval = config.eval_interval
    eval_iters = config.eval_iters
    
    if args.dry_run:
        print("🚨 DRY RUN MODE ACTIVE - Training for 20 steps only.")
        max_steps = 20
        eval_interval = 10
        eval_iters = 2
        
    # Check if instruct binary data is ready
    val_bin_path = "checkpoints/instruct_val.bin"
    if not os.path.exists(config.instruct_bin) or not os.path.exists(val_bin_path):
        print(f"❌ Instruction binary dataset not found at {config.instruct_bin}. Please run: python data/data_prep.py --mode instruct first.")
        sys.exit(1)
        
    print("Initialize RAVEN-2 model...")
    model = Raven2(config)
    model.to(config.device)
    
    # Load pre-trained base model weights
    if not os.path.exists(config.pretrain_ckpt) and not args.resume:
        print(f"⚠️ Pretrained checkpoint not found at {config.pretrain_ckpt}.")
        print("We will initialize with random weights (fine-tuning from scratch).")
    elif not args.resume:
        print(f"🔄 Loading pre-trained base weights from {config.pretrain_ckpt}...")
        ckpt = torch.load(config.pretrain_ckpt, map_location=config.device)
        model.load_state_dict(ckpt["model_state"])
        print("✅ Base weights loaded successfully.")
        
    # Configure optimizer with instruction tuning learning rate
    optimizer = configure_optimizers(model, config.weight_decay, config.instruct_lr)
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
        "phase": "instruction_tuning"
    }
    
    training_state["model"] = model
    training_state["optimizer"] = optimizer
    training_state["scaler"] = scaler
    training_state["config_snapshot"] = config_snapshot
    
    latest_ckpt_path = "checkpoints/raven2_instruct_latest.pt"
    if args.resume or os.path.exists(latest_ckpt_path):
        ckpt_to_load = latest_ckpt_path if os.path.exists(latest_ckpt_path) else config.instruct_ckpt
        if os.path.exists(ckpt_to_load):
            start_step, best_val_loss, train_losses, val_losses = load_checkpoint(ckpt_to_load, model, optimizer, scaler)
            training_state["step"] = start_step
            training_state["best_val_loss"] = best_val_loss
            training_state["train_losses"] = train_losses
            training_state["val_losses"] = val_losses
            
    # Initialize Dataloaders
    print("Initializing memory-mapped dataloaders with background prefetching...")
    train_loader = PrefetchDataLoader(config.instruct_bin, config.block_size, config.batch_size)
    val_loader = PrefetchDataLoader(val_bin_path, config.block_size, config.batch_size)
    
    model.train()
    step = start_step
    
    print("\n🚀 Beginning RAVEN-2 Instruction Tuning Loop...")
    print(f"Peak LR: {config.instruct_lr:.2e} | Floor LR: {config.instruct_min_lr:.2e} | Target Steps: {max_steps:,}")
    print("─" * 90)
    print(f"{'Step':8s} | {'Train Loss':10s} | {'Val Loss':10s} | {'LR':8s} | {'Tokens/s':10s} | {'Mem GB':8s} | {'ETA':10s}")
    print("─" * 90)
    
    t0 = time.time()
    
    while step < max_steps:
        lr = compute_lr(step, max_steps, config.instruct_lr, config.instruct_min_lr, config.instruct_warmup)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
        loss_accum = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for micro_step in range(config.gradient_accumulation_steps):
            x, y = train_loader.get_batch()
            x, y = x.to(config.device), y.to(config.device)
            
            # Apply loss masking
            y_masked = mask_targets(y)
            
            with torch.amp.autocast(device_type="cuda", dtype=config.dtype, enabled=config.use_amp):
                # Model forward with raw inputs
                logits, _, _ = model(x)
                # Compute masked cross entropy manually
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
        
        # Evaluate validation loss and sample generation
        if step % eval_interval == 0 or step == max_steps:
            val_loss = estimate_instruct_loss(model, val_loader, eval_iters)
            val_losses.append((step, val_loss))
            training_state["val_losses"] = val_losses
            
            # Print sample generation Q&A
            generate_sample_qna(model, config.tokenizer_path)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                training_state["best_val_loss"] = best_val_loss
                save_checkpoint(
                    config.instruct_ckpt,
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
    
    print("\n🎉 Instruction Tuning Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    save_checkpoint(
        config.instruct_ckpt,
        step, model, optimizer, scaler,
        best_val_loss, train_losses, val_losses, config_snapshot
    )

if __name__ == "__main__":
    main()
