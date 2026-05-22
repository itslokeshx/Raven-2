# train.py
# Production-grade Pre-training script for RAVEN-2 (100M parameter LLM)
# Implements Cosine LR Decay with Warmup, weight decay parameter groups,
# zero-copy memory mapped dataloading with prefetching, AMP,
# gradient checkpointing, resume safety, Ctrl+C interrupt handler, and Drive sync.

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

# Global reference to model/optimizer/scaler for Ctrl+C handler
training_state = {
    "step": 0,
    "model": None,
    "optimizer": None,
    "scaler": None,
    "best_val_loss": float("inf"),
    "train_losses": [],
    "val_losses": [],
    "config_snapshot": {},
    "output_ckpt_path": config.pretrain_ckpt
}

def ctrl_c_handler(signum, frame):
    print("\n🛑 Clean interrupt received! Saving checkpoint before exit...")
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


class MemmapDataLoader:
    """Zero-copy DataLoader using numpy.memmap.
    Samples random chunks of block_size + 1 from binary uint16 files.
    """
    def __init__(self, bin_path, block_size, batch_size):
        self.bin_path = bin_path
        self.block_size = block_size
        self.batch_size = batch_size
        
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary file not found: {bin_path}")
            
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)
        
    def get_batch(self):
        max_idx = self.n_tokens - self.block_size - 1
        if max_idx <= 0:
            # Fallback for very small datasets (e.g. dry-runs / tests)
            x = torch.zeros((self.batch_size, self.block_size), dtype=torch.long)
            y = torch.zeros((self.batch_size, self.block_size), dtype=torch.long)
            return x, y
            
        ix = np.random.randint(0, max_idx, (self.batch_size,))
        
        x_list, y_list = [], []
        for i in ix:
            chunk = self.data[i : i + self.block_size + 1].astype(np.int64)
            x_list.append(chunk[:-1])
            y_list.append(chunk[1:])
            
        x = torch.from_numpy(np.stack(x_list))
        y = torch.from_numpy(np.stack(y_list))
        return x, y


import queue
import threading

class PrefetchDataLoader:
    """Wrapper that prefetches batches in a background thread to maximize GPU utilization.
    """
    def __init__(self, bin_path, block_size, batch_size, queue_size=4):
        self.loader = MemmapDataLoader(bin_path, block_size, batch_size)
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._producer, daemon=True)
        self.thread.start()
        
    def _producer(self):
        while not self.stop_event.is_set():
            try:
                batch = self.loader.get_batch()
                # Use a timeout to regularly check for stop event
                self.queue.put(batch, timeout=0.1)
            except queue.Full:
                continue
            except Exception:
                break
                
    def get_batch(self):
        # Fallback to direct loading if queue empty (should rarely happen)
        try:
            return self.queue.get(timeout=1.0)
        except queue.Empty:
            return self.loader.get_batch()
            
    def close(self):
        self.stop_event.set()
        # Drain the queue to unblock the producer
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except Exception:
            pass


def compute_lr(step, max_iters, learning_rate, min_lr, warmup_iters):
    """Cosine learning rate decay with linear warmup.
    """
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters
    if step > max_iters:
        return min_lr
    decay_ratio = (step - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def configure_optimizers(model, weight_decay, learning_rate):
    """Configure AdamW with two parameter groups: decay and no_decay.
    Weights of 2D parameters (Linear weights) get decay, 1D parameters (Norms, biases) get zero decay.
    """
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear,)
    blacklist_weight_modules = (nn.Embedding, RMSNorm, nn.LayerNorm)
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn
            
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
                
    # Verify everything is accounted for
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"Parameters {str(inter_params)} in both groups!"
    assert len(param_dict.keys() - union_params) == 0, f"Parameters {str(param_dict.keys() - union_params)} not categorized!"
    
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}
    ]
    
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))
    return optimizer


def save_checkpoint(path, step, model, optimizer, scaler, best_val_loss, train_losses, val_losses, config_snapshot):
    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "step": step,
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "train_loss_history": train_losses,
        "val_loss_history": val_losses,
        "config": config_snapshot
    }
    torch.save(state, path)
    print(f"💾 Checkpoint saved: {path} (Step {step}, Best Val Loss {best_val_loss:.4f})")
    
    # Sync to Google Drive if in Colab
    if os.path.exists("/content/drive"):
        try:
            drive_dest_dir = config.drive_backup_dir
            os.makedirs(drive_dest_dir, exist_ok=True)
            drive_path = os.path.join(drive_dest_dir, os.path.basename(path))
            shutil.copy(path, drive_path)
            print(f"☁️ Backup synced to Google Drive: {drive_path}")
        except Exception as e:
            print(f"⚠️ Drive backup sync failed: {e}")

def load_checkpoint(path, model, optimizer, scaler):
    if not os.path.exists(path):
        # Check in drive backup
        drive_path = os.path.join(config.drive_backup_dir, os.path.basename(path)) if os.path.exists("/content/drive") else None
        if drive_path and os.path.exists(drive_path):
            print(f"Found Google Drive checkpoint, restoring: {drive_path}")
            shutil.copy(drive_path, path)
        else:
            return 0, float("inf"), [], []
            
    print(f"🔄 Loading checkpoint from {path}...")
    state = torch.load(path, map_location=config.device)
    
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if scaler is not None and state["scaler_state"] is not None:
        scaler.load_state_dict(state["scaler_state"])
        
    print(f"✅ Successfully resumed from step {state['step']} with val_loss={state['best_val_loss']:.4f}")
    return state["step"], state["best_val_loss"], state.get("train_loss_history", []), state.get("val_loss_history", [])


@torch.no_grad()
def estimate_loss(model, dataloader, eval_iters, pad_id):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = dataloader.get_batch()
        x, y = x.to(config.device), y.to(config.device)
        with torch.amp.autocast(device_type="cuda", dtype=config.dtype, enabled=config.use_amp):
            _, loss, _ = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    parser = argparse.ArgumentParser(description="RAVEN-2 Model Pre-training")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Dry run with 20 steps only for testing")
    parser.add_argument("--steps", type=int, default=None, help="Force override total pre-training steps")
    args = parser.parse_args()
    
    # 1. Setup config overrides
    max_steps = config.max_iters
    if args.dry_run:
        print("🚨 DRY RUN MODE ACTIVE - Training for 20 steps only.")
        max_steps = 20
        config.eval_interval = 10
        config.eval_iters = 2
    elif args.steps is not None:
        max_steps = args.steps
        
    # Check if pre-training binary data is ready
    if not os.path.exists(config.train_bin) or not os.path.exists(config.val_bin):
        print(f"❌ Binary dataset not found at {config.train_bin}. Please run data/data_prep.py --mode pretrain first.")
        sys.exit(1)
        
    print("Initialize RAVEN-2 model...")
    model = Raven2(config)
    model.to(config.device)
    
    # Configure optimizer
    optimizer = configure_optimizers(model, config.weight_decay, config.learning_rate)
    
    # Configure scaler for mixed precision training
    scaler = torch.amp.GradScaler(device_type="cuda", enabled=config.use_amp)
    
    # Load checkpoint if requested
    start_step = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    
    # Snapshot of configuration for metadata logging
    config_snapshot = {
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_layer": config.n_layer,
        "ffn_mult": config.ffn_mult,
        "device": config.device,
        "dtype": config.dtype_str
    }
    
    # Sync global references for interrupt handler
    training_state["model"] = model
    training_state["optimizer"] = optimizer
    training_state["scaler"] = scaler
    training_state["config_snapshot"] = config_snapshot
    
    # Check if a checkpoint exists and we either specified --resume or are auto-resuming
    latest_ckpt_path = "checkpoints/raven2_pretrain_latest.pt"
    if args.resume or os.path.exists(latest_ckpt_path):
        ckpt_to_load = latest_ckpt_path if os.path.exists(latest_ckpt_path) else config.pretrain_ckpt
        if os.path.exists(ckpt_to_load) or (os.path.exists("/content/drive") and os.path.exists(os.path.join(config.drive_backup_dir, os.path.basename(ckpt_to_load)))):
            start_step, best_val_loss, train_losses, val_losses = load_checkpoint(ckpt_to_load, model, optimizer, scaler)
            training_state["step"] = start_step
            training_state["best_val_loss"] = best_val_loss
            training_state["train_losses"] = train_losses
            training_state["val_losses"] = val_losses
            
    # Initialize Prefetching Dataloaders
    print("Initializing streaming memory-mapped dataloaders with background prefetching...")
    train_loader = PrefetchDataLoader(config.train_bin, config.block_size, config.batch_size)
    val_loader = PrefetchDataLoader(config.val_bin, config.block_size, config.batch_size)
    
    # Start training
    model.train()
    step = start_step
    
    print("\n🚀 Beginning RAVEN-2 Pre-training Loop...")
    print(f"Peak LR: {config.learning_rate:.2e} | Floor LR: {config.min_lr:.2e} | Target Steps: {max_steps:,}")
    print("─" * 90)
    print(f"{'Step':8s} | {'Train Loss':10s} | {'Val Loss':10s} | {'LR':8s} | {'Tokens/s':10s} | {'Mem GB':8s} | {'ETA':10s}")
    print("─" * 90)
    
    t0 = time.time()
    
    while step < max_steps:
        # Determine step learning rate
        lr = compute_lr(step, max_steps, config.learning_rate, config.min_lr, config.warmup_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
        # Accumulate gradients over micro-steps
        loss_accum = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for micro_step in range(config.gradient_accumulation_steps):
            x, y = train_loader.get_batch()
            x, y = x.to(config.device), y.to(config.device)
            
            # Forward pass inside automatic mixed precision
            with torch.amp.autocast(device_type="cuda", dtype=config.dtype, enabled=config.use_amp):
                _, loss, _ = model(x, y)
                # Scale loss to account for gradient accumulation
                loss = loss / config.gradient_accumulation_steps
                
            loss_accum += loss.item()
            
            # Backward pass with scaled loss
            scaler.scale(loss).backward()
            
        # Unscale, Clip gradients, Step optimizer, update scale
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        # Log performance
        step += 1
        training_state["step"] = step
        train_losses.append(loss_accum)
        training_state["train_losses"] = train_losses
        
        # Evaluate validation loss periodically
        if step % config.eval_interval == 0 or step == max_steps:
            val_loss = estimate_loss(model, val_loader, config.eval_iters, config.pad_id)
            val_losses.append((step, val_loss))
            training_state["val_losses"] = val_losses
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                training_state["best_val_loss"] = best_val_loss
                save_checkpoint(
                    config.pretrain_ckpt,
                    step, model, optimizer, scaler,
                    best_val_loss, train_losses, val_losses, config_snapshot
                )
                
            # Log metrics
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            
            # Calculate speed metrics
            tokens_per_step = config.batch_size * config.gradient_accumulation_steps * config.block_size
            tok_per_sec = tokens_per_step * config.eval_interval / dt
            
            # Calculate memory metrics
            mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            
            # Calculate ETA
            steps_left = max_steps - step
            eta_seconds = (dt / config.eval_interval) * steps_left
            if eta_seconds > 3600:
                eta = f"{eta_seconds/3600:.1f} hrs"
            elif eta_seconds > 60:
                eta = f"{eta_seconds/60:.1f} mins"
            else:
                eta = f"{eta_seconds:.0f} s"
                
            print(f"{step:8d} | {loss_accum:10.4f} | {val_loss:10.4f} | {lr:.2e} | {tok_per_sec:10.0f} | {mem_gb:8.2f} | {eta:10s}")
            
            # Save losses to JSON
            with open("checkpoints/losses.json", "w") as f:
                json.dump({"train": train_losses, "val": val_losses}, f)
                
        # Save latest checkpoint periodically for easy recovery
        if step % 1000 == 0:
            save_checkpoint(
                latest_ckpt_path,
                step, model, optimizer, scaler,
                best_val_loss, train_losses, val_losses, config_snapshot
            )
            
    # Clean shutdown dataloaders
    train_loader.close()
    val_loader.close()
    
    print("\n🎉 Pre-training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    # Save final model
    save_checkpoint(
        config.pretrain_ckpt,
        step, model, optimizer, scaler,
        best_val_loss, train_losses, val_losses, config_snapshot
    )

if __name__ == "__main__":
    main()
