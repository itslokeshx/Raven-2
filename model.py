# model.py
# Complete, production-grade PyTorch implementation of RAVEN-2 (100M parameter LLM)
# Implements RMSNorm, RoPE, Grouped Query Attention (GQA), and SwiGLU

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class RMSNorm(nn.Module):
    """RMSNorm (Root Mean Square Layer Normalization) without mean subtraction and bias.
    Used in LLaMA, Mistral, and other modern LLMs for faster and stable training.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_rope_freqs(head_dim, seq_len, base=10000.0):
    """Precompute rotary positional embedding cos/sin tables.
    Matches RoPE mathematics, treating head_dim as pairs.
    """
    assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
    # inv_freq shape: (head_dim // 2,)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # t shape: (seq_len,)
    t = torch.arange(seq_len, dtype=torch.float32)
    # freqs shape: (seq_len, head_dim // 2)
    freqs = torch.outer(t, inv_freq)
    # Precompute cos and sin
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rope(x, cos, sin, position_ids=None):
    """Apply Rotary Position Embeddings (RoPE) to queries/keys.
    x shape: (batch, heads, seq_len, head_dim)
    cos, sin shape: (max_seq_len, head_dim // 2)
    position_ids shape: (batch, seq_len)
    """
    batch, heads, seq_len, head_dim = x.shape
    
    if position_ids is not None:
        # Index cos and sin for the specific position IDs
        # Shape: (batch, 1, seq_len, head_dim // 2)
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        # Sliced: (1, 1, seq_len, head_dim // 2)
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0).to(x.device)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0).to(x.device)
        
    # Cast cos and sin to match x's dtype
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)

    # Split x into two halves along the last dimension
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    
    # Rotate: [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.cat([rx1, rx2], dim=-1)


class CausalSelfAttention(nn.Module):
    """Grouped Query Attention (GQA) with rotary position embeddings and KV cache.
    Significantly reduces KV cache memory consumption for fast inference.
    """
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        
        # Combined QKV projection: Query (n_head * hd) and Key/Value (2 * n_kv_head * hd)
        self.qkv_proj = nn.Linear(
            cfg.n_embd, 
            (self.n_head + 2 * self.n_kv_head) * self.head_dim, 
            bias=cfg.bias
        )
        self.out_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=cfg.bias)

    def forward(self, x, cos, sin, past_kv=None, position_ids=None):
        batch, seq_len, n_embd = x.shape
        
        # QKV Projection
        qkv = self.qkv_proj(x)
        
        # Split QKV into q, k, v
        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_head * self.head_dim
        q, k, v = torch.split(qkv, [q_dim, kv_dim, kv_dim], dim=-1)
        
        # Reshape to (batch, head, seq_len, head_dim)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to query and key
        q = apply_rope(q, cos, sin, position_ids)
        k = apply_rope(k, cos, sin, position_ids)
        
        # Update KV cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        present_kv = (k, v)
        
        # Grouped Query Attention (replicate Keys & Values to match Query heads)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
            
        # Run FlashAttention (via auto-dispatching scaled_dot_product_attention)
        # causal = True only if seq_len > 1 and we are not using KV cache
        is_causal = (seq_len > 1) and (past_kv is None)
        
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=config.dropout if self.training else 0.0, 
            is_causal=is_causal
        )
        
        # Reshape and project out
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.out_proj(out)
        return out, present_kv


class SwiGLU(nn.Module):
    """SwiGLU (Swish Gated Linear Unit) feed-forward network.
    Uses LLaMA dimension rounding formula: hidden_dim = round(n_embd * ffn_mult * 2 / 3 / 256) * 256.
    """
    def __init__(self, n_embd, hidden_dim, bias=False):
        super().__init__()
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=bias)  # Gate projection
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=bias)  # Up projection
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=bias)  # Down projection

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer Block supporting optional gradient checkpointing.
    """
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd)
        
        # Calculate SwiGLU hidden dim
        raw_hidden = int(cfg.n_embd * cfg.ffn_mult * 2 / 3)
        hidden_dim = round(raw_hidden / 256) * 256
        self.ffn = SwiGLU(cfg.n_embd, hidden_dim, bias=cfg.bias)

    def forward(self, x, cos, sin, past_kv=None, position_ids=None):
        # Attention block with residual
        normed_x = self.attn_norm(x)
        attn_out, present_kv = self.attn(normed_x, cos, sin, past_kv, position_ids)
        x = x + attn_out
        
        # FFN block with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x, present_kv


class Raven2(nn.Module):
    """The main RAVEN-2 language model (100M parameter LLM built from scratch).
    """
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        
        # Embedding layer
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        
        # Transformer layers
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        
        # Final normalization
        self.final_norm = RMSNorm(cfg.n_embd)
        
        # LM Head
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=cfg.bias)
        
        # Weight tying
        if cfg.weight_tying:
            self.lm_head.weight = self.token_embedding.weight
            
        # Precompute RoPE freqs and register as buffers
        cos, sin = precompute_rope_freqs(cfg.n_embd // cfg.n_head, cfg.block_size, cfg.rope_base)
        self.register_buffer("cos_freqs", cos)
        self.register_buffer("sin_freqs", sin)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report parameter count
        self.count_parameters()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Normal distribution with small std dev for weights
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding = self.token_embedding.weight.numel()
        
        print("─" * 40)
        print(f" RAVEN-2 Model Initialized")
        print(f" Total parameters:     {total/1e6:.2f}M")
        print(f" Trainable parameters: {trainable/1e6:.2f}M")
        print(f" Embedding parameters: {embedding/1e6:.2f}M")
        print(f" Non-embedding params: {(total-embedding)/1e6:.2f}M")
        print("─" * 40)
        
        assert 90e6 <= total <= 115e6, f"Parameter count out of range: {total/1e6:.2f}M"

    def forward(self, idx, targets=None, past_kvs=None, position_ids=None):
        batch, seq_len = idx.shape
        
        # If position_ids is None, create it dynamically
        if position_ids is None:
            if past_kvs is not None and past_kvs[0] is not None:
                past_length = past_kvs[0][0].shape[2]
                position_ids = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=idx.device).unsqueeze(0)
            else:
                position_ids = torch.arange(0, seq_len, dtype=torch.long, device=idx.device).unsqueeze(0)
                
        # Token Embeddings
        x = self.token_embedding(idx)
        x = self.dropout(x)
        
        # Cos/Sin for RoPE (registered buffers)
        cos = self.cos_freqs
        sin = self.sin_freqs
        
        present_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            
            # Support gradient checkpointing during training
            if self.config.gradient_checkpointing and self.training:
                # Custom forward helper for gradient checkpointing
                def create_checkpoint_forward(layer):
                    def custom_forward(*args):
                        return layer(*args)
                    return custom_forward
                
                # Checkpoint must receive tensors that require grad
                x, present_kv = torch.utils.checkpoint.checkpoint(
                    create_checkpoint_forward(block),
                    x, cos, sin, past_kv, position_ids,
                    use_reentrant=False
                )
            else:
                x, present_kv = block(x, cos, sin, past_kv, position_ids)
                
            if present_kvs is not None:
                present_kvs.append(present_kv)
                
        # Final Norm
        x = self.final_norm(x)
        
        # Output Head
        if targets is not None:
            logits = self.lm_head(x)
            # Ignore padding token (0) in loss calculation
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=self.config.pad_id
            )
        else:
            # During generation, we only care about the very last token's logits
            if past_kvs is not None:
                logits = self.lm_head(x[:, -1:, :])
            else:
                logits = self.lm_head(x)
            loss = None
            
        return logits, loss, present_kvs

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=40,
                 top_p=0.9, repetition_penalty=1.1, use_kv_cache=True):
        """Efficient autoregressive token generation supporting KV caching, temperature scaling,
        repetition penalty, and top-k/top-p filtering. Stops on config.eos_id.
        """
        self.eval()
        batch = idx.size(0)
        
        generated = []
        past_kvs = None
        
        # Store context tokens for repetition penalty tracking
        context_tokens = idx.clone()
        
        for step in range(max_new_tokens):
            # Cut down block size if context overflows
            if context_tokens.size(1) >= self.config.block_size:
                # If not using KV cache, crop. If using KV cache, we shouldn't overflow block size
                if use_kv_cache:
                    break
                else:
                    idx = idx[:, -self.config.block_size:]
            
            # Forward pass: if past_kvs is present, only pass the last generated token
            if use_kv_cache and past_kvs is not None:
                input_idx = idx[:, -1:]
                # Position is the index of the last token
                position_ids = torch.tensor([[context_tokens.size(1) - 1]], dtype=torch.long, device=idx.device)
                logits, _, past_kvs = self.forward(input_idx, past_kvs=past_kvs, position_ids=position_ids)
            else:
                logits, _, past_kvs = self.forward(idx)
                
            # Logits shape: (batch, seq_len, vocab_size) -> take last token
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(batch):
                    # Find unique tokens generated so far in this batch
                    unique_toks = torch.unique(context_tokens[b])
                    for t in unique_toks:
                        t_idx = t.item()
                        if next_token_logits[b, t_idx] < 0:
                            next_token_logits[b, t_idx] *= repetition_penalty
                        else:
                            next_token_logits[b, t_idx] /= repetition_penalty
            
            # Apply Temperature scaling
            if temperature == 0.0:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                
                # Apply Top-K filtering
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                    
                # Apply Top-P (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter mask back to original logits shape
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample from multinomial distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            # Append next token
            idx = torch.cat([idx, next_token], dim=-1)
            context_tokens = torch.cat([context_tokens, next_token], dim=-1)
            generated.append(next_token)
            
            # Stop if EOS token is generated
            if batch == 1 and next_token.item() == self.config.eos_id:
                break
                
        # Return only newly generated tokens
        if len(generated) > 0:
            return torch.cat(generated, dim=-1)
        else:
            return torch.empty((batch, 0), dtype=torch.long, device=idx.device)


if __name__ == "__main__":
    # Test block to verify parameter count and forward pass works
    print("Testing RAVEN-2 model initialization...")
    model = Raven2(config)
    x = torch.randint(0, config.vocab_size, (2, 64))
    logits, loss, _ = model(x, targets=x)
    print("Forward pass successful!")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
