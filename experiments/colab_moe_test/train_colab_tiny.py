import os
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from huggingface_hub import upload_folder
import wandb
from tqdm import tqdm

# --- Configuration (Tiny Colab Version) ---
CONFIG_DICT = {
    "model_type": "yaongi_nemotron_tiny",
    "n_layers": 8,           # Reduced from 52
    "d_model": 256,          # Reduced from 2048
    "d_state": 64,           # Reduced from 128
    "vocab_size": 32768,
    "num_experts": 8,        # Reduced from 64
    "top_k": 2,              # Reduced from 6
    "n_heads": 8,            # Reduced from 32
    "n_kv_heads": 2,         # Reduced from 8
    "engram_layers": [1],    # Adjusted index
    "engram_avg_pool": 2,
    "init_scale": 0.01,
    "layer_norm_epsilon": 1e-5,
    "tie_word_embeddings": False,
}

TRAIN_CONFIG = {
    "project_name": "Yaongi-Tiny-Colab-Test",
    "batch_size": 32,      
    "grad_accum": 1,       
    "max_seq_len": 512,      # Reduced
    "max_steps": 200,        # Quick Test
    "save_steps": 50,
    "push_to_hub": False,    # Optional for test
    "hf_repo_id": "gyung/Yaongi-Tiny-Test",
    # WSD Scheduler
    "lr_max": 2e-3,       
    "lr_min": 1e-4,
    "warmup_steps": 20,  
    "decay_start_step": 160, 
}

# --- 1. Teon Optimizer (Pure PyTorch) ---
def zeropower_via_newtonschulz5(G, steps=5):
    assert len(G.shape) in (2, 3)
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.float() # Use float32 on T4/CPU for stability
    
    if G.ndim == 2:
        if G.size(0) > G.size(1): X = X.T
    else: # 3D
        if G.size(1) > G.size(2): X = X.transpose(1, 2)
        
    X = X / (X.norm(dim=-1, keepdim=True).norm(dim=-2, keepdim=True) + 1e-7)
    
    for _ in range(steps):
        if X.ndim == 2:
            A = X @ X.T
        else:
            A = torch.bmm(X, X.transpose(1, 2))
        
        B = b * A + c * (torch.bmm(A, A) if X.ndim==3 else A @ A)
        
        if X.ndim == 2:
            X = a * X + B @ X
        else:
            X = a * X + torch.bmm(B, X)
            
    if G.ndim == 2:
        if G.size(0) > G.size(1): X = X.T
    else:
        if G.size(1) > G.size(2): X = X.transpose(1, 2)
        
    return X.to(G.dtype)

class Teon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf
                
                if update.ndim >= 2:
                    g_ortho = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    if update.ndim == 2:
                        scale = max(1, p.size(0)/p.size(1))**0.5
                    else:
                        scale = max(1, p.size(1)/p.size(2))**0.5
                    p.data.add_(g_ortho, alpha=-lr * scale)
                else:
                    p.data.add_(update, alpha=-lr)

# --- 2. Model Components (Pure PyTorch) ---

class YaongiConfig(PretrainedConfig):
    model_type = "yaongi_nemotron"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in CONFIG_DICT.items():
            setattr(self, k, kwargs.get(k, v))

class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class Mamba2Block(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        try:
            from mamba_ssm import Mamba2
            self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            self.is_slow = False
            print("✅ Mamba-SSM Kernel Loaded")
        except ImportError:
            self.is_slow = True
            print("⚠️ using Slow Fallback for Mamba")
            # Structure must match Mamba2 weights for compatibility
            self.in_proj = nn.Linear(d_model, d_model*2 + d_state*2 + 8, bias=False) # Rough approx for sizing
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            
    def forward(self, x):
        if not self.is_slow:
            return self.mamba(x)
        else:
            # Simple simulation of mixing
            B, T, C = x.shape
            h = self.in_proj(x)
            h = F.silu(h[:, :, :C]) # Gating simulation
            return self.out_proj(h)

class NemotronMoE(nn.Module):
    def __init__(self, d_model, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False) 
        
        self.experts_up = nn.ModuleList([
            nn.Linear(d_model, d_model * 4, bias=False) for _ in range(num_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(d_model * 4, d_model, bias=False) for _ in range(num_experts)
        ])
        self.activation = SquaredReLU()
        self.shared_expert = nn.Sequential(
             nn.Linear(d_model, d_model * 4, bias=False),
             SquaredReLU(),
             nn.Linear(d_model * 4, d_model, bias=False)
        )

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        
        logits = self.router(x_flat)
        probs = torch.sigmoid(logits)
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-6)
        
        out = self.shared_expert(x_flat) 
        
        for k in range(self.top_k):
            expert_indices_k = topk_indices[:, k]
            val_k = topk_probs[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                 mask = (expert_indices_k == e)
                 if mask.any():
                     sub_x = x_flat[mask]
                     ex_out = self.experts_down[e](self.activation(self.experts_up[e](sub_x)))
                     out[mask] += val_k[mask] * ex_out
                     
        return out.view(B, T, D)

class SimpleEngram(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.gate = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x, input_ids):
        return x + torch.sigmoid(self.gate(x)) # Dummy implementation

class GQAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, n_kv_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim * n_kv_heads, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim * n_kv_heads, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, C))

class HybridMoEEngram(PreTrainedModel):
    config_class = YaongiConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        d_model = config.d_model
        
        self.embed = nn.Embedding(config.vocab_size, d_model)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(config.n_layers):
            if i in config.engram_layers:
                block = SimpleEngram(d_model, config.vocab_size)
            elif i % 2 == 1: # More frequent attention for tiny model
                block = GQAttention(d_model, config.n_heads, config.n_kv_heads)
            else:
                block = nn.Sequential(
                    Mamba2Block(d_model, config.d_state),
                    NemotronMoE(d_model, config.num_experts, config.top_k)
                )
            self.layers.append(block)
            self.norms.append(RMSNorm(d_model))
            
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, config.vocab_size, bias=False) 
        self.post_init()
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            if isinstance(layer, SimpleEngram):
                out = layer(x, input_ids)
            else:
                out = layer(x)
            x = residual + out
            
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        return {'logits': logits, 'loss': loss}

# Scheduler
def get_wsd_lr(step, cfg):
    max_lr = cfg['lr_max']
    min_lr = cfg['lr_min']
    warmup = cfg['warmup_steps']
    decay_start = cfg['decay_start_step']
    max_steps = cfg['max_steps']
    
    if step < warmup: return max_lr * (step / warmup)
    if step < decay_start: return max_lr
    decay_steps = max_steps - decay_start
    progress = (step - decay_start) / decay_steps
    return max_lr - (max_lr - min_lr) * progress

# Train
def train():
    cfg = TRAIN_CONFIG
    wandb.init(project=cfg["project_name"], config=CONFIG_DICT)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    config = YaongiConfig(**CONFIG_DICT)
    model = HybridMoEEngram(config).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e9:.4f}B")
    
    teon_params = [p for p in model.parameters() if p.ndim >= 2]
    adam_params = [p for p in model.parameters() if p.ndim < 2]

    optim_teon = Teon(teon_params, lr=cfg['lr_max'])
    optim_adam = torch.optim.AdamW(adam_params, lr=0.001)
    
    # Dummy Data
    class DummyDataset(IterableDataset):
        def __init__(self, seq_len): self.seq_len = seq_len
        def __iter__(self):
            while True: yield torch.randint(0, 32000, (self.seq_len + 1,))
            
    loader = DataLoader(DummyDataset(cfg['max_seq_len']), batch_size=cfg['batch_size'])
    
    model.train()
    step = 0
    pbar = tqdm(total=cfg['max_steps'])
    
    for batch in loader:
        if step >= cfg['max_steps']: break
        
        lr = get_wsd_lr(step, cfg)
        for g in optim_teon.param_groups: g['lr'] = lr
        
        outputs = model(batch.to(device), labels=batch.to(device))
        loss = outputs['loss']
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim_teon.step()
        optim_adam.step()
        optim_teon.zero_grad()
        optim_adam.zero_grad()
        step += 1
        
        wandb.log({"train/loss": loss.item(), "train/lr": lr, "step": step})
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

if __name__ == "__main__":
    train()
