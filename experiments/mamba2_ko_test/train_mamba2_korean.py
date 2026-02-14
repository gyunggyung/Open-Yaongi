
# train_mamba2_korean.py
# Mamba-2 (Pure PyTorch) + Muon Optimizer + Custom BPE Tokenizer
# Designed for fast training on Google Colab (T4 GPU)

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tqdm.auto import tqdm

# ==========================================
# 0. Configuration
# ==========================================
class Config:
    # Model Architecture (Nano Mamba-2)
    vocab_size = 8000   # Reduced vocab for speed
    d_model = 384       # Dimension
    n_layers = 6        # Number of layers
    n_heads = 6         # Number of heads
    d_head = 64         # Head dimension
    d_state = 16        # SSM state dimension
    
    # Training Hyperparameters
    batch_size = 64     # Reduced for safety
    seq_len = 256
    lr_muon = 0.005     # Lowered significantly (0.02 -> 0.005)
    lr_adam = 0.0005    # Lowered standard LR (0.001 -> 0.0005)
    epochs = 3
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()
print(f"Running on: {config.device}")
# ==========================================
# 1. Muon Optimizer (The "AdamW Killer")
# ==========================================
def zeropower_via_newtonschulz5(G, steps=5):
    """
    Muon's core: Newton-Schulz iteration for orthogonalization.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.float() # Force FP32 for T4 (Turing does not support BF16 well)
    if G.size(0) > G.size(1):
        X = X.T
    
    # Scale to ensure spectral norm < \sqrt{4/3} approx
    X = X / (X.norm() + 1e-7)
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """
    Muon Optimizer for 2D matrices (Linear layers).
    Use AdamW for 1D tensors (Biases, LayerNorms, Embeddings).
    """
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
                
                # Init momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                
                # Muon Update: Orthogonalize the update matrix
                if g.ndim == 2 and g.size(0) > 32 and g.size(1) > 32:
                    g_ortho = zeropower_via_newtonschulz5(g, steps=ns_steps)
                    scale = max(1, p.size(0)/p.size(1))**0.5
                    p.data.add_(g_ortho, alpha=-lr * scale)
                else:
                    p.data.add_(g, alpha=-lr)

# ==========================================
# 2. Tokenizer & Dataset
# ==========================================
def train_custom_tokenizer(dataset, vocab_size=8000):
    print(f"Training Custom BPE Tokenizer (Vocab: {vocab_size})...")
    
    # Dump dataset to text file for tokenizer training
    if not os.path.exists("train_corpus.txt"):
        with open("train_corpus.txt", "w", encoding="utf-8") as f:
            for item in tqdm(dataset, desc="Exporting text"):
                # Combining instruction and output for causal language modeling
                text = f"Q:{item['instruction']} A:{item['output']}\n"
                f.write(text)
    
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<s>", "</s>"],
        min_frequency=2
    )
    
    tokenizer.train(["train_corpus.txt"], trainer)
    return tokenizer

class WrappedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.token_to_id("<pad>")
        self.eos_token_id = tokenizer.token_to_id("</s>")
        self.vocab_size = tokenizer.get_vocab_size()
    
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

class FastKoDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_len):
        self.samples = []
        print("Tokenizing dataset...")
        for item in tqdm(dataset):
            text = f"Q:{item['instruction']} A:{item['output']}</s>"
            ids = tokenizer.encode(text)
            
            # Simple truncation/padding
            if len(ids) > seq_len: 
                ids = ids[:seq_len]
            else: 
                ids = ids + [tokenizer.pad_token_id]*(seq_len - len(ids))
            
            self.samples.append(ids)
            
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return torch.tensor(self.samples[idx], dtype=torch.long)

# ==========================================
# 3. Model Architecture (Mamba-2)
# ==========================================
class Mamba2Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_state = cfg.d_state
        self.d_inner = cfg.n_heads * cfg.d_head
        
        # Combined projection for Z, X, B, C, A
        # Dimensions: 
        # Z: d_inner
        # X: d_inner
        # B: n_heads * d_state
        # C: n_heads * d_state
        # A: n_heads
        proj_dim = (2 * self.d_inner) + (2 * self.n_heads * self.d_state) + self.n_heads
        self.in_proj = nn.Linear(self.d_model, proj_dim, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_inner)
        
        # A parameter (log space)
        self.A_log = nn.Parameter(torch.randn(self.n_heads))
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.n_heads, self.d_head))

    def forward(self, x):
        B, L, _ = x.shape
        zxbca = self.in_proj(x)
        
        x_t, z_t, b_t, c_t, a_t_logits = zxbca.split([
            self.d_inner, self.d_inner,
            self.n_heads * self.d_state, self.n_heads * self.d_state,
            self.n_heads
        ], dim=-1)
        
        # Reshape
        x_t = x_t.view(B, L, self.n_heads, self.d_head)
        z_t = z_t.view(B, L, self.n_heads, self.d_head)
        b_t = b_t.view(B, L, self.n_heads, self.d_state)
        c_t = c_t.view(B, L, self.n_heads, self.d_state)
        
        # Decay: exp(-exp(A_log)) -> range (0, 1)
        decay = -torch.exp(a_t_logits.float()) 
        decay = torch.exp(decay) # (B, L, H)
        
        # SSD Recurrence (Parallel/Chunk scan is ideal, loop for simplicity on CPU/T4)
        states = torch.zeros(B, self.n_heads, self.d_head, self.d_state, device=x.device)
        y_list = []
        
        for t in range(L):
            xt_step = x_t[:, t] # (B, H, P)
            bt_step = b_t[:, t] # (B, H, N)
            
            # Broadcast decay
            decay_step = decay[:, t].view(B, self.n_heads, 1, 1)
            states = states * decay_step
            
            # Outer product update: X * B^T
            update = torch.matmul(xt_step.unsqueeze(-1), bt_step.unsqueeze(-2))
            states = states + update
            
            # Output: H * C
            ct_step = c_t[:, t].unsqueeze(-1) # (B, H, N, 1)
            yt_step = torch.matmul(states, ct_step).squeeze(-1) # (B, H, P)
            
            # D skip connection
            yt_step = yt_step + xt_step * self.D.view(1, self.n_heads, self.d_head)
            
            y_list.append(yt_step)
            
        y = torch.stack(y_list, dim=1).view(B, L, self.d_inner)
        
        # Gating (Swish)
        y = y * F.silu(z_t.view(B, L, -1))
        y = self.norm(y)
        return self.out_proj(y)

class NanoMamba2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Mamba2Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight # Weight Tying

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    print(f"Running on {config.device}")
    
    # 1. Load Data
    print("Loading Dataset (beomi/KoAlpaca-v1.1a)...")
    ds_raw = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
    
    # 2. Train/Load Tokenizer
    custom_tokenizer = train_custom_tokenizer(ds_raw, vocab_size=config.vocab_size)
    tokenizer = WrappedTokenizer(custom_tokenizer)
    print(f"Tokenizer Ready. Vocab Size: {tokenizer.vocab_size}")
    config.vocab_size = tokenizer.vocab_size
    
    # 3. Prepare DataLoader
    train_ds = FastKoDataset(ds_raw, tokenizer, config.seq_len)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    
    # 4. Initialize Model
    model = NanoMamba2(config).to(config.device)
    print(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 5. Optimizers (Muon + AdamW)
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and p.size(0) > 32 and p.size(1) > 32:
            muon_params.append(p)
        else:
            adam_params.append(p)
            
    optim_muon = Muon(muon_params, lr=config.lr_muon)
    optim_adam = torch.optim.AdamW(adam_params, lr=config.lr_adam)
    print(f"Optimizer: Muon ({len(muon_params)} params) + AdamW ({len(adam_params)} params)")
    
    # 6. Training Loop
    model.train()
    for epoch in range(config.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for step, inputs in enumerate(pbar):
            inputs = inputs.to(config.device)
            input_ids = inputs[:, :-1]
            target_ids = inputs[:, 1:]
            
            optim_muon.zero_grad()
            optim_adam.zero_grad()
            
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), 
                                   target_ids.reshape(-1), 
                                   ignore_index=tokenizer.pad_token_id)
            # NaN Check
            if torch.isnan(loss):
                print("!! LOSS IS NAN !! Stopping.")
                break
            
            loss.backward()
            
            # Gradient Clipping (Essential for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optim_muon.step()
            optim_adam.step()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Simple Generation Preview
            if step % 100 == 0:
                with torch.no_grad():
                    ctx = torch.tensor([tokenizer.encode("Q:대한민국의 수도는? A:")]).to(config.device)
                    for _ in range(20):
                        logits = model(ctx)
                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                        ctx = torch.cat([ctx, next_token], dim=1)
                        if next_token.item() == tokenizer.eos_token_id: break
                    print(f"\n[Preview] {tokenizer.decode(ctx[0].tolist())}\n")

    print("Training Complete.")

if __name__ == "__main__":
    main()
