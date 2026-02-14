import os
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig, AutoConfig
from huggingface_hub import Repository, create_repo, upload_folder
import wandb
from tqdm import tqdm
import glob
import random
from huggingface_hub import login

# Auto-Login to HF if token is present
if "HF_TOKEN" in os.environ:
    print("🔑 Found HF_TOKEN, logging in...")
    login(token=os.environ["HF_TOKEN"])

# FP8 Imports
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    print("✅ Transformer Engine (FP8) Available")
    TE_AVAILABLE = True
except ImportError:
    print("⚠️ Transformer Engine not found. Falling back to sub-optimal standard Linear.")
    TE_AVAILABLE = False
    class TE_Mock:
        Linear = nn.Linear
    te = TE_Mock()

# --- Configuration (Nemotron 3 Nano + Teon Style) ---
CONFIG_DICT = {
    "model_type": "yaongi_nemotron",
    "n_layers": 52,
    "d_model": 2048,
    "d_state": 128,
    "vocab_size": 32768,
    "num_experts": 64,
    "top_k": 6,
    "n_heads": 32,
    "n_kv_heads": 8,
    "engram_layers": [2, 26],
    "engram_avg_pool": 2,
    "init_scale": 0.01,
    "layer_norm_epsilon": 1e-5,
    "tie_word_embeddings": False,
}

TRAIN_CONFIG = {
    "project_name": "Yaongi-Nemotron-Teon-H100",
    "batch_size": 128,      
    "grad_accum": 2,       
    "max_seq_len": 4096,
    "max_steps": 5000,
    "save_steps": 500,
    "push_to_hub": True, 
    "hf_repo_id": "gyung/Yaongi-Nemotron-4B-Teon", # User ID reflected
    "use_fp8": True,
    # WSD Scheduler
    "lr_max": 0.002,       
    "lr_min": 1e-5,
    "warmup_steps": 100,  
    "decay_start_step": 4000, 
}

# --- 1. Teon Optimizer (Tensorized Muon) ---
# Paper: https://arxiv.org/html/2601.23261v2
# Key Idea: Stack compatible 2D tensors into 3D and orthogonalize jointly.

def zeropower_via_newtonschulz5(G, steps=5):
    # Newton-Schulz iteration for 2D or 3D tensors
    # If 3D (Teon): [Batch, N, M] -> orthogonalize each N,M matrix in parallel
    assert len(G.shape) in (2, 3)
    
    # Constants for NS-5
    a, b, c = (3.4445, -4.7750,  2.0315)
    
    X = G.bfloat16()
    
    # Transpose if N > M to ensure X * X.T is the smaller dimension
    if G.ndim == 2:
        if G.size(0) > G.size(1): X = X.T
    else: # 3D
        if G.size(1) > G.size(2): X = X.transpose(1, 2)
        
    X = X / (X.norm(dim=-1, keepdim=True).norm(dim=-2, keepdim=True) + 1e-7) # Normalize roughly
    
    for _ in range(steps):
        if X.ndim == 2:
            A = X @ X.T
        else:
            A = torch.bmm(X, X.transpose(1, 2))
            
        B = b * A + c * torch.bmm(A, A) if X.ndim==3 else b * A + c * A @ A
        
        if X.ndim == 2:
            X = a * X + B @ X
        else:
            X = a * X + torch.bmm(B, X)
            
    if G.ndim == 2:
        if G.size(0) > G.size(1): X = X.T
    else:
        if G.size(1) > G.size(2): X = X.transpose(1, 2)
        
    return X

class Teon(torch.optim.Optimizer):
    """
    Teon Optimizer: Supports both standard Muon (2D) and Tensorized Muon (3D).
    Pass 'params' as a list of tensors. 
    To use Teon stacking, manually group parameters into a single stacked tensor view 
    or logic inside step (simplified here to auto-detect 3D grads if user provided).
    
    In this implementation, we rely on the Training Loop to provide stacked gradients
    or we treat standard parameters as standard Muon.
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
                
                # Standard Momentum
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf
                
                # Teon/Muon Orthogonalization
                if update.ndim >= 2:
                    # If 3D, it's a stacked expert tensor -> Teon Mode
                    # If 2D, it's standard Muon Mode
                    g_ortho = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    
                    # Scaling rule (paper suggests max(rows, cols)^0.5 usually)
                    if update.ndim == 2:
                        scale = max(1, p.size(0)/p.size(1))**0.5
                    else:
                        scale = max(1, p.size(1)/p.size(2))**0.5 # [Batch, Row, Col]
                        
                    p.data.add_(g_ortho, alpha=-lr * scale)
                else:
                    # Fallback for 1D (should use AdamW really, but safety check)
                    p.data.add_(update, alpha=-lr)

# --- 2. Model Components (Transformers Compatible) ---

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
    def __init__(self, d_model, d_state=128):
        super().__init__()
        try:
            from mamba_ssm import Mamba2
            self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            self.is_slow = False
        except ImportError:
            self.is_slow = True
            # CPU Fallback (Simple Linear Approximation for structure, NOT functional Mamba)
            # Real Mamba CPU inference requires a custom loop kernel, typically provided by mamba.cpp or specialized pytorch impl.
            # Here we provide a structural placeholder so loading doesn't crash.
            self.in_proj = nn.Linear(d_model, d_model*2, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            
    def forward(self, x):
        if not self.is_slow:
            return self.mamba(x)
        else:
            # Fake Forward for CPU compatibility testing / Loading
            return self.out_proj(F.silu(self.in_proj(x))[:, :, :x.shape[-1]])

class NemotronMoE(nn.Module):
    def __init__(self, d_model, num_experts=64, top_k=6):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False) 
        
        # Teon Optimization: We hold experts in a ModuleList for standard usage,
        # but for optimization, we will access them as a group.
        # Structure: Expert = Up(4D) -> Act -> Down(D)
        
        # Expert Up Projections
        self.experts_up = nn.ModuleList([
            te.Linear(d_model, d_model * 4, bias=False) for _ in range(num_experts)
        ])
        # Expert Down Projections
        self.experts_down = nn.ModuleList([
            te.Linear(d_model * 4, d_model, bias=False) for _ in range(num_experts)
        ])
        
        self.activation = SquaredReLU()
        self.shared_expert = nn.Sequential(
             te.Linear(d_model, d_model * 4, bias=False),
             SquaredReLU(),
             te.Linear(d_model * 4, d_model, bias=False)
        )

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        
        logits = self.router(x_flat)
        probs = torch.sigmoid(logits) # Sigmoid gating per Nemotron
        
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        # Nemotron uses sum of sigmoid outputs, not softmax, but typically normalized for stability
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-6)
        
        out = self.shared_expert(x_flat) 
        
        # Simple loop implementation (Fast enough for training with TE)
        # For Teon, this structure allows us to grab all self.experts_up parameters easily
        for k in range(self.top_k):
            expert_indices_k = topk_indices[:, k]
            expert_weights_k = topk_probs[:, k].unsqueeze(-1)
            
            for e in range(self.num_experts):
                 mask = (expert_indices_k == e)
                 if mask.any():
                     sub_x = x_flat[mask]
                     # Up -> Act -> Down
                     ex_out = self.experts_down[e](self.activation(self.experts_up[e](sub_x)))
                     out[mask] += expert_weights_k[mask] * ex_out
                     
        return out.view(B, T, D)

# ... (omitted Engram/GQAttention for brevity, assumption: same as before but inheriting PreTrainedModel) ...
# Re-implementing simplified versions for the full script context

class SimpleEngram(nn.Module):
    def __init__(self, d_model, vocab_size, n_gram=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.gate = te.Linear(d_model, d_model, bias=False)
    def forward(self, x, input_ids):
        # Dummy Logic for simplicity in this artifact
        return x

class GQAttention(nn.Module):
    def __init__(self, d_model, n_heads=32, n_kv_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = te.Linear(d_model, d_model, bias=False)
        self.k_proj = te.Linear(d_model, self.head_dim * n_kv_heads, bias=False)
        self.v_proj = te.Linear(d_model, self.head_dim * n_kv_heads, bias=False)
        self.o_proj = te.Linear(d_model, d_model, bias=False)
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
                block = SimpleEngram(d_model, 200000) # Simplified
            elif i % 6 == 5:
                block = GQAttention(d_model, config.n_heads, config.n_kv_heads)
            else:
                block = nn.Sequential(
                    Mamba2Block(d_model, config.d_state),
                    NemotronMoE(d_model, config.num_experts, config.top_k)
                )
            self.layers.append(block)
            self.norms.append(RMSNorm(d_model))
            
        self.final_norm = RMSNorm(d_model)
        self.lm_head = te.Linear(d_model, config.vocab_size, bias=False) 
        
        self.post_init()
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, te.Linear)):
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
            x = residual + out # Simple residual
            
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        return {'logits': logits, 'loss': loss}

# --- 3. WSD Scheduler ---
def get_wsd_lr(step, cfg):
    max_lr = cfg['lr_max']
    min_lr = cfg['lr_min']
    warmup = cfg['warmup_steps']
    decay_start = cfg['decay_start_step']
    max_steps = cfg['max_steps']
    
    if step < warmup:
        return max_lr * (step / warmup)
    if step < decay_start:
        return max_lr
    decay_steps = max_steps - decay_start
    progress = (step - decay_start) / decay_steps
    return max_lr - (max_lr - min_lr) * progress

# --- 4. Main Training ---

def train():
    cfg = TRAIN_CONFIG
    wandb.init(project=cfg["project_name"], config=CONFIG_DICT)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    config = YaongiConfig(**CONFIG_DICT)
    model = HybridMoEEngram(config).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    
    # --- Teon Grouping Logic ---
    # We group MoE experts into stacked tensors for Teon optimization
    teon_params = [] # List of stacked tensors (or single 2D)
    adam_params = []
    
    # 1. Collect MoE Experts for Stacking
    for name, module in model.named_modules():
        if isinstance(module, NemotronMoE):
            # Stack Up Projections: [64, D_in, D_out]
            up_weights = [m.weight for m in module.experts_up]
            # Stack Down Projections
            down_weights = [m.weight for m in module.experts_down]
            
            # Note: PyTorch Optimizer usually expects leaf parameters.
            # Teon trick: We can't actually 'stack' parameters into a new parameter list unless we reshape the model.
            # Instead, for this implementation, we will pass them as individual 2D tensors to Teon,
            # BUT we modify Teon step to support `params` being a list of list??
            # Standard Pytorch implementation limitation: We can't optimize a 'virtual' stacked tensor easily.
            # COMPROMISE for Stability: 
            # We will use standard Muon (2D) for now as rewriting the full backward graph for stacked weights 
            # requires modifying the Model definition to use a single 3D weight Parameter [Experts, In, Out].
            # Converting ModuleList of Linears to a single Batched Linear is the correct Teon approach.
            pass

    # REVISION: To enable True Teon, we must treat params individually but conceptually optimized.
    # Since we can't easily refactor the whole model to 3D weights in this single file without breaking load_state,
    # We will use the Muon optimizer as defined but apply it to the 2D weights.
    # The Teon paper's main benefit is inter-layer, which we approximate by using Muon on High-Rank MoE matrices.
    
    for p in model.parameters():
        if p.ndim >= 2 and p.size(0) > 32 and p.size(1) > 32:
            teon_params.append(p)
        else:
            adam_params.append(p)

    optim_teon = Teon(teon_params, lr=cfg['lr_max'])
    optim_adam = torch.optim.AdamW(adam_params, lr=0.001)
    
    # Load Custom Tokenizer
    tokenizer_path = "./custom_tokenizer"
    if os.path.exists(tokenizer_path):
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        print(f"✅ Loaded Custom Tokenizer (vocab={tokenizer.vocab_size})")
    else:
        print("⚠️ Custom tokenizer not found. Using fallback (bert-base). This may cause mismatches!")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Vocab Size Check
    if tokenizer.vocab_size != config.vocab_size:
        print(f"⚠️ WARNING: Tokenizer vocab ({tokenizer.vocab_size}) != Model config ({config.vocab_size})")
        # Adjusting model config if explicitly allowed (or just warning)
    
    dataset = JsonlDataset(data_dir="./data_cache", tokenizer=tokenizer, seq_len=cfg['max_seq_len'])
    loader = DataLoader(dataset, batch_size=cfg['batch_size'])
    
    model.train()
    step = 0
    pbar = tqdm(total=cfg['max_steps'])

    accum_steps = cfg['grad_accum']
    optimizer_step = 0
    fp8_recipe = te.common.recipe.Format.HYBRID
    
    for batch in loader:
        if optimizer_step >= cfg['max_steps']: break
        
        lr = get_wsd_lr(optimizer_step, cfg)
        for param_group in optim_teon.param_groups: param_group['lr'] = lr
        
        with te.fp8_autocast(enabled=TE_AVAILABLE, fp8_recipe=DelayedScaling(fp8_format=fp8_recipe)):
             outputs = model(batch.to(device), labels=batch.to(device))
             loss = outputs['loss'] / accum_steps
            
        loss.backward()
        step += 1
        
        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_teon.step()
            optim_adam.step()
            optim_teon.zero_grad()
            optim_adam.zero_grad()
            optimizer_step += 1
            
            wandb.log({"train/loss": loss.item() * accum_steps, "train/lr": lr, "step": optimizer_step})
            pbar.update(1)
            
            if optimizer_step % cfg['save_steps'] == 0:
                save_path = f"./checkpoints/checkpoint_{optimizer_step}"
                model.save_pretrained(save_path)
                # tokenizer is already loaded from custom_tokenizer at start of train()
                tokenizer.save_pretrained(save_path)
                
                if cfg['push_to_hub']:
                    try:
                        upload_folder(
                            folder_path=save_path,
                            repo_id=cfg['hf_repo_id'],
                            commit_message=f"Step {optimizer_step}"
                        )
                        print(f"🚀 Pushed to Hub: {cfg['hf_repo_id']}")
                    except Exception as e:
                        print(f"⚠️ Hub Upload Failed: {e}")

class JsonlDataset(IterableDataset):
    def __init__(self, data_dir, tokenizer, seq_len=4096):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.files = glob.glob(f"{data_dir}/**/*.jsonl", recursive=True)
        if not self.files:
            print(f"⚠️ No data found in {data_dir}. Generating random data for test.")
            self.use_dummy = True
        else:
            self.use_dummy = False
            print(f"📂 Found {len(self.files)} data files for training.")

    def __iter__(self):
        if self.use_dummy:
            while True:
                yield torch.randint(0, self.tokenizer.vocab_size, (self.seq_len + 1,))
        else:
            while True:
                # Shuffle files for better mixing
                import random
                random.shuffle(self.files)
                for file_path in self.files:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            buffer = []
                            for line in f:
                                try:
                                    obj = json.loads(line)
                                    text = obj.get("text", "")
                                    if not text: continue
                                    
                                    # Tokenize
                                    tokens = self.tokenizer.encode(text, add_special_tokens=True)
                                    buffer.extend(tokens)
                                    
                                    # Yield chunks
                                    while len(buffer) >= self.seq_len + 1:
                                        chunk = buffer[:self.seq_len + 1]
                                        yield torch.tensor(chunk, dtype=torch.long)
                                        buffer = buffer[self.seq_len + 1:]
                                except:
                                    continue
                    except Exception as e:
                        print(f"⚠️ Error reading {file_path}: {e}")
                        continue

if __name__ == "__main__":
    train()
