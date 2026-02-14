import torch
import sys
import os

# Add the directory containing the training script to path
sys.path.append(os.path.join(os.getcwd(), 'experiments', 'h100_moe_training'))

try:
    from train_h100_moe import HybridMoEEngram, CONFIG
    print("✅ Successfully imported HybridMoEEngram and CONFIG")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

def test_model_instantiation():
    print("🚀 Starting Model Instantiation Test...")
    
    # Use a smaller config for quick CPU test
    test_config = CONFIG['model'].copy()
    test_config['n_layers'] = 4 # Reduce layers
    test_config['num_experts'] = 4 # Reduce experts
    test_config['engram_layers'] = [2]
    
    try:
        model = HybridMoEEngram(test_config)
        print("✅ Model instantiated successfully")
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Total Parameters (Tiny Test Config): {total_params:,}")
        
        # Dummy forward pass
        input_ids = torch.randint(0, test_config['vocab_size'], (2, 32)) # Batch=2, Seq=32
        print("🔄 Running dummy forward pass...")
        
        output = model(input_ids)
        logits = output['logits']
        
        assert logits.shape == (2, 32, test_config['vocab_size']), f"Shape mismatch: {logits.shape}"
        print("✅ Forward pass successful! Output shape:", logits.shape)
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_instantiation()
